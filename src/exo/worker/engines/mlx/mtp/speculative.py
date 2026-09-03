"""exo-side glue for MTP speculative decoding (stages 0 and 1).

Everything here is inert unless the operator sets EXO_MTP=1 — drafting is
default-OFF while the integration soaks. When enabled, a request drafts
only if ALL of these hold:

  - the topology is single-node (stage 0) or PIPELINE-sharded (stage 1);
    tensor sharding is still refused — see `plan_mtp` below;
  - the request carries no images (the vision path patches embed_tokens
    around the trunk forward; untested with the capture wrapper);
  - the model's family is registered and its artifact dir contains the
    family's sidecar (e.g. mtp-head-q6.safetensors).

Anything failing above falls back to the stock decode path silently (one
log line, no error): the head is a bonus, never a dependency.

Stage 1 in one paragraph. The decode loop is not forked; it runs on every
pipeline rank, parameterized by a Coordinator (pipeline.py). The head loads
on the LAST rank only — during prefill that is the one rank whose capture
point holds the trunk's true final activation, because PipelineLastLayer's
all_gather is gated off while is_prefill is set — and since the head samples
the draft, ALL sampling for the request moves to that rank too, or the ranks'
RNG streams diverge on the first draft. Two small broadcasts per step carry
the tokens and the accept/reject verdict to the ranks that do not draft.

A consequence worth stating plainly, because it is what makes the gate safe
to widen: on a non-last rank the head is simply absent (`plan.head is None`),
so stage 1 costs those nodes no extra memory at all.

Known trade, deliberate for stage 0: an MTP request bypasses the KV
prefix pool. The head must be seeded with per-position hidden states
from position 0 (see loop.py's alignment docstring), which a reused
prefix does not provide; vqlab/serve.py makes the same call. Pairing a
head cache with each pooled prefix is the stage-0.5 design that lifts
this — until then, long-prompt agent loops should weigh 1.5-1.8x decode
against losing incremental prefill.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Generator

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.download.download_utils import build_model_path
from exo.worker.engines.mlx.types import Model
from exo.worker.runner.bootstrap import logger

from .loop import load_mtp_head, mtp_stream_generate
from .pipeline import Coordinator, is_pipeline_model, make_coordinator
from .registry import resolve

_HEAD_CACHE: dict[str, Any] = {}
_HEAD_FAILED: set[str] = set()


def mtp_enabled() -> bool:
    return os.environ.get("EXO_MTP") == "1"


@dataclass
class MTPPlan:
    """How this rank participates in a speculative request.

    `head` is None on every rank but the last: those ranks run the loop for
    its side effects on their own KV caches and take every token from the
    control broadcasts. `stage` is for logging only.
    """

    coord: Coordinator
    head: Any | None
    stage: int

    @property
    def drafts(self) -> bool:
        return self.head is not None


def plan_mtp(
    model: Model,
    model_id: str | None,
    group: mx.distributed.Group | None,
    has_vision: bool,
) -> MTPPlan | None:
    """The speculative plan for this request, or None for the stock path.

    The topology gate, in one place:

      single node (no group, or world size 1)   -> stage 0
      multi-node AND pipeline-sharded           -> stage 1
      multi-node, NOT pipeline-sharded (tensor) -> refused

    The pipeline test asks the MODEL, not the placement metadata
    (`is_pipeline_model` looks for the wrapper layers `pipeline_auto_parallel`
    installs). That is deliberate: the property stage 1 depends on is the
    all_gather inside PipelineLastLayer, and the wrappers ARE that property.
    Metadata would only tell us how the instance was meant to be built.

    Tensor sharding stays refused because its ranks each hold a slice of every
    layer, so no rank ever holds a whole final hidden state to draft from; the
    head would need its own tensor-parallel split. That is not a gate that can
    be widened by relaxing a condition — it needs a different head.
    """
    if not mtp_enabled() or not model_id:
        return None
    if has_vision:
        return None

    multi_node = group is not None and group.size() > 1
    if multi_node and not is_pipeline_model(model):
        logger.info(
            "EXO_MTP=1 but this instance is not pipeline-sharded "
            "(tensor-parallel drafting is out of scope); decoding "
            "without drafting"
        )
        return None

    coord = make_coordinator(group)
    stage = 1 if multi_node else 0

    # A non-last pipeline rank never loads a head — it has nothing to draft
    # from — but it must still take the speculative path, or it would run a
    # different number of forwards than its peers and the pipeline would
    # wedge on the first mismatched send.
    head = _load_head(model, model_id) if coord.is_last else None

    # AGREE on that, do not each decide it. Head loading can fail on one node
    # and succeed on another — a half-downloaded sidecar, a stale registry
    # entry against a differently-pinned mlx-lm — and ranks that disagreed
    # about whether this request is speculative would run different forward
    # counts and deadlock in recv_like. So the last rank's answer is the
    # instance's answer. One broadcast per request; the per-step cost is
    # unaffected.
    #
    # Everything checked ABOVE this line is uniform across ranks by
    # construction (same task, same model) with one operator-owned
    # exception: EXO_MTP itself must be set identically on every node of the
    # instance. Setting it on some nodes only is a misconfiguration that
    # stage 0 shares, and it is a hang, not a wrong answer.
    verdict = coord.broadcast(mx.array([1 if head is not None else 0], dtype=mx.int32))
    if int(verdict[0].item()) != 1:
        return None

    logger.info(f"MTP stage {stage} engaged for {model_id} ({coord!r})")
    return MTPPlan(coord=coord, head=head, stage=stage)


def _load_head(model: Model, model_id: str) -> Any | None:
    if model_id in _HEAD_CACHE:
        return _HEAD_CACHE[model_id]
    if model_id in _HEAD_FAILED:
        return None
    try:
        spec = resolve(model)
        sidecar = build_model_path(model_id) / spec.sidecar_name
        if not sidecar.exists():
            logger.info(
                f"EXO_MTP=1 but no sidecar {spec.sidecar_name} in the "
                f"{model_id} artifact; decoding without drafting"
            )
            _HEAD_FAILED.add(model_id)
            return None
        before = mx.get_active_memory()
        head, spec = load_mtp_head(model, sidecar=sidecar)
        logger.info(
            f"MTP head ({spec.name}) loaded for {model_id}: "
            f"{(mx.get_active_memory() - before) / 2**30:.2f} GiB resident"
        )
        _HEAD_CACHE[model_id] = head
        return head
    except Exception:
        # A malformed sidecar or a stale registry entry must not take
        # serving down; the stock path is always available.
        logger.opt(exception=True).warning(
            f"MTP head load failed for {model_id}; decoding without drafting"
        )
        _HEAD_FAILED.add(model_id)
        return None


def _pipeline_prefill_ctx(model: Model) -> Callable[[], Any]:
    """Factory for the context the MTP loop wraps its prompt chunks in.

    `is_prefill=True` turns off PipelineLastLayer's per-forward all_gather,
    which is what makes prompt processing affordable: without it every chunk
    would gather a [1, chunk, hidden] tensor to every rank. The last rank —
    the only one that seeds a head — has the true activation either way,
    because it IS the end of the pipeline.

    Restored in a finally, so a cancelled or failed prefill cannot leave the
    instance stuck in prefill mode for the next request.
    """

    @contextlib.contextmanager
    def ctx():
        from exo.worker.engines.mlx.auto_parallel import set_pipeline_prefill

        set_pipeline_prefill(model, is_prefill=True)
        try:
            yield
        finally:
            set_pipeline_prefill(model, is_prefill=False)

    return ctx


@dataclass
class SpecResponse:
    """Duck-compatible with the mlx_lm GenerationResponse fields exo reads."""

    text: str
    token: int
    finish_reason: str | None
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    logprobs: mx.array | None = None
    from_draft: bool = False
    acceptance: float = 0.0


def mtp_responses(
    model: Model,
    tokenizer: TokenizerWrapper,
    prompt_tokens: mx.array,
    plan: MTPPlan,
    *,
    max_tokens: int,
    temp: float,
    top_p: float,
    min_p: float,
    top_k: int,
    logits_processors: list[Callable[..., Any]] | None,
    prefill_step_size: int | None,
    want_logprobs: bool = False,
) -> Generator[SpecResponse, None, None]:
    """mtp_stream_generate adapted to the stream exo's decode loop consumes.

    Two impedance fixes:
      - MTPResponse has no prompt stats; the loop owns its own prefill, so
        time-to-first-token stands in for prefill wall time (it includes
        one draft+verify step — a slight understatement of prompt_tps,
        never an overstatement).
      - The loop can yield a trailing detokenizer flush AFTER the token
        carrying finish_reason; exo's consumer breaks at the first
        finish_reason, so finish is held back and re-attached to the
        genuinely last response.
    """
    n_prompt = int(prompt_tokens.shape[-1])
    start = time.perf_counter()
    first_token_at: float | None = None
    pending: SpecResponse | None = None
    pending_finish: str | None = None
    final_acceptance = 0.0

    for r in mtp_stream_generate(
        model,
        tokenizer,
        prompt_tokens,
        plan.head,
        coord=plan.coord,
        prefill_ctx=_pipeline_prefill_ctx(model) if plan.stage else None,
        max_tokens=max_tokens,
        temp=temp,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        logits_processors=logits_processors,
        prefill_step_size=prefill_step_size or 2048,
        want_logprobs=want_logprobs,
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        prompt_time = (first_token_at - start) or 1e-9
        final_acceptance = r.acceptance
        resp = SpecResponse(
            text=r.text,
            token=r.token,
            logprobs=r.logprobs,
            finish_reason=None,
            prompt_tokens=n_prompt,
            prompt_tps=n_prompt / prompt_time,
            generation_tokens=r.generation_tokens,
            generation_tps=r.generation_tps,
            peak_memory=r.peak_memory,
            from_draft=r.from_draft,
            acceptance=r.acceptance,
        )
        if pending is not None:
            yield pending
        pending = resp
        pending_finish = r.finish_reason
    if pending is not None:
        pending.finish_reason = pending_finish or "stop"
        logger.info(
            f"MTP decode done: {pending.generation_tokens} tokens at "
            f"{pending.generation_tps:.2f} tok/s, acceptance "
            f"{final_acceptance:.3f}"
        )
        yield pending
