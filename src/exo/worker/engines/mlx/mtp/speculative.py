"""exo-side glue for MTP speculative decoding (stage 0: single node).

Everything here is inert unless the operator sets EXO_MTP=1 — drafting is
default-OFF while the integration soaks. When enabled, a request drafts
only if ALL of these hold:

  - the instance is a single node (pipeline verify is stage 1);
  - the request carries no images (the vision path patches embed_tokens
    around the trunk forward; untested with the capture wrapper);
  - the model's family is registered and its artifact dir contains the
    family's sidecar (e.g. mtp-head-q6.safetensors).

Anything failing above falls back to the stock decode path silently (one
log line, no error): the head is a bonus, never a dependency.

Known trade, deliberate for stage 0: an MTP request bypasses the KV
prefix pool. The head must be seeded with per-position hidden states
from position 0 (see loop.py's alignment docstring), which a reused
prefix does not provide; vqlab/serve.py makes the same call. Pairing a
head cache with each pooled prefix is the stage-0.5 design that lifts
this — until then, long-prompt agent loops should weigh 1.5-1.8x decode
against losing incremental prefill.
"""

from __future__ import annotations

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
from .registry import resolve

_HEAD_CACHE: dict[str, Any] = {}
_HEAD_FAILED: set[str] = set()


def mtp_enabled() -> bool:
    return os.environ.get("EXO_MTP") == "1"


def maybe_mtp_head(
    model: Model,
    model_id: str | None,
    group: mx.distributed.Group | None,
    has_vision: bool,
) -> Any | None:
    """The drafting head for this request, or None for the stock path."""
    if not mtp_enabled() or not model_id:
        return None
    if group is not None and group.size() > 1:
        return None  # stage 1: verify through the pipeline
    if has_vision:
        return None
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


@dataclass
class SpecResponse:
    """Duck-compatible with the mlx_lm GenerationResponse fields exo reads."""

    text: str
    finish_reason: str | None
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    from_draft: bool = False
    acceptance: float = 0.0


def mtp_responses(
    model: Model,
    tokenizer: TokenizerWrapper,
    prompt_tokens: mx.array,
    head: Any,
    *,
    max_tokens: int,
    temp: float,
    top_p: float,
    min_p: float,
    top_k: int,
    logits_processors: list[Callable[..., Any]] | None,
    prefill_step_size: int | None,
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
        head,
        max_tokens=max_tokens,
        temp=temp,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        logits_processors=logits_processors,
        prefill_step_size=prefill_step_size or 2048,
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        prompt_time = (first_token_at - start) or 1e-9
        final_acceptance = r.acceptance
        resp = SpecResponse(
            text=r.text,
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
