"""The batch engine's drafting path: ExoBatchGenerator's contract, backed by
mtp/batch_loop.py instead of mlx-lm's BatchGenerator.

Same `submit / step / cancel / close / has_work` surface and the same
per-token response building (`emit_token`), so the runner's BatchGenerator
wrapper cannot tell the two apart. What differs is underneath: every row
verifies one drafted token per step, and a request is prefilled at
`submit` with the head seeded over its prompt (batch_loop.admit) rather
than handed to mlx-lm's prompt batch.

Two trades this path makes, both inherited from the sequential MTP path:

  - KV prefix pool, with the head's cache stored BESIDE the trunk's in the
    same pool entry (entry = trunk caches + [head cache]). The head is one
    row per committed token, so the pool trims both together and a restored
    prefix hands back a head that is exactly as far along as the trunk; only
    the suffix is prefilled and the head is seeded over it. A row that did
    not draft (vision) stores a trunk-only entry, which a drafting request
    cannot use (no head history) and prefills from scratch.
    A drafting row therefore always prefills its whole prompt.
  - a request with images decodes without drafting (batch_loop's non-
    drafting row): the head's seeding calls embed_tokens over the prompt,
    and doing so under the vision embedding patch would feed it garbage.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import mlx.core as mx
from mlx_lm.generate import generation_stream
from mlx_lm.sample_utils import make_logits_processors
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.api.types import TopLogprobItem
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    encode_prompt,
    has_non_kv_caches,
    snapshot_ssm_states,
)
from exo.worker.engines.mlx.constants import (
    DEFAULT_TOP_LOGPROBS,
    MAX_TOKENS,
    prefill_step_size_for,
)
from exo.worker.engines.mlx.generator.batch_generate import (
    _MIN_PREFIX_HIT_RATIO_TO_UPDATE,
    _EngineTask,
    emit_token,
)
from exo.worker.engines.mlx.generator.generate import (
    ban_token_ids,
    eos_ids_from_tokenizer,
    extract_top_logprobs,
    patch_embed_tokens,
)
from exo.worker.engines.mlx.mtp.batch_loop import MTPBatch, Row, RowParams, admit
from exo.worker.engines.mlx.mtp.capture import capture_input
from exo.worker.engines.mlx.mtp.registry import resolve
from exo.worker.engines.mlx.mtp.sampling import make_distribution
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.utils_mlx import (
    fix_unmatched_think_end_tokens,
    system_prompt_token_count,
)
from exo.worker.engines.mlx.vision import (
    MediaRegion,
    VisionProcessor,
    VisionResult,
    prepare_vision,
)
from exo.worker.runner.bootstrap import logger


def _prefill_step_size(model_id: str | None) -> int:
    """Same precedence as generator/generate.py's prefill: the ring-wide env
    override, then the per-family table, then 4096."""
    env = os.getenv("EXO_PREFILL_STEP_SIZE")
    if env is not None:
        return int(env)
    return prefill_step_size_for(model_id) or 4096


def split_pool_entry(
    entry: list, n_trunk: int, *, drafts: bool, hit_len: int
) -> tuple[list, Any, int]:
    """(trunk caches, head cache or None, usable prefix length) from a pool
    entry restored at `hit_len`. A drafting row needs a head cache sitting at
    exactly hit_len; an entry without one (stored by a non-drafting row) or
    with one elsewhere is unusable for drafting -> prefix length 0, fresh
    caches. A non-drafting row uses the trunk part of any entry."""
    trunk, extra = list(entry[:n_trunk]), list(entry[n_trunk:])
    if hit_len <= 0:
        return trunk, None, 0
    if not drafts:
        return trunk, None, hit_len
    if extra and int(getattr(extra[0], "offset", -1) or 0) == hit_len:
        return trunk, extra[0], hit_len
    return [], None, 0


@dataclass(eq=False)
class ExoMTPBatchGenerator:
    model: Model
    tokenizer: TokenizerWrapper
    head: Any
    vision_processor: VisionProcessor | None = None
    kv_prefix_cache: KVPrefixCache | None = None

    _batch: MTPBatch = field(init=False)
    _pending: list[Row] = field(default_factory=list, init=False)
    _active_tasks: dict[int, _EngineTask] = field(default_factory=dict, init=False)
    _uid_count: int = field(default=0, init=False)
    _stack: contextlib.ExitStack = field(init=False)
    _make_draft_cache: Callable[[], Any] = field(init=False)

    def __post_init__(self) -> None:
        spec = resolve(self.model)
        arch = spec.arch_module(self.model)
        core = getattr(getattr(self.model, "language_model", self.model), "model", None)
        if core is None:
            raise RuntimeError(
                f"{type(self.model).__name__} exposes neither `.model` nor "
                f"`.language_model.model`; no capture point for the MTP head"
            )
        # The capture wrapper stays installed for the engine's lifetime (the
        # sequential path installs it per request); `close` removes it.
        self._stack = contextlib.ExitStack()
        get_h = self._stack.enter_context(capture_input(core, spec.capture))
        if hasattr(self.head, "make_draft_cache"):
            self._make_draft_cache = self.head.make_draft_cache
        else:
            self._make_draft_cache = lambda: spec.make_draft_cache(arch)
        self._batch = MTPBatch(
            self.model,
            self.head,
            get_h,
            copy_caches=spec.cache_semantics != "reassign",
        )
        self._eos = set(eos_ids_from_tokenizer(self.tokenizer))
        self._step_count = 0
        self._n_trunk = len(self.model.make_cache())
        logger.info(f"batch engine drafting with the {spec.name} MTP head")

    @property
    def has_work(self) -> bool:
        return bool(self._active_tasks) or bool(self._pending) or len(self._batch) > 0

    @property
    def batch_size(self) -> int:
        return len(self._batch)

    def submit(
        self,
        task_params: TextGenerationTaskParams,
        prompt: str,
        on_prefill_progress: Callable[[int, int], None] | None = None,
        distributed_prompt_progress_callback: Callable[[], None] | None = None,
        on_generation_token: Callable[[], None] | None = None,
    ) -> int:
        all_prompt_tokens = encode_prompt(self.tokenizer, prompt)
        all_prompt_tokens = fix_unmatched_think_end_tokens(
            all_prompt_tokens, self.tokenizer
        )

        vision: VisionResult | None = None
        media_regions: list[MediaRegion] = []
        if self.vision_processor is not None:
            try:
                vision = prepare_vision(
                    images=task_params.images,
                    chat_template_messages=task_params.chat_template_messages,
                    vision_processor=self.vision_processor,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    model_id=task_params.model,
                    task_params=task_params,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Vision processing failed, falling back to text-only"
                )
        if vision is not None:
            all_prompt_tokens = vision.prompt_tokens
            media_regions = vision.media_regions

        seed = task_params.seed if task_params.seed is not None else 42
        mx.random.seed(seed)

        temp = task_params.temperature if task_params.temperature is not None else 0.7
        dist = make_distribution(
            temp=temp,
            top_p=task_params.top_p if task_params.top_p is not None else 1.0,
            min_p=task_params.min_p if task_params.min_p is not None else 0.05,
            top_k=task_params.top_k if task_params.top_k is not None else 0,
        )
        processors: list[Callable[[mx.array, mx.array], mx.array]] = (
            make_logits_processors(
                repetition_penalty=task_params.repetition_penalty,
                repetition_context_size=task_params.repetition_context_size
                if task_params.repetition_context_size is not None
                else 20,
                presence_penalty=task_params.presence_penalty,
                frequency_penalty=task_params.frequency_penalty,
            )
        )
        if task_params.bench:
            processors = [ban_token_ids(list(self._eos))] + processors

        params = RowParams(
            max_tokens=task_params.max_output_tokens or MAX_TOKENS,
            dist=dist,
            processors=processors,
            eos=self._eos,
            drafts=vision is None,
        )

        n_prompt = len(all_prompt_tokens)

        # Prefix pool: same lookup as the plain engine, plus the head cache
        # riding in the entry (see the module docstring).
        cache: list | None = None
        hcache: Any = None
        hit_len = 0
        matched_index: int | None = None
        is_exact = False
        pool = self.kv_prefix_cache
        if pool is not None and task_params.use_prefix_cache:
            entry, remaining, matched_index, is_exact = pool.get_kv_cache(
                self.model, all_prompt_tokens, media_regions=media_regions
            )
            raw_hit = n_prompt - len(remaining)
            cache, hcache, hit_len = split_pool_entry(
                entry, self._n_trunk, drafts=params.drafts, hit_len=raw_hit
            )
            if raw_hit > 0 and hit_len == 0:
                logger.info(
                    f"KV cache entry at {raw_hit}/{n_prompt} tokens has no aligned "
                    f"MTP head cache; prefilling from scratch"
                )
                cache, matched_index, is_exact = None, None, False
            elif hit_len > 0:
                logger.info(
                    f"KV cache hit: {hit_len}/{n_prompt} tokens cached "
                    f"({100 * hit_len / n_prompt:.1f}%) [mtp]"
                )

        prefill_ctx = None
        if vision is not None:
            _v = vision
            _start = hit_len

            def prefill_ctx():  # type: ignore[no-redef]
                return patch_embed_tokens(
                    self.model,
                    _v.embeddings,
                    _start,
                    n_prompt - 1,
                    image_token_id=_v.image_token_id,
                )

        snapshots: list[CacheSnapshot] = []
        want_snapshots = pool is not None and task_params.use_prefix_cache

        def on_chunk(trunk: list, head_cache: Any) -> None:
            # Recurrent (ArraysCache) layers cannot be trimmed, so the pool
            # restores them from snapshots; snapshot the COMBINED entry so the
            # indices line up with what gets stored below.
            if not want_snapshots:
                return
            combined = list(trunk) + ([head_cache] if params.drafts else [])
            if has_non_kv_caches(combined):
                snapshots.append(snapshot_ssm_states(combined))
                if len(snapshots) > 16:
                    half = len(snapshots) // 2
                    del snapshots[1:half:2]

        uid = self._uid_count
        self._uid_count += 1

        start = time.perf_counter()
        with mx.stream(generation_stream):
            row = admit(
                self.model,
                self.head,
                self._batch.get_h,
                all_prompt_tokens,
                params,
                uid=uid,
                make_draft_cache=self._make_draft_cache,
                prefill_step_size=_prefill_step_size(task_params.model),
                prefill_ctx=prefill_ctx,
                on_progress=on_prefill_progress,
                cache=cache,
                hcache=hcache,
                start_pos=hit_len,
                on_chunk=on_chunk,
            )
        elapsed = time.perf_counter() - start
        uncached = n_prompt - hit_len
        prefill_tps = uncached / elapsed if elapsed > 0 else 0.0
        prefix_cache_hit: Literal["none", "partial", "exact"] = "none"
        if matched_index is not None and hit_len > 0:
            if is_exact:
                prefix_cache_hit = "exact"
                assert pool is not None
                prefill_tps = pool.prefill_tps[matched_index]
            else:
                prefix_cache_hit = "partial"

        if want_snapshots:
            assert pool is not None
            self._save_prefix_cache(
                pool, task_params, all_prompt_tokens, row, snapshots,
                hit_len, matched_index, media_regions, prefill_tps,
            )
        self._pending.append(row)

        self._active_tasks[uid] = _EngineTask(
            uid=uid,
            task_params=task_params,
            all_prompt_tokens=all_prompt_tokens,
            prefix_hit_length=hit_len,
            matched_index=matched_index,
            detokenizer=self.tokenizer.detokenizer,
            on_generation_token=on_generation_token,
            generation_start_time=time.perf_counter(),
            prefill_tps=prefill_tps,
            prefix_cache_hit=prefix_cache_hit,
            media_regions=media_regions,
        )
        return uid

    def _save_prefix_cache(
        self,
        pool: KVPrefixCache,
        task_params: TextGenerationTaskParams,
        all_prompt_tokens: mx.array,
        row: Row,
        snapshots: list[CacheSnapshot],
        hit_len: int,
        matched_index: int | None,
        media_regions: list[MediaRegion],
        prefill_tps: float,
    ) -> None:
        """Store the prefilled prompt (trunk caches + head cache) the way the
        plain engine does at insert time: the pool deep-copies, so the row
        keeps its own caches. Same update-vs-add rule as the plain engine."""
        try:
            entry = list(row.cache) + ([row.hcache] if row.drafts else [])
            n = len(all_prompt_tokens)
            hit_ratio = hit_len / n if n > 0 else 0.0
            min_hit = max(1000, system_prompt_token_count(task_params, self.tokenizer))
            if matched_index is not None and (
                hit_len >= min_hit and hit_ratio >= _MIN_PREFIX_HIT_RATIO_TO_UPDATE
            ):
                pool.update_kv_cache(
                    matched_index, all_prompt_tokens, entry, snapshots,
                    restore_pos=hit_len, media_regions=media_regions,
                    prefill_tps=prefill_tps,
                )
            else:
                pool.add_kv_cache(
                    all_prompt_tokens, entry, snapshots,
                    media_regions=media_regions, prefill_tps=prefill_tps,
                )
        except Exception:
            logger.warning("Failed to save prefix cache [mtp]", exc_info=True)

    def step(self) -> list[tuple[int, GenerationResponse]]:
        if not self.has_work:
            return []

        _tic = time.perf_counter()
        with mx.stream(generation_stream):
            if self._pending:
                self._batch.extend(self._pending)
                self._pending.clear()
            row_steps = self._batch.step()
        _next_elapsed = time.perf_counter() - _tic

        results: list[tuple[int, GenerationResponse]] = []
        for rs in row_steps:
            state = self._active_tasks.get(rs.uid)
            if state is None:
                logger.warning(f"response uid {rs.uid} was not found - should be active")
                self._batch.remove([rs.uid])
                continue
            for em in rs.tokens:

                def _logprobs(
                    em=em, state=state
                ) -> tuple[float, list[TopLogprobItem]]:
                    r32 = em.logits.astype(mx.float32)
                    lp = r32 - mx.logsumexp(r32, axis=-1, keepdims=True)
                    with mx.stream(generation_stream):
                        return extract_top_logprobs(
                            logprobs=lp,
                            tokenizer=self.tokenizer,
                            top_logprobs=state.task_params.top_logprobs
                            or DEFAULT_TOP_LOGPROBS,
                            selected_token=em.token,
                        )

                out, is_done = emit_token(state, em.token, em.finish, logprobs=_logprobs)
                results.append((rs.uid, out))
                if is_done:
                    del self._active_tasks[rs.uid]
                    if em.finish is None:
                        # A stop SEQUENCE (text) ended it; the loop only
                        # knows about stop tokens and max_tokens, so the
                        # row is still live in the batch.
                        self._batch.remove([rs.uid])
                    acc = rs.accepted / rs.steps if rs.steps else 0.0
                    logger.info(
                        f"MTP batch row done: {state.completion_tokens} tokens, "
                        f"acceptance {acc:.3f} over {rs.steps} steps, "
                        f"{len(self._batch)} rows still decoding"
                    )
                    break

        _step_elapsed = time.perf_counter() - _tic
        self._step_count += 1
        if self._step_count % 64 == 0 and row_steps:
            logger.debug(
                f"mtp step overhead: {(_step_elapsed - _next_elapsed) * 1000:.2f}ms "
                f"(next={_next_elapsed * 1000:.2f}ms total={_step_elapsed * 1000:.2f}ms "
                f"rows={len(self._batch)})"
            )
        return results

    def cancel(self, uids: list[int]) -> None:
        drop = set(uids)
        self._pending = [r for r in self._pending if r.uid not in drop]
        self._batch.remove(uids)
        for uid in uids:
            self._active_tasks.pop(uid, None)

    def close(self) -> None:
        self._batch.filter([])
        self._pending.clear()
        self._stack.close()
        mx.clear_cache()
