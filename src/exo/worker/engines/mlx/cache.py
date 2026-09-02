import contextlib
import gc
import os
from copy import copy as _shallow_copy
from copy import deepcopy
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
import psutil
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    ChunkedKVCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.models.deepseek_v4 import (
    DeepseekV4Cache,
)
from mlx_lm.models.deepseek_v4 import (
    _CompressorBranch as CompressorBranch,  # type: ignore
)
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.memory import Memory
from exo.worker.engines.mlx.constants import CACHE_GROUP_SIZE, KV_CACHE_BITS
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.vision import MediaRegion


# Fraction of device memory above which LRU eviction kicks in.
# Smaller machines need more aggressive eviction.
def _default_memory_threshold() -> float:
    total_gb = Memory.from_bytes(psutil.virtual_memory().total).in_gb
    if total_gb >= 128:
        return 0.85
    if total_gb >= 64:
        return 0.80
    if total_gb >= 32:
        return 0.75
    return 0.70


_MEMORY_THRESHOLD = float(
    os.environ.get("EXO_MEMORY_THRESHOLD", _default_memory_threshold())
)

# Primary KV-pool eviction is by the pool's OWN token footprint, NOT total
# system memory. Rationale (measured 2026-06-25): the old system-memory trigger
# fired on the MODEL's transient inference spike (rank-0 node 52%->82%), not on
# pool size — and because it compared a cluster-wide pressure to each node's
# *local* threshold, the ring nodes made DIFFERENT eviction decisions, diverging
# their pools. Divergent pools => the prefix-reuse prefill collective gets a
# different token count per node => tensor-ring desync => wedge. A token budget
# is a deterministic quantity that is identical on every node (same prompts
# added in the same order), so all nodes evict identically and never diverge.
# ~0.4MB KV/token on the 397B => 48k tokens ~= 19GB of pool. Tune via env.
_KV_POOL_MAX_TOKENS = int(os.environ.get("EXO_KV_POOL_MAX_TOKENS", 48_000))

# Last-resort OOM guard. Cluster-CONSISTENT: cluster-max pressure vs this single
# global threshold (NOT the per-node _MEMORY_THRESHOLD that caused the divergence)
# so all nodes still decide identically. Set high — the token budget should
# normally keep the pool well clear of this.
_KV_OOM_THRESHOLD = float(os.environ.get("EXO_KV_OOM_THRESHOLD", 0.92))

# Cap retained SSM/compressor snapshots per pool entry. Each CacheSnapshot is a
# full deepcopy of the (non-trimmable) V4 compressor pool, whose bytes grow with
# its token_count. Retaining one at EVERY prefill-chunk boundary (as the merge
# in update_kv_cache did) accumulates ~1 snapshot per 2048 tokens across
# agent-loop turns, so total snapshot bytes grow O(context^2): measured ~11GB
# (~5.5GB/node) at 81k ctx on DeepSeek-V4-Flash — the dominant term in the
# long-context GPU (Metal) OOM, and counted by NEITHER the token-budget nor the
# system-RAM guard. _thin_snapshots keeps a geometric spread (largest per
# log2(token_count) bucket + always the newest) so per-entry snapshot memory is
# O(context) (~2x the resident cache) while every restore target stays within
# ~2x of a kept snapshot. Tune via env.
_MAX_SNAPSHOTS_PER_ENTRY = int(
    os.environ.get("EXO_KV_MAX_SNAPSHOTS_PER_ENTRY", 8)
)


class CacheSnapshot:
    """Snapshot of states at a known token position."""

    def __init__(
        self,
        states: list[
            RotatingKVCache | ArraysCache | CacheList | DeepseekV4Cache | None
        ],
        token_count: int,
    ):
        self.states = states
        self.token_count = token_count


def _detached_copy(a: mx.array) -> mx.array:
    dtype = a.dtype
    if dtype == mx.bfloat16:
        return mx.array(np.array(a.astype(mx.float32))).astype(mx.bfloat16)
    return mx.array(np.array(a))


def copy_rotating_kv_cache(cache: RotatingKVCache) -> RotatingKVCache | None:
    """
    Deepcopy copies the metadata associated with an mx array.
    Specifically, it shares a shared_ptr to the underlying data and
    the mlx graph inputs of the array. This causes a memory leak for rotating
    kv cache. By creating an np array, no metadata is stored so the old cache
    can be cleaned up nicely.
    """
    if cache.keys is None or cache.values is None:
        return None
    n = min(cache.max_size, cache.keys.shape[2])
    k_slice = _detached_copy(cache.keys[..., -n:, :])
    v_slice = _detached_copy(cache.values[..., -n:, :])
    mx.eval(k_slice, v_slice)
    snap = RotatingKVCache.__new__(RotatingKVCache)
    snap.keys = k_slice
    snap.values = v_slice
    snap.offset = cache.offset
    snap._idx = n
    snap.keep = cache.keep
    snap.max_size = cache.max_size
    return snap


def _copy_arrays_cache(ac: ArraysCache) -> ArraysCache:
    entries: list[mx.array | None] = []
    for entry in ac.cache:  # type: ignore[reportUnknownMemberType]
        if entry is None:
            entries.append(None)
            continue
        assert isinstance(entry, mx.array)
        entries.append(_detached_copy(entry))
    copy = ArraysCache(len(entries))
    copy.cache = entries  # type: ignore[reportUnknownMemberType]
    return copy


def _copy_cache_list(cl: CacheList) -> CacheList:
    inners: list[object] = list(cl)  # type: ignore[reportUnknownArgumentType]
    copied: list[object] = []
    for inner in inners:
        if isinstance(inner, RotatingKVCache):
            snap = copy_rotating_kv_cache(inner)
            copied.append(snap if snap is not None else deepcopy(inner))
        elif isinstance(inner, ArraysCache):
            copied.append(_copy_arrays_cache(inner))
        else:
            copied.append(deepcopy(inner))
    return CacheList(*copied)


def _detached_copy_or_none(a: mx.array | None) -> mx.array | None:
    if a is None:
        return None
    out = _detached_copy(a)
    mx.eval(out)
    return out


def _copy_compressor_branch(b: CompressorBranch) -> CompressorBranch:
    out = CompressorBranch.__new__(CompressorBranch)
    out.buffer_kv = _detached_copy_or_none(b.buffer_kv)
    out.buffer_gate = _detached_copy_or_none(b.buffer_gate)
    out.prev_kv = _detached_copy_or_none(b.prev_kv)
    out.prev_gate = _detached_copy_or_none(b.prev_gate)
    out.pool = _detached_copy_or_none(b.pool)
    out.buffer_lengths = deepcopy(b.buffer_lengths)
    out.pool_lengths = deepcopy(b.pool_lengths)
    out.buffer_count = deepcopy(b.buffer_count)
    out._new_pool_lengths = deepcopy(b._new_pool_lengths)
    return out


def _copy_v4_cache(c: DeepseekV4Cache) -> DeepseekV4Cache:
    snap = DeepseekV4Cache.__new__(DeepseekV4Cache)

    local: RotatingKVCache = c.local
    local_snap = copy_rotating_kv_cache(local)
    if local_snap is None:
        local_snap = RotatingKVCache.__new__(RotatingKVCache)
        local_snap.keys = None
        local_snap.values = None
        local_snap.offset = local.offset
        local_snap._idx = 0
        local_snap.keep = local.keep
        local_snap.max_size = local.max_size
    snap.local = local_snap

    snap._branches = {
        key: _copy_compressor_branch(branch) for key, branch in c._branches.items()
    }
    snap._pending_lengths = deepcopy(c._pending_lengths)
    return snap


def _duck_copy_cache(entry: object) -> object:
    """Snapshot a cache entry of a class this module doesn't know (e.g.
    mlx_vlm's ArraysCache, which glm5_next builds for its recurrent layers).
    Shallow-copy the object and detach its array-list state; mlx arrays are
    immutable, so everything else is safe to share."""
    dup = _shallow_copy(entry)
    cache_list = getattr(dup, "cache", None)
    if isinstance(cache_list, list):
        dup.cache = [  # type: ignore[attr-defined]
            _detached_copy(v) if isinstance(v, mx.array) else v
            for v in cache_list
        ]
    return dup


def copy_snapshot_entry(
    entry: ArraysCache | RotatingKVCache | CacheList | DeepseekV4Cache | None,
) -> ArraysCache | RotatingKVCache | CacheList | DeepseekV4Cache | None:
    match entry:
        case None:
            return None
        case RotatingKVCache():
            snap = copy_rotating_kv_cache(entry)
            return snap if snap is not None else deepcopy(entry)
        case ArraysCache():
            return _copy_arrays_cache(entry)
        case CacheList():
            return _copy_cache_list(entry)
        case DeepseekV4Cache():
            return _copy_v4_cache(entry)
        case _:
            return _duck_copy_cache(entry)  # type: ignore[return-value]


def snapshot_ssm_states(cache: KVCacheType) -> CacheSnapshot:
    states: list[
        RotatingKVCache | ArraysCache | CacheList | DeepseekV4Cache | None
    ] = []
    for c in cache:
        if isinstance(c, ArraysCache):
            states.append(_copy_arrays_cache(c))
        elif isinstance(c, RotatingKVCache):
            states.append(copy_rotating_kv_cache(c))
        elif isinstance(c, CacheList) and not bool(c.is_trimmable()):  # type: ignore[reportUnknownMemberType]
            states.append(_copy_cache_list(c))
        elif isinstance(c, DeepseekV4Cache):
            states.append(_copy_v4_cache(c))
        elif is_non_trimmable_cache_entry(c):
            # Unknown non-trimmable class (mlx_vlm caches): duck-copy, or the
            # restore branch in generate.py finds None and silently keeps the
            # un-rolled-back recurrent state -- phantom trailing tokens.
            states.append(_duck_copy_cache(c))  # type: ignore[arg-type]
        else:
            states.append(None)
    token_count = cache_length(cache)
    return CacheSnapshot(states=states, token_count=token_count)


def _find_nearest_snapshot(
    snapshots: list[CacheSnapshot],
    target_token_count: int,
) -> CacheSnapshot | None:
    best: CacheSnapshot | None = None
    for snap in snapshots:
        if snap.token_count <= target_token_count and (
            best is None or snap.token_count > best.token_count
        ):
            best = snap
    return best


def _thin_snapshots(
    snapshots: list[CacheSnapshot] | None,
    max_keep: int = _MAX_SNAPSHOTS_PER_ENTRY,
) -> list[CacheSnapshot] | None:
    """Bound per-entry snapshot memory to O(context) instead of O(context^2).

    Each CacheSnapshot is a full copy of the non-trimmable V4 compressor pool,
    whose size grows with its token_count. Retaining a snapshot at every
    prefill-chunk boundary (as update_kv_cache's merge did) keeps one per ~2048
    tokens, so total snapshot bytes grow quadratically with the conversation
    length and drive the V4 long-context GPU OOM.

    Keep at most one snapshot per log2(token_count) bucket (a geometric spread
    over [0, newest]) and always the newest — needed for exact / continuation
    reuse (the common agent-loop case). That bounds the count to ~log2(context)
    and total bytes to ~2x the resident cache, while keeping every restore
    target within ~2x of a retained snapshot: partial / branch reuse degrades to
    at-most-2x re-prefill rather than falling all the way back to a full prefill.
    """
    if not snapshots or len(snapshots) <= max_keep:
        return snapshots
    ordered = sorted(snapshots, key=lambda s: s.token_count)
    newest = ordered[-1]
    # Largest snapshot per log2 bucket => geometric coverage of the token axis.
    by_bucket: dict[int, CacheSnapshot] = {}
    for s in ordered:
        bucket = max(1, s.token_count).bit_length()
        cur = by_bucket.get(bucket)
        if cur is None or s.token_count > cur.token_count:
            by_bucket[bucket] = s
    kept = sorted(by_bucket.values(), key=lambda s: s.token_count)
    if newest.token_count != kept[-1].token_count:
        kept.append(newest)
    # Belt-and-suspenders hard cap: keep the newest max_keep by token_count.
    if len(kept) > max_keep:
        kept = kept[-max_keep:]
    logger.info(
        f"KVMEM thin-snapshots {len(snapshots)}->{len(kept)} "
        f"kept_tc={[s.token_count for s in kept]}"
    )
    return kept


def is_non_trimmable_cache_entry(c: object) -> bool:
    """A cache entry is non-trimmable if `trim(n)` can't roll back its full
    state — meaning the prefill +2 rollback must snapshot+restore it instead.
    """
    if isinstance(c, (ArraysCache, RotatingKVCache)):
        return True
    if isinstance(c, CacheList):
        return not bool(c.is_trimmable())  # type: ignore[reportUnknownMemberType]
    if isinstance(c, DeepseekV4Cache):
        return True
    # Duck-type the rest: mlx_vlm ships its OWN cache classes (e.g.
    # mlx_vlm.models.cache.ArraysCache for glm5_next's recurrent layers),
    # which fail every isinstance above, take the trim() branch, and raise.
    # An entry that says it isn't trimmable, or that has no trim() at all,
    # cannot take that branch -- whatever package it came from.
    is_trimmable = getattr(c, "is_trimmable", None)
    if callable(is_trimmable):
        try:
            return not bool(is_trimmable())
        except Exception:
            return True
    return not callable(getattr(c, "trim", None))


def has_non_kv_caches(cache: KVCacheType) -> bool:
    """Check if a cache contains any ArraysCache (SSM) entries."""
    return any(is_non_trimmable_cache_entry(c) for c in cache)


def chunked_rollback_ok(
    cache: KVCacheType, tokens_to_trim: int, restore_pos: int
) -> bool:
    """Whether a pool restore is sound for every ChunkedKVCache entry.

    ChunkedKVCache (llama4-style chunked attention) EVICTS history: at the
    START of every forward, llama4 calls maybe_trim_front(), which drops the
    buffer down to the last chunk_size columns and advances start_position —
    computed from BUFFER geometry (keys.shape[2]), assuming offset sits near
    the buffer end. A pool restore breaks that assumption, so a rollback is
    only correct when, for every chunked entry:
      1. the trim fits the logical window — ChunkedKVCache.trim() silently
         CLAMPS to (offset - start_position), which would leave a cache whose
         offset lies about its contents;
      2. after the NEXT forward's trim-front (start_position advancing by
         buf_len - chunk_size), the restore point's chunk is still fully
         retained — otherwise continuation either attends to a chunk whose
         earlier keys were evicted, or writes at a negative buffer index.
    The dominant chat-append flow trims 0 tokens with the offset near the
    buffer end and passes trivially; deep divergent-history rollbacks are
    sent back to a fresh prefill.
    """
    for c in cache:
        if not isinstance(c, ChunkedKVCache):
            continue
        window = c.offset - c.start_position
        if tokens_to_trim > window:
            return False
        buf_len = c.keys.shape[2] if c.keys is not None else 0
        start_after = c.start_position
        if buf_len >= c.chunk_size:
            start_after += buf_len - c.chunk_size
        if start_after > (restore_pos // c.chunk_size) * c.chunk_size:
            return False
    return True


class KVPrefixCache:
    def __init__(self, group: mx.distributed.Group | None):
        self.prompts: list[mx.array] = []  # mx array of tokens (ints)
        self.caches: list[KVCacheType] = []
        self._snapshots: list[list[CacheSnapshot] | None] = []
        self._media_regions: list[list["MediaRegion"]] = []
        self._last_used: list[int] = []  # monotonic counter of last access per entry
        self.prefill_tps: list[float] = []
        self._access_counter: int = 0
        self._group = group

    def clear(self):
        """Clear all cached prompts and caches."""
        self.prompts.clear()
        self.caches.clear()
        self._snapshots.clear()
        self._media_regions.clear()
        self._last_used.clear()
        self.prefill_tps.clear()

    # --- KVMEM instrumentation (Scout 2026-06-25) -----------------------------
    # Measures how close the KV prefix pool runs to the eviction ceiling, and
    # whether the deepcopy spike on add/update/restore crosses it. Pure logging:
    # uses LOCAL psutil pressure only (module-level get_memory_used_percentage),
    # NEVER the instance all_gather path — adding a collective for logging would
    # desync the tensor ring. Each node logs its own pressure to its own
    # exo-supervised.log; compare M3 vs M4 logs to find the bottleneck node.
    def _pool_tokens(self) -> int:
        total = 0
        for p in self.prompts:
            with contextlib.suppress(Exception):
                total += len(p)
        return total

    def _log_pool(self, event: str, **extra: object) -> None:
        mem = get_memory_used_percentage()  # local psutil — no collective
        parts = " ".join(f"{k}={v}" for k, v in extra.items())
        logger.info(
            f"KVMEM event={event} pool={len(self.caches)} "
            f"pool_tokens={self._pool_tokens()} mem={mem:.4f} "
            f"thr={_MEMORY_THRESHOLD:.4f} {parts}".rstrip()
        )

    def add_kv_cache(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        ssm_snapshots: list[CacheSnapshot] | None = None,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ):
        """Add a new cache entry. Evicts LRU entries if memory is high."""
        self._log_pool("add-pre", tokens=len(prompt_tokens))
        self._evict_if_needed()
        self.prompts.append(prompt_tokens)
        self.caches.append(deepcopy(cache))
        self._snapshots.append(_thin_snapshots(ssm_snapshots))
        self._media_regions.append(media_regions or [])
        self.prefill_tps.append(prefill_tps)
        self._access_counter += 1
        self._last_used.append(self._access_counter)
        # add-post mem minus add-pre mem = the deepcopy spike that lands AFTER
        # the evictor already decided there was room.
        self._log_pool("add-post", tokens=len(prompt_tokens))

    def update_kv_cache(
        self,
        index: int,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list[CacheSnapshot] | None,
        restore_pos: int,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ):
        """Update an existing cache entry in-place."""
        old_snapshots = self._snapshots[index]
        merged: list[CacheSnapshot] = []
        if old_snapshots:
            merged = [s for s in old_snapshots if s.token_count <= restore_pos]
        if snapshots:
            merged.extend(snapshots)

        self.prompts[index] = prompt_tokens
        self.caches[index] = deepcopy(cache)
        self._snapshots[index] = _thin_snapshots(merged) or None
        self._media_regions[index] = media_regions or []
        self.prefill_tps[index] = prefill_tps
        self._access_counter += 1
        self._last_used[index] = self._access_counter
        self._log_pool("update", idx=index, tokens=len(prompt_tokens))

    def _get_snapshot(
        self, entry_index: int, target_token_count: int
    ) -> tuple[int, CacheSnapshot | None]:
        if not has_non_kv_caches(self.caches[entry_index]):
            return target_token_count, None

        snapshots = self._snapshots[entry_index]
        if not snapshots:
            return 0, None

        snap = _find_nearest_snapshot(snapshots, target_token_count)
        if snap is not None:
            return snap.token_count, snap

        return 0, None

    def get_kv_cache(
        self,
        model: Model,
        prompt_tokens: mx.array,
        media_regions: list["MediaRegion"] | None = None,
    ) -> tuple[KVCacheType, mx.array, int | None, bool]:
        """Get KV cache for prompt, returning remaining tokens to prefill.

        Returns:
            Tuple of (cache, remaining_tokens, matched_index, is_exact) where:
            - cache: KV cache to use for generation
            - remaining_tokens: tokens that still need prefilling
            - matched_index: index of the matched entry (None if no match)
            - is_exact: True if the full prompt matched the cached entry

        For models with SSM layers (which are ArraysCache in mlx), the cache is trimmed to the
        nearest SSM snapshot position at or before the match point for correctness.
        Same for rotating KV Cache.

        Media region validation: if the token-level prefix match extends into
        a cached media region whose content_hash differs from the query's, the
        match is truncated to the start of that region.
        """
        max_length = len(prompt_tokens)
        query_regions = media_regions or []

        best_index: int | None = None
        best_length = 0
        is_exact = False

        # Find best cache match
        for i, cached_prompt in enumerate(self.prompts):
            length = get_prefix_length(prompt_tokens, cached_prompt)
            if length > 0:
                length = self._validate_media_match(
                    length,
                    self._media_regions[i],
                    query_regions,
                )
            if length >= max_length - 1:
                best_index, best_length = i, length
                is_exact = True
                break
            if length > best_length:
                best_index, best_length = i, length

        if best_index is None:
            return make_kv_cache(model), prompt_tokens, None, False

        # For exact match: trim to max_length-1 so remaining has the last token
        # For partial match: trim to best_length, remaining has suffix to prefill
        # This ensures stream_generate always has at least one token to start with
        has_ssm = has_non_kv_caches(self.caches[best_index])
        cached_length = cache_length(self.caches[best_index])
        if has_ssm:
            target = best_length
        else:
            desired = (max_length - 1) if is_exact else best_length
            target = min(cached_length, desired)
        restore_pos, restore_snap = self._get_snapshot(best_index, target)

        # No usable snapshot — need fresh cache
        if restore_snap is None and has_ssm:
            return make_kv_cache(model), prompt_tokens, None, False

        if not chunked_rollback_ok(
            self.caches[best_index], cached_length - restore_pos, restore_pos
        ):
            self._log_pool(
                "chunked-miss", idx=best_index, trim=cached_length - restore_pos
            )
            return make_kv_cache(model), prompt_tokens, None, False

        prompt_cache = deepcopy(self.caches[best_index])
        self._log_pool("restore", idx=best_index, tokens=cached_length)
        tokens_to_trim = cached_length - restore_pos
        if tokens_to_trim > 0:
            trim_cache(prompt_cache, tokens_to_trim, restore_snap)
            # Reset cache offset to match trimmed length
            for c in prompt_cache:
                if isinstance(c, (ArraysCache, RotatingKVCache)):
                    continue
                if isinstance(c, DeepseekV4Cache):
                    continue
                if hasattr(c, "offset"):
                    c.offset = restore_pos

        self._access_counter += 1
        self._last_used[best_index] = self._access_counter
        remaining = prompt_tokens[restore_pos:]

        return prompt_cache, remaining, best_index, is_exact

    @staticmethod
    def _validate_media_match(
        match_length: int,
        cached_regions: list["MediaRegion"],
        query_regions: list["MediaRegion"],
    ) -> int:
        if not cached_regions:
            return match_length

        query_by_start: dict[int, "MediaRegion"] = {
            r.start_pos: r for r in query_regions
        }

        for cached_r in cached_regions:
            if cached_r.start_pos >= match_length:
                break
            query_r = query_by_start.get(cached_r.start_pos)
            if query_r is None:
                continue
            if query_r.content_hash != cached_r.content_hash:
                logger.info(
                    f"Media region mismatch at pos {cached_r.start_pos}: "
                    f"cached={cached_r.content_hash[:12]}... "
                    f"query={query_r.content_hash[:12]}... — "
                    f"truncating match from {match_length} to {cached_r.start_pos}"
                )
                match_length = cached_r.start_pos
                break

        return match_length

    def _evict_lru(self, reason: str) -> None:
        """Pop the least-recently-used pool entry. The LRU index is a function of
        the access sequence, which is identical on every ring node, so every node
        evicts the SAME entry — keeping the pools consistent."""
        lru_index = self._last_used.index(min(self._last_used))
        evicted_tokens = len(self.prompts[lru_index])
        self.prompts.pop(lru_index)
        self.caches.pop(lru_index)
        self._snapshots.pop(lru_index)
        self._media_regions.pop(lru_index)
        self._last_used.pop(lru_index)
        self.prefill_tps.pop(lru_index)
        self._log_pool("evict", tokens=evicted_tokens, reason=reason)

    def _evict_if_needed(self):
        """Evict LRU entries to keep the pool bounded.

        Primary trigger is the pool's OWN token footprint vs a fixed budget — a
        DETERMINISTIC quantity identical on every node — so all nodes evict the
        same entries and the pools never diverge (divergence wedges the ring; see
        the _KV_POOL_MAX_TOKENS note). The system-memory check is kept ONLY as a
        last-resort OOM guard and is cluster-consistent (cluster-max vs one global
        threshold), so it too decides identically on all nodes.
        """
        if len(self.caches) == 0:
            return

        self._log_pool("evict-check")
        evicted_any = False

        # Primary: deterministic pool-footprint budget (consistent across nodes).
        # Keep >=1 entry (the active conversation) even if it alone exceeds budget.
        while len(self.caches) > 1 and self._pool_tokens() > _KV_POOL_MAX_TOKENS:
            self._evict_lru("budget")
            evicted_any = True

        # Safety net: hard OOM guard. cluster-max pressure (all_gather, identical
        # on every node) vs one global threshold => every node makes the same call
        # in lockstep, so the all_gather stays balanced and pools stay consistent.
        while (
            len(self.caches) > 0
            and self.get_memory_used_percentage() > _KV_OOM_THRESHOLD
        ):
            self._evict_lru("oom")
            evicted_any = True

        if evicted_any:
            gc.collect()
            mx.clear_cache()
            self._log_pool("evict-done")

    def get_memory_used_percentage(self) -> float:
        local_pressure: float = get_memory_used_percentage()

        if self._group is None:
            return local_pressure

        all_pressure = mx.distributed.all_gather(
            mx.array([local_pressure], dtype=mx.float32),
            group=self._group,
        )
        # .item() evals.
        max_pressure = float(mx.max(all_pressure).item())
        return max_pressure


def trim_cache(
    cache: KVCacheType,
    num_tokens: int,
    snapshot: CacheSnapshot | None = None,
) -> None:
    for i, c in enumerate(cache):
        non_trimmable = isinstance(c, (ArraysCache, RotatingKVCache)) or (
            isinstance(c, CacheList) and not bool(c.is_trimmable())  # type: ignore[reportUnknownMemberType]
        )
        if non_trimmable:
            if snapshot is not None and snapshot.states[i] is not None:
                restored = copy_snapshot_entry(snapshot.states[i])
                if restored is not None:
                    cache[i] = restored  # type: ignore
            elif isinstance(c, (ArraysCache, RotatingKVCache)):
                c.state = [None] * len(c.state)
                if isinstance(c, RotatingKVCache):
                    c.offset = 0
                    c._idx = 0
            else:
                # CacheList without a snapshot — zero each inner cache's state
                for inner in c:  # type: ignore[reportUnknownVariableType]
                    if isinstance(inner, (ArraysCache, RotatingKVCache)):
                        inner.state = [None] * len(inner.state)
                        if isinstance(inner, RotatingKVCache):
                            inner.offset = 0
                            inner._idx = 0
        else:
            c.trim(num_tokens)


def encode_prompt(tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
    """Encode a prompt string to token array.

    For chat-templated prompts (which have their own structure markers like
    <|im_user|>, <|im_middle|>, etc.), we should NOT add BOS/EOS tokens as
    that would corrupt the prompt structure.
    """
    # Chat templates define their own structure - don't add BOS/EOS
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return mx.array(prompt_tokens)


def _entry_length(
    c: KVCache
    | RotatingKVCache
    | QuantizedKVCache
    | ArraysCache
    | CacheList
    | DeepseekV4Cache,
) -> int:
    # Use .offset attribute which KVCache types have (len() not implemented in older QuantizedKVCache).
    if hasattr(c, "offset"):
        return c.offset
    # For CacheList
    if hasattr(c, "size"):
        return int(c.size())  # type: ignore
    return 0


def cache_length(cache: KVCacheType) -> int:
    """Get the number of tokens in a KV cache."""
    return max((_entry_length(c) for c in cache), default=0)


def get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    """Find the length of the common prefix between two token arrays."""
    n = min(int(prompt.shape[0]), int(cached_prompt.shape[0]))
    if n == 0:
        return 0

    equal = mx.equal(prompt[:n], cached_prompt[:n]).astype(mx.int32)
    prefix_mask = mx.cumprod(equal)  # stays 1 until first mismatch, then 0 forever
    return int(mx.sum(prefix_mask).item())


def get_available_memory() -> Memory:
    mem: int = psutil.virtual_memory().available
    return Memory.from_bytes(mem)


def get_memory_used_percentage() -> float:
    mem = psutil.virtual_memory()
    # percent is 0-100
    return float(mem.percent / 100)


def make_kv_cache(
    model: Model, max_kv_size: int | None = None, keep: int = 0
) -> KVCacheType:
    assert hasattr(model, "layers")

    if hasattr(model, "make_cache"):
        logger.info("Using MLX LM's make cache")
        return model.make_cache()  # type: ignore

    if max_kv_size is None:
        if KV_CACHE_BITS is None:
            logger.info("Using default KV cache")
            return [KVCache() for _ in model.layers]
        else:
            logger.info("Using quantized KV cache")
            return [
                QuantizedKVCache(group_size=CACHE_GROUP_SIZE, bits=KV_CACHE_BITS)
                for _ in model.layers
            ]
    else:
        logger.info(f"Using rotating KV cache with {max_kv_size=} with {keep=}")
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]
