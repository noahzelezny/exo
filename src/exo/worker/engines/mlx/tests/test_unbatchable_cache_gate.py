# pyright: reportPrivateUsage=false
"""Chunked-KV models must route to the sequential engine.

mlx_lm's batch engine raises "ChunkedKVCache does not yet support batching
with history" (_merge_caches) the moment a request lands, killing the runner —
hit live with Llama-4-Scout on 2026-07-15. The builder probes the model's
cache layout and falls back to SequentialGenerator with the prefix pool off.
No weights needed — the probe only inspects make_cache() types.
"""

from mlx_lm.models.cache import ChunkedKVCache, KVCache

from exo.worker.engines.mlx.builder import _has_unbatchable_cache


class _ChunkedModel:
    """Cache layout shaped like llama4: chunked attention on 3 of 4 layers."""

    def make_cache(self):
        return [
            ChunkedKVCache(chunk_size=8192) if (i + 1) % 4 != 0 else KVCache()
            for i in range(8)
        ]


class _PlainModel:
    def make_cache(self):
        return [KVCache() for _ in range(8)]


class _NoMakeCache:
    pass


class _RaisingModel:
    def make_cache(self):
        raise RuntimeError("sharded model quirk")


def test_chunked_cache_model_is_unbatchable():
    assert _has_unbatchable_cache(_ChunkedModel()) is True


def test_plain_kv_model_stays_batched():
    assert _has_unbatchable_cache(_PlainModel()) is False


def test_model_without_make_cache_stays_batched():
    assert _has_unbatchable_cache(_NoMakeCache()) is False


def test_probe_failure_fails_open_to_batched():
    assert _has_unbatchable_cache(_RaisingModel()) is False
