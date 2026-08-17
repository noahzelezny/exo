# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false
"""Prefix-pool reuse for chunked-KV (llama4-style) models.

llama4 calls ChunkedKVCache.maybe_trim_front() at the START of every forward,
which drops the buffer to its last chunk_size columns based on BUFFER geometry
(keys.shape[2]) — it assumes offset sits near the buffer end, which a pool
rollback breaks. chunked_rollback_ok gates each hit on (1) the trim fitting
the logical window (trim() silently clamps) and (2) the restore point's chunk
surviving the NEXT forward's trim-front. Unsafe hits fall back to a fresh
prefill. End-to-end tests prove pooled continuation is logit-identical to a
fresh full prefill on a tiny random-weight llama4 whose prompts cross real
eviction boundaries (chunk_size must exceed the 256-col buffer step, as on
the real model where chunk_size=8192).
"""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ChunkedKVCache, KVCache
from mlx_lm.models.llama4 import Model, ModelArgs

from exo.worker.engines.mlx.cache import KVPrefixCache, chunked_rollback_ok

CHUNK = 256


def _chunked(offset: int, start_position: int, chunk_size: int = CHUNK):
    c = ChunkedKVCache(chunk_size=chunk_size)
    c.offset = offset
    c.start_position = start_position
    return c


# --- guard arithmetic (keys=None -> buffer term drops out) -------------------


def test_zero_trim_at_buffer_end_ok():
    cache = [_chunked(550, 200), KVCache()]
    assert chunked_rollback_ok(cache, tokens_to_trim=0, restore_pos=550) is True


def test_trim_beyond_window_rejected():
    # window = 550 - 200 = 350; trim 450 would silently clamp
    cache = [_chunked(550, 200)]
    assert chunked_rollback_ok(cache, tokens_to_trim=450, restore_pos=100) is False


def test_restore_chunk_not_retained_rejected():
    # restore_pos 250 sits in chunk [0, 256); start_position 200 > 0 means
    # that chunk's earlier keys were already evicted.
    cache = [_chunked(550, 200)]
    assert chunked_rollback_ok(cache, tokens_to_trim=300, restore_pos=250) is False


def test_buffer_geometry_projects_next_trim_front():
    # Logical state says start_position=0, but a 512-col buffer means the next
    # forward's maybe_trim_front advances start to 256 — a restore into chunk
    # [0, 256) must be rejected even though nothing is evicted *yet*.
    c = _chunked(300, 0)
    c.keys = mx.zeros((1, 2, 512, 8))
    c.values = mx.zeros((1, 2, 512, 8))
    assert chunked_rollback_ok([c], tokens_to_trim=150, restore_pos=150) is False
    # ...while a restore in the surviving tail chunk is fine.
    assert chunked_rollback_ok([c], tokens_to_trim=1, restore_pos=299) is True


def test_plain_kv_entries_ignored():
    assert chunked_rollback_ok([KVCache()], tokens_to_trim=999, restore_pos=0) is True


# --- end-to-end on a tiny random llama4 -------------------------------------


@pytest.fixture(scope="module")
def tiny_llama4() -> Model:
    args = ModelArgs(
        model_type="llama4",
        text_config=dict(
            attention_bias=False,
            attention_chunk_size=CHUNK,
            head_dim=8,
            hidden_size=32,
            interleave_moe_layer_step=1,
            intermediate_size=64,
            intermediate_size_mlp=64,
            max_position_embeddings=2048,
            model_type="llama4_text",
            num_attention_heads=4,
            num_experts_per_tok=1,
            num_hidden_layers=4,  # 3 chunked + 1 global, like the real thing
            num_key_value_heads=2,
            num_local_experts=2,
            rms_norm_eps=1e-5,
            rope_scaling=None,
            rope_theta=10000.0,
            use_qk_norm=True,
            vocab_size=64,
        ),
    )
    mx.random.seed(7)
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _forward(model: Model, tokens: mx.array, cache) -> mx.array:
    out = model(tokens[None], cache=cache)
    mx.eval(out)
    return out[0, -1]


def _prefill_in_calls(model: Model, tokens: mx.array, cache, step: int = 200):
    for i in range(0, tokens.shape[0], step):
        _forward(model, tokens[i : i + step], cache)


def _stored_evicted_cache(model: Model, tokens: mx.array):
    """Multi-call prefill so maybe_trim_front actually evicts (start>0)."""
    cache = model.make_cache()
    _prefill_in_calls(model, tokens, cache)
    assert any(isinstance(c, ChunkedKVCache) and c.start_position > 0 for c in cache), (
        "test must exercise a cache that actually evicted"
    )
    return cache


def test_append_flow_reuses_and_matches_fresh(tiny_llama4: Model):
    """Turn-2 continuation from an evicted pooled cache == fresh full prefill."""
    mx.random.seed(11)
    turn1 = mx.random.randint(0, 64, (550,))
    turn2 = mx.random.randint(0, 64, (50,))
    full = mx.concatenate([turn1, turn2])

    fresh_logits = _forward(tiny_llama4, full, tiny_llama4.make_cache())

    pool = KVPrefixCache(group=None)
    pool.add_kv_cache(turn1, _stored_evicted_cache(tiny_llama4, turn1))

    cache_r, remaining, matched, _ = pool.get_kv_cache(tiny_llama4, full)
    assert matched is not None, "append-style hit must reuse the pool"
    assert remaining.shape[0] == turn2.shape[0]

    pooled_logits = _forward(tiny_llama4, remaining, cache_r)
    assert mx.allclose(fresh_logits, pooled_logits, atol=1e-3, rtol=1e-3)


def test_exact_hit_small_trim_matches_fresh(tiny_llama4: Model):
    """Replaying the exact stored prompt trims 1 token near the buffer end."""
    mx.random.seed(13)
    prompt = mx.random.randint(0, 64, (550,))

    fresh_logits = _forward(tiny_llama4, prompt, tiny_llama4.make_cache())

    pool = KVPrefixCache(group=None)
    pool.add_kv_cache(prompt, _stored_evicted_cache(tiny_llama4, prompt))

    cache_r, remaining, matched, is_exact = pool.get_kv_cache(tiny_llama4, prompt)
    assert matched is not None and is_exact
    pooled_logits = _forward(tiny_llama4, remaining, cache_r)
    assert mx.allclose(fresh_logits, pooled_logits, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("diverge_at", [100, 450])
def test_divergent_history_falls_back_to_fresh(tiny_llama4: Model, diverge_at: int):
    """Rollbacks into evicted/soon-evicted chunks must NOT reuse the pool.
    diverge_at=100 exceeds the logical window (trim clamp); 450 fits the
    window but its chunk won't survive the next trim-front."""
    mx.random.seed(17)
    stored_prompt = mx.random.randint(0, 64, (550,))
    divergent = mx.concatenate(
        [stored_prompt[:diverge_at], (stored_prompt[diverge_at:] + 1) % 64]
    )

    pool = KVPrefixCache(group=None)
    pool.add_kv_cache(stored_prompt, _stored_evicted_cache(tiny_llama4, stored_prompt))

    cache_r, remaining, matched, _ = pool.get_kv_cache(tiny_llama4, divergent)
    assert matched is None, "unsafe chunked rollback must not reuse the pool"
    assert remaining.shape[0] == divergent.shape[0]

    fresh_logits = _forward(tiny_llama4, divergent, tiny_llama4.make_cache())
    pooled_logits = _forward(tiny_llama4, remaining, cache_r)
    assert mx.allclose(fresh_logits, pooled_logits, atol=1e-3, rtol=1e-3)
