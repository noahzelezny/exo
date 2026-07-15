# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false
"""Llama4ShardingStrategy: dispatch + shard geometry, offline.

Sharding only slices weights at build time (rank/size math); collectives run
at forward time, so a fake 2-rank group lets us verify the full
tensor_auto_parallel path — dispatch picks the llama4 strategy, attention
projections split by heads, SwitchGLU experts and the shared expert split on
the intermediate dim, the router stays whole, and the MoE gets the ShardedMoE
all-sum wrapper — without a distributed backend.
"""

import mlx.core as mx
import pytest
from mlx_lm.models.llama4 import Model, ModelArgs

from exo.worker.engines.mlx.auto_parallel import ShardedMoE, tensor_auto_parallel


class _FakeGroup:
    def size(self) -> int:
        return 2

    def rank(self) -> int:
        return 0


def _tiny_llama4() -> Model:
    args = ModelArgs(
        model_type="llama4",
        text_config=dict(
            attention_bias=False,
            attention_chunk_size=256,
            head_dim=8,
            hidden_size=32,
            interleave_moe_layer_step=1,
            intermediate_size=64,
            intermediate_size_mlp=64,
            max_position_embeddings=2048,
            model_type="llama4_text",
            num_attention_heads=4,
            num_experts_per_tok=1,
            num_hidden_layers=4,
            num_key_value_heads=2,
            num_local_experts=2,
            rms_norm_eps=1e-5,
            rope_scaling=None,
            rope_theta=10000.0,
            use_qk_norm=True,
            vocab_size=64,
        ),
    )
    mx.random.seed(3)
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _drive(gen):
    while True:
        try:
            next(gen)
        except StopIteration as e:
            return e.value


def test_tensor_auto_parallel_shards_llama4():
    model = _tiny_llama4()
    original_call = Model.__call__
    try:
        sharded = _drive(tensor_auto_parallel(model, _FakeGroup()))
    finally:
        # patch_tensor_model rebinds the CLASS __call__; restore so other
        # tests in this process see the pristine Model.
        Model.__call__ = original_call

    assert sharded is model
    for layer in model.layers:
        attn = layer.self_attn
        # 4 heads / 2 kv heads split across 2 ranks
        assert attn.n_heads == 2
        assert attn.n_kv_heads == 1
        # q: (4*8, 32) -> (16, 32); k/v: (2*8, 32) -> (8, 32)
        assert attn.q_proj.weight.shape[0] == 16
        assert attn.k_proj.weight.shape[0] == 8
        assert attn.v_proj.weight.shape[0] == 8
        # o: (32, 4*8) -> input-sharded (32, 16)
        assert attn.o_proj.weight.shape[-1] == 16

        # Scout-style: every layer is MoE, wrapped for the block all_sum
        assert isinstance(layer.feed_forward, ShardedMoE)
        assert isinstance(layer.feed_forward.sharding_group, _FakeGroup)
        moe = layer.feed_forward.original_layer
        # SwitchGLU expert weights are (E, out, in): intermediate 64 -> 32
        assert moe.experts.gate_proj.weight.shape == (2, 32, 32)
        assert moe.experts.up_proj.weight.shape == (2, 32, 32)
        assert moe.experts.down_proj.weight.shape == (2, 32, 32)
        # shared expert splits like a dense MLP
        assert moe.shared_expert.gate_proj.weight.shape[0] == 32
        assert moe.shared_expert.down_proj.weight.shape[-1] == 32
        # router replicated: still scores both experts from full hidden
        assert moe.router.weight.shape == (2, 32)


def test_unsupported_model_still_raises():
    class NotAModel:
        pass

    with pytest.raises(ValueError, match="Unsupported model type"):
        _drive(tensor_auto_parallel(NotAModel(), _FakeGroup()))
