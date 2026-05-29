# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

KV_GROUP_SIZE: int | None = 32
# KV_BITS feeds generate.py's maybe_quantize_kv_cache (quantized_kv_start=0),
# which quantizes the cache during generation regardless of how it was
# created. Global default stays None (FP16): some MoE families produce
# gibberish on a quantized KV cache, so quantization is opted into
# per-model via `kv_bits_for()` below.
KV_BITS: int | None = None
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
# See KV_BITS note above re: per-model gating; stays None so the
# non-make_cache path defaults to FP16.
KV_CACHE_BITS: int | None = None


def kv_bits_for(model_id: str | None) -> int | None:
    """Per-model KV cache quantization override.

    Returns the bits value to pass to stream_generate's `kv_bits=` (and to
    `pipeline_parallel_prefill`). None = FP16 (unquantized, safe default);
    an int = quantize the KV cache to that bit width at generation time.

    The decision lives here, not in a config file, because it tracks model-
    architecture quirks discovered empirically — every entry should cite
    the test that proved it. Add/remove cases when new evidence lands.

    Confirmed-safe at 8-bit:
      - mlx-community/DeepSeek-V4-Flash : MLA attention tolerates 8-bit;
        2026-05-28 smoke test ("Count 1-10") returned coherent output,
        fixes the long-context kIOGPUCommandBufferCallbackErrorOutOfMemory.

    Confirmed-broken at 8-bit (stay None):
      - mlx-community/Qwen3.5-122B-A10B-4bit : 2026-05-18 produced "alyze"
        gibberish; 2026-05-28 re-confirmed.
      - GLM-4.7 family : reported broken on quantized KV (2026-05-18).
      - Any other Qwen MoE checkpoint until tested.

    Untested → returns KV_BITS (None) as the safe default.
    """
    if not model_id:
        return KV_BITS
    if "DeepSeek-V4" in model_id:
        return 8
    return KV_BITS


DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
