# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

KV_GROUP_SIZE: int | None = 32
# Scout patch 2026-05-14: KV cache quantization was OFF (None), so long
# agent-loop contexts ran an unquantized KV cache that blew past M4's
# ~107GB Metal wired limit -> kIOGPUCommandBufferCallbackErrorOutOfMemory
# -> runner SIGABRT. KV_BITS feeds generate.py's maybe_quantize_kv_cache
# (quantized_kv_start=0), which quantizes the cache during generation
# regardless of how it was created — this is the knob that actually
# affects make_cache models like Qwen3.5. 8-bit ~halves KV memory at
# negligible quality cost.
# Scout 2026-05-28 (afternoon): reverted to None per Noah — Qwen MoE still
# gibberishes at 8-bit KV (2026-05-18 finding re-confirmed today) but V4-Flash
# OOMs without it. Moved to per-model gating via `kv_bits_for()` below; the
# generator now resolves at call time using task.model. These constants stay
# at None so the default for any new/unknown model is the safe choice.
KV_BITS: int | None = None
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
# See KV_BITS note above re: per-model gating. Stays None to default the
# non-make_cache path to FP16; V4-Flash uses model.make_cache anyway so
# this path isn't on its hot codepath.
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

def prefill_step_size_for(model_id: str | None) -> int | None:
    """Per-model prefill chunk-size override (tokens per prefill chunk).

    Returns the chunk size to use instead of the 4096 upstream default, or
    None to keep the default. EXO_PREFILL_STEP_SIZE still wins over both.

    Like `kv_bits_for`, entries are empirical and must cite the evidence:

    Confirmed-needed small chunks:
      - GLM-5.3 family : 34 deltanet layers hold per-token recurrent
        intermediates in the lazy graph across a chunk, so the prefill
        transient scales with chunk_size x per-layer STATE size
        (16.8MB/layer), not the KV row. 2026-09-01: OOMed BOTH boxes of a
        224GB pair on the 135GB 3.6bpw at 4096; 512 caps the transient.

    Confirmed-hurt by small chunks (keep None):
      - Qwen3.5-397B (qwen4_exp) : also has ArraysCache entries (deltanet
        + PLE slots), so any "has SSM caches" heuristic catches it — but
        its recurrent state is small and it served 4096-token chunks for
        weeks. 2026-09-02: a blanket SSM->512 default made its prefill
        unusably slow (8x the chunks, each with a blocking per-chunk eval
        over the TCP ring). This per-model table replaces that heuristic.
    """
    if not model_id:
        return None
    if "glm-5.3" in model_id.lower():
        return 512
    return None


DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
