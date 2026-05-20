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
KV_BITS: int | None = None
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
# Scout patch 2026-05-14: also quantize at cache-creation time for models
# that go through exo's own make_kv_cache path (non-make_cache models).
KV_CACHE_BITS: int | None = None

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
