"""Absorbed-MLA fast path for GLM-5-Next (glm5_next) at small L.

Vendored from VQLab: src/vqlab/glm5_shim.py @ 31b11b7 (merged 3cf0af9,
2026-09-02). Copied, not imported — exo never imports vqlab. The patched
`__call__` body below is byte-for-byte the upstream
`mlx_vlm.models.glm5_next.language.Glm5NextSparseAttention.__call__`
except for the two `L == 1` route tests, which become `L <=
absorb_max_L`. Deviation from the VQLab source: none in behaviour; the
module-level `scaled_dot_product_attention` import stays function-local
so importing this module never imports mlx_vlm.

WHERE IT IS INSTALLED IN EXO: `mtp/speculative.py:_load_head`, only when
a glm5_next MTP head is about to be built — i.e. only under EXO_MTP=1
for a glm5_next model with a sidecar present. With EXO_MTP unset nothing
in this file ever runs and the stock decode path is untouched.

Why
---
``Glm5NextSparseAttention.__call__`` takes two different attention routes:

  L == 1  ABSORBED   -- the query is projected into the 512-d KV-latent space
                        (``embed_q(q)``) and attends directly against the
                        latent cache; ``unembed_out`` is applied to the
                        *output*.  Cost is O(L * H * Kv * 512).

  L  > 1  UNABSORBED -- the whole latent cache is expanded per head into
                        k = ``embed_q(kv_latent)``  [B, H, Kv, 256] and
                        v = ``unembed_out(kv_latent)`` [B, H, Kv, 256]
                        before attention.  Cost is O(Kv * H * 512 * 256),
                        i.e. INDEPENDENT of L and ~2 orders of magnitude
                        above the absorbed route at L = 2.

That expansion is the right trade for a long prefill (it amortises over
thousands of queries) and exactly the wrong one for MTP speculative
verification, which runs L = 2 against a long cache.  Measured in VQLab on
one fa layer (M4, bf16, random init, single stream) -- NOT re-measured here:

    Kv        L=1 (ms)   L=2 stock (ms)   ratio
      128       0.78          1.34         1.69
      512       0.79          2.49         3.19
     2048       0.87          6.07         6.96
     8192       0.86         20.45        23.78
    13312       0.83         33.43        40.22

The two routes are algebraically identical (the standard MLA absorption
identity: q Wq^T kv^T == (q Wq^T) kv^T and (P kv) Wu^T == P (kv Wu^T)),
so taking the absorbed route at small L changes cost, not math. VQLab
verified the two routes agree to one bf16 ULP.

What this shim does
-------------------
Monkeypatches ``Glm5NextSparseAttention.__call__`` so that the absorbed
route is taken for every ``L <= absorb_max_L`` (default 8), not just
L == 1.  The topk / sparse-mask construction is left EXACTLY as upstream
wrote it: at L == 1 the selected keys are still gathered, and at L > 1 the
``put_along_axis`` sparse bool mask over [.., Kv] is still what enforces
selection.  Only the k/v construction switches.

``install()`` is idempotent and returns True if it patched.
"""

from typing import Any, Optional

import mlx.core as mx

_INSTALLED = False
_ORIG = None

#: attention with this many query positions or fewer takes the absorbed route
ABSORB_MAX_L = 8


def _patched_call(
    self,
    x: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[Any] = None,
) -> mx.array:
    from mlx_vlm.models.base import scaled_dot_product_attention

    B, L, D = x.shape

    qr = self.q_a_layernorm(self.q_a_proj(x))
    q = self.q_b_proj(qr)
    q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)

    compressed_kv = self.kv_a_proj_with_mqa(x)
    kv_latent = self.kv_a_layernorm(compressed_kv)
    kv_latent = mx.expand_dims(kv_latent, axis=1)

    if cache is not None:
        kv_latent, _ = cache[0].update_and_fetch(kv_latent, kv_latent)
    else:
        cache = [None] * 2

    topk_indices = self.indexer(x, qr, mask, cache=cache[1])
    attn_mask = mask
    if topk_indices is not None:
        Kv = kv_latent.shape[2]
        valid_sel = topk_indices >= 0
        if L == 1:
            clamped = mx.clip(topk_indices[:, :, 0, :], 0, Kv - 1)
            idx = clamped[..., None]
            kv_latent = mx.take_along_axis(
                kv_latent,
                mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                axis=2,
            )
            sel_mask = valid_sel[:, :, 0, :][:, :, None, :]
            if mask is not None and mask.dtype == mx.bool_:
                mkeys = mask.reshape(B, -1, Kv)[:, 0, :]
                gathered = mx.take_along_axis(
                    mx.broadcast_to(mkeys[:, None, :], (B, clamped.shape[1], Kv)),
                    clamped,
                    axis=-1,
                )
                sel_mask = sel_mask & gathered[:, :, None, :]
            attn_mask = sel_mask
        else:
            shape = list(topk_indices.shape)
            shape[-1] = Kv + 1
            safe_idx = mx.where(valid_sel, topk_indices, Kv)
            sparse_mask = mx.zeros(shape, dtype=mx.bool_)
            sparse_mask = mx.put_along_axis(
                sparse_mask, safe_idx, mx.array(True), axis=-1
            )
            sparse_mask = sparse_mask[..., :Kv]
            if mask is not None and mask.dtype == mx.bool_:
                sparse_mask = sparse_mask & mask
            attn_mask = sparse_mask

    if (
        cache is not None
        and cache[0] is not None
        and cache[1] is not None
        and cache[1].keys is not None
    ):
        cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

    # ---- the only behavioural change: widen the absorbed route to small L ----
    absorbed = L <= getattr(self, "absorb_max_L", ABSORB_MAX_L)
    if absorbed:
        q = self.embed_q(q)
        k = v = kv_latent
    else:
        k = self.embed_q(kv_latent, transpose=False)
        v = self.unembed_out(kv_latent)

    output = scaled_dot_product_attention(
        q, k, v, cache=cache, scale=self.scale, mask=attn_mask
    )
    if absorbed:
        output = self.unembed_out(output)

    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output)


def install(absorb_max_L: int = ABSORB_MAX_L) -> bool:
    """Patch Glm5NextSparseAttention.__call__ in place. Idempotent."""
    global _INSTALLED, _ORIG, ABSORB_MAX_L
    ABSORB_MAX_L = absorb_max_L
    if _INSTALLED:
        return False
    from mlx_vlm.models.glm5_next.language import Glm5NextSparseAttention

    _ORIG = Glm5NextSparseAttention.__call__
    Glm5NextSparseAttention.__call__ = _patched_call
    _INSTALLED = True
    return True


def uninstall() -> bool:
    global _INSTALLED
    if not _INSTALLED:
        return False
    from mlx_vlm.models.glm5_next.language import Glm5NextSparseAttention

    Glm5NextSparseAttention.__call__ = _ORIG
    _INSTALLED = False
    return True


def is_installed() -> bool:
    return _INSTALLED
