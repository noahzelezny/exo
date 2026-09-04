"""GLM-5.3 (glm5_next) MTP drafting head.

Vendored from VQLab: src/vqlab/mtp_head_glm5.py @ 9eff3f0 (2026-09-02).
Copied rather than imported — exo never imports vqlab. Fix bugs THERE
first, then re-vendor. Deviation from that source: none in the module
body; only this header and the `vqlab_mtp` metadata key (kept as-is, so
sidecars built by `vqlab mtp-pack` load here unchanged).

Upstream GLM-5.3-Flash ships its MTP head as a PLAIN layer one past the
trunk (`layers.45` on Flash: eh_proj/enorm/hnorm glue, a NoPE-MLA
sparse-attention block with the DSA indexer, a 288-expert MoE, and its
own `shared_head.norm`). Extract it with VQLab's `vqlab mtp-extract`
(--key-regex '\\.layers\\.45\\.'); this module only LOADS the packed q6
sidecar (`mtp-head-q6.safetensors`, 889 tensors).

Two structural facts, both read off the graft's key set (2026-09-02),
that make this head NOT a trunk `Glm5NextDecoderLayer`:

  - NO hyper-connection weights. The trunk's layers are hc layers; the
    MTP layer is a plain-residual DeepSeek-style block. Instantiating
    the trunk layer class would leave randomly-initialized hc modules
    in the path, so this module assembles the block from the attention
    and MoE sub-modules and runs the residual wiring itself, at
    (B, T, D) — no hc broadcast, no mean-collapse.
  - Its OWN final norm (`shared_head.norm`), applied before the
    trunk's shared lm_head — the DeepSeek MTP convention.

The trunk is NoPE (qk_rope_head_dim=0): there are no rotary positions
to align, so the head-cache offset question that qwen4_exp's head had
to solve does not exist here. Cache offsets still gate the attention
mask, so the committed-alignment scheme (one head row per COMMITTED
token) is kept for the mask math and for parity with the other heads.

The `model` bound here is the mlx_vlm glm5_next LanguageModel (the
object with `.model` = Glm5NextModel and `.args.model_type ==
"glm5_next"`); when holding the full VLM wrapper, pass its
`.language_model`. `arch` is the module those classes live in
(mlx_vlm.models.glm5_next.language), per the registry contract.

MEASURED IN VQLAB, SINGLE BOX, NOT HERE: acceptance 0.8516 pooled over
12 prompts x 128 tokens (q6 head, 2.7bpw trunk, M4, 2026-09-02) and
1.05x end-to-end WITHOUT the absorbed-MLA shim. Nothing in exo has
measured either number on a cluster.
"""
from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

_GLUE = ("eh_proj", "enorm", "hnorm", "shared_head")


class MTPHeadGlm5:
    """One drafting head bound to a loaded glm5_next trunk."""

    def __init__(self, model, arch):
        lang = getattr(model, "language_model", model)
        core = lang.model
        cfg = core.config
        self.model = lang
        self.core = core
        self.arch = arch
        self.cfg = cfg
        self.D = cfg.hidden_size
        self.eps = cfg.rms_norm_eps
        self.tie = cfg.tie_word_embeddings
        self.lm_head = None if self.tie else lang.lm_head

        # The block's halves, straight from the arch module. DeepseekV32MoE
        # is imported into the glm5_next language module's namespace, which
        # keeps this resolution version-proof against mlx_vlm reshuffles.
        self.self_attn = arch.Glm5NextSparseAttention(cfg)
        self.mlp = arch.DeepseekV32MoE(cfg)
        self.input_layernorm = nn.RMSNorm(self.D, eps=self.eps)
        self.post_attention_layernorm = nn.RMSNorm(self.D, eps=self.eps)

        # Glue + the head's own final norm (shared_head.norm upstream).
        self.enorm = nn.RMSNorm(self.D, eps=self.eps)
        self.hnorm = nn.RMSNorm(self.D, eps=self.eps)
        self.eh_proj = nn.Linear(2 * self.D, self.D, bias=False)
        self.final_norm = nn.RMSNorm(self.D, eps=self.eps)

    # ------------------------------------------------------------- caches
    def make_draft_cache(self):
        """The fa-layer cache shape the sparse attention expects:
        CacheList(main-KV, indexer-KV), mlx_vlm classes throughout."""
        return self.arch.CacheList(self.arch.KVCache(), self.arch.KVCache())

    # -------------------------------------------------------------- build
    def _modules(self):
        return {
            "self_attn": self.self_attn,
            "mlp": self.mlp,
            "input_layernorm": self.input_layernorm,
            "post_attention_layernorm": self.post_attention_layernorm,
            "enorm": self.enorm,
            "hnorm": self.hnorm,
            "eh_proj": self.eh_proj,
            "final_norm": self.final_norm,
        }

    def load_graft(self, g, prefix="layers.45."):
        """Fill from the bf16 graft (keys as mtp-extract leaves them).

        Applies the same two transforms the trunk's sanitize applies to
        every layer — stacking the per-expert weights into switch_mlp and
        splitting kv_b_proj into embed_q / unembed_out — then loads the
        rest by name.
        """
        cfg = self.cfg
        w = {k[len(prefix):]: v for k, v in g.items() if k.startswith(prefix)}
        if not w:
            raise SystemExit(f"FAIL: no graft keys under prefix {prefix!r}")

        # 1. Expert stack -> switch_mlp (mirrors DSV32Model.sanitize).
        for m in ("gate_proj", "up_proj", "down_proj"):
            key0 = f"mlp.experts.0.{m}.weight"
            if key0 in w:
                stacked = mx.stack([
                    w.pop(f"mlp.experts.{e}.{m}.weight")
                    for e in range(cfg.n_routed_experts)
                ])
                w[f"mlp.switch_mlp.{m}.weight"] = stacked

        # 2. kv_b_proj split -> embed_q / unembed_out (bf16 path of the
        #    trunk sanitize; the graft is unquantized by construction).
        v = w.pop("self_attn.kv_b_proj.weight")
        head_dim = cfg.qk_nope_head_dim + cfg.v_head_dim
        v = v.reshape(cfg.num_attention_heads, head_dim, -1)
        w["self_attn.embed_q.weight"] = mx.contiguous(
            v[:, : cfg.qk_nope_head_dim, :].swapaxes(-1, -2))
        w["self_attn.unembed_out.weight"] = mx.contiguous(
            v[:, cfg.qk_nope_head_dim :, :])

        # 3. Rename the head's own final norm to this module's slot.
        w["final_norm.weight"] = w.pop("shared_head.norm.weight")

        # 4. Load by name, refusing silent drops in either direction.
        mods = self._modules()
        slots = set()
        for name, mod in mods.items():
            slots |= {f"{name}.{k}" for k, _ in tree_flatten(mod.parameters())}
        unmatched = sorted(set(w) - slots)
        if unmatched:
            raise SystemExit(
                f"FAIL: {len(unmatched)} graft keys found no parameter "
                f"slot, e.g. {unmatched[:4]}")
        missing = sorted(slots - set(w))
        if missing:
            raise SystemExit(
                f"FAIL: {len(missing)} head parameters got no graft "
                f"tensor, e.g. {missing[:4]}")
        for name, mod in mods.items():
            mod.update(tree_unflatten(
                [(k[len(name) + 1:], v) for k, v in w.items()
                 if k.startswith(name + ".")]))
        mx.eval([p for m in mods.values()
                 for _, p in tree_flatten(m.parameters())])
        return self

    def quantize(self, bits=6, group_size=32, expert_bits=None):
        """Quantize the head, mirroring the trunk's own quant_predicate:
        the router gate and the DSA indexer stay 8-bit/gs64 (their
        precision is load-bearing for expert and key selection), the
        expert stack takes `expert_bits` (~96% of the weight), everything
        else `bits`."""
        eb = bits if expert_bits is None else expert_bits

        def predicate(path, mod):
            if not hasattr(mod, "to_quantized"):
                return False
            if path.endswith("mlp.gate") or ".indexer" in path:
                return {"group_size": 64, "bits": 8}
            if ".switch_mlp." in path or path.endswith(
                    ("gate_proj", "up_proj", "down_proj")):
                return {"group_size": group_size, "bits": eb}
            return {"group_size": group_size, "bits": bits}

        for m in (self.self_attn, self.mlp):
            nn.quantize(m, class_predicate=predicate)
        return self

    def save(self, path, bits, group_size=32, expert_bits=None):
        flat = {}
        for name, mod in self._modules().items():
            for k, v in tree_flatten(mod.parameters()):
                flat[f"{name}.{k}"] = v
        mx.save_safetensors(str(path), flat, metadata={
            "format": "mlx",
            "vqlab_mtp": json.dumps({
                "family": "glm5_next", "bits": bits,
                "group_size": group_size, "expert_bits": expert_bits,
            }),
        })
        return flat

    @classmethod
    def from_sidecar(cls, model, arch, path):
        w = mx.load(str(path))
        meta = mx.load(str(path), return_metadata=True)[1]
        cfg = json.loads(meta.get("vqlab_mtp", "{}"))
        head = cls(model, arch)
        bits, gs = cfg.get("bits"), cfg.get("group_size", 32)
        if bits:
            # Replay the exact quantization recipe before filling, or the
            # packed shapes will not line up (same trap as the qwen heads).
            head.quantize(bits=bits, group_size=gs,
                          expert_bits=cfg.get("expert_bits"))
        mods = head._modules()
        for name, mod in mods.items():
            mod.update(tree_unflatten(
                [(k[len(name) + 1:], v) for k, v in w.items()
                 if k.startswith(name + ".")]))
        mx.eval([p for m in mods.values()
                 for _, p in tree_flatten(m.parameters())])
        return head

    # --------------------------------------------------------------- draft
    def _trunk(self, h_row, nxt_id, cache=None):
        """(trunk hidden at t, token t+1) -> the head's output activation.

        `h_row` is the capture at the trunk's final `norm` INPUT — the
        mean-collapsed (B, T, D) hidden, which is what hnorm expects
        (eh_proj is 2D -> D). The block is plain-residual at (B, T, D);
        the mask is built the way the trunk builds its fa mask.
        """
        e = self.enorm(self.core.embed_tokens(nxt_id))
        hs = self.hnorm(h_row)
        x = self.eh_proj(mx.concatenate([e, hs], axis=-1))
        mask = self.arch.create_attention_mask(
            x, cache[0] if cache is not None else None, return_array=True)
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        x = x + r
        x = x + self.mlp(self.post_attention_layernorm(x))
        return self.final_norm(x)

    def draft_logits(self, h_row, nxt_id, cache=None):
        """(trunk hidden at t, token t+1) -> logits for token t+2."""
        out = self._trunk(h_row, nxt_id, cache)
        return (self.core.embed_tokens.as_linear(out) if self.tie
                else self.lm_head(out))

    def advance(self, h_row, nxt_id, cache):
        """Fill the head's cache over these positions without the
        vocab-wide projection (prompt seeding)."""
        self._trunk(h_row, nxt_id, cache)
        return cache
