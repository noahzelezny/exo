"""The qwen4_exp multi-token-prediction head: build, quantize, save, load.

One module so the wiring lives in exactly one place. Every detail below was
settled by measurement (2026-08-30); see mtp_probe.py for the evidence and
research/flash-next/LEDGER.md for the numbers.

Wiring, per the llama.cpp qwen4-exp port (PR #27739) with the ambiguities
resolved against the architecture itself:

    h_row -> RMSNorm(hc*D, group_size=D)   one statistic per stream, as every
                                           other wide norm in this arch does
                                           (measured 0.6992 vs 0.6562 flat)
          -> reshape to [hc, D] streams
    e     -> RMSNorm(D)                    shared across streams
    per stream: fc_embedding @ e + fc_hidden @ h_stream
          -> ONE standard qwen4_exp full-attention block (own 512-expert MoE)
          -> the head's own hyper_connection_mixer (carries the final norm)
          -> the shared lm_head / tied embedding

The norms MUST be applied by the architecture's own RMSNorm, which is
zero-centered (y = norm(x) * (1 + weight)). Hand-rolling `n * w` drops the
+1.0 and drives draft acceptance to exactly 0.0 -- that single mistake was
the entire reason the head looked dead. Do not "simplify" it back.
"""
from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx.utils import tree_flatten, tree_unflatten

SIDECAR_NAME = "mtp-head-q6.safetensors"


def _quantizable(path, mod):
    """nn.quantize hands the predicate EVERY submodule, norms included, and
    raises on anything without to_quantized -- so gate on that, not on names.
    The MoE router stays full precision, mirroring qwen4_exp's own
    quant_predicate."""
    return hasattr(mod, "to_quantized") and not path.endswith("mlp.gate")


# Measured from the bf16 graft header (2026-08-31): the 512-expert MoE stack
# is 4.688 of the head's 4.856 GiB -- 96.5% of it, in TWO tensors. Everything
# else together (attention, hyper-connections, the fc fuse, the shared expert,
# the norms) is 0.168 GiB. So head size is essentially a single dial, the
# expert bit-width, and protecting all the small modules at a high bit-width
# is nearly free: +0.02 GiB going from 6-bit to 8-bit across all of them.
#
# This is a pure speed/memory search. Head precision CANNOT affect output
# quality -- the trunk verifies every drafted token, so a coarser head costs
# a rejection, never a wrong token. The only risk is acceptance, and acceptance
# is measurable directly (`vqlab mtp-probe`).
_EXPERT_SUBSTR = "switch_mlp"


def _mixed_predicate(bits, expert_bits, group_size):
    """Per-module bit-widths. nn.quantize lets a predicate return the kwargs
    for to_quantized, so one pass can carry two bit-widths."""
    def predicate(path, mod):
        if not _quantizable(path, mod):
            return False
        b = expert_bits if _EXPERT_SUBSTR in path else bits
        return {"group_size": group_size, "bits": b}
    return predicate


class MTPHead:
    """One drafting head bound to a loaded qwen4_exp trunk."""

    def __init__(self, model, arch):
        core = model.model
        args_t = core.args
        self.model = model
        self.core = core
        self.arch = arch
        self.args = args_t
        self.tie = model.args.text.tie_word_embeddings
        self.lm_head = None if self.tie else model.lm_head
        self.D = args_t.hidden_size
        self.hc = core.hc
        self.eps = getattr(args_t, "rms_norm_eps", 1e-6)
        fa_idx = [i for i, l in enumerate(core.layers)
                  if l.layer_type == "full_attention"][0]
        self.fa_idx = fa_idx
        self.block = type(core.layers[fa_idx])(args_t, fa_idx)
        # The head has no PLE bank. A zero-filled stand-in is NOT a no-op
        # through this class, so the submodule has to go entirely.
        self.block.ple = None
        self.mixer = type(core.hyper_connection_mixer)(args_t, use_combine=False)
        theta = float((getattr(args_t, "mtp", {}) or {}).get(
            "rope_theta", 10_000_000))
        self.rope = type(core.rope)(core.rope.dim, theta)
        self.norm_e = None
        self.norm_h = None
        self.fc = None

    # ---------------------------------------------------------------- build
    def _norm(self, dim, w, group_size=None):
        n = self.arch.RMSNorm(dim, group_size=group_size, eps=self.eps)
        n.weight = w.astype(mx.float32)
        return n

    def load_graft(self, g):
        """Fill from an upstream bf16 `mtp.*` graft (keys already stripped)."""
        layer_w = {}
        for k, v in g.items():
            if not k.startswith("layers.0."):
                continue
            k = k[len("layers.0."):]
            # Upstream stacks the experts fused; apply the same split the
            # model's own sanitize applies to trunk layers.
            if k.endswith("mlp.experts.gate_up_proj"):
                mid = v.shape[-2] // 2
                layer_w["mlp.switch_mlp.gate_proj.weight"] = v[..., :mid, :]
                layer_w["mlp.switch_mlp.up_proj.weight"] = v[..., mid:, :]
            elif k.endswith("mlp.experts.down_proj"):
                layer_w["mlp.switch_mlp.down_proj.weight"] = v
            else:
                layer_w[k] = v
        slots = {k for k, _ in tree_flatten(self.block.parameters())}
        unmatched = sorted(set(layer_w) - slots)
        if unmatched:
            raise SystemExit(f"FAIL: {len(unmatched)} graft keys found no "
                             f"parameter slot, e.g. {unmatched[:4]}")
        self.block.load_weights(list(layer_w.items()), strict=False)
        self.mixer.load_weights(
            [(k[len("hyper_connection_mixer."):], v) for k, v in g.items()
             if k.startswith("hyper_connection_mixer.")], strict=False)
        self.norm_e = self._norm(self.D, g["pre_fc_norm_embedding.weight"])
        self.norm_h = self._norm(self.hc * self.D,
                                 g["pre_fc_norm_hidden.weight"],
                                 group_size=self.D)
        self.fc = mx.concatenate([g["fc_embedding.weight"],
                                  g["fc_hidden.weight"]], axis=1)
        mx.eval(self.block.parameters(), self.mixer.parameters())
        return self

    def quantize(self, bits=6, group_size=32, expert_bits=None):
        """Quantize the head. `expert_bits` overrides `bits` for the MoE
        expert stack, which is 96.5% of the weight; `bits` then applies to the
        3.5% of small modules, where protecting precision costs almost
        nothing. `expert_bits=None` reproduces the uniform recipe."""
        eb = bits if expert_bits is None else expert_bits
        for m in (self.block, self.mixer):
            nn.quantize(m, class_predicate=_mixed_predicate(
                bits, eb, group_size))
        mx.eval(self.block.parameters(), self.mixer.parameters())
        return self

    # ------------------------------------------------------------- sidecar
    def save(self, path, bits, group_size=32, expert_bits=None):
        flat = {f"block.{k}": v
                for k, v in tree_flatten(self.block.parameters())}
        flat.update({f"mixer.{k}": v
                     for k, v in tree_flatten(self.mixer.parameters())})
        flat["norm_e.weight"] = self.norm_e.weight
        flat["norm_h.weight"] = self.norm_h.weight
        flat["fc.weight"] = self.fc
        mx.save_safetensors(str(path), flat, metadata={
            "format": "mlx",
            "mtplx_compatible": "false",
            "vqlab_mtp": json.dumps({"bits": bits, "group_size": group_size,
                                     "expert_bits": expert_bits,
                                     "fa_idx": self.fa_idx}),
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
            # Build the quantized module shape first, then fill it. The recipe
            # must be replayed exactly -- a mixed-bit sidecar loaded as uniform
            # gives every expert tensor the wrong packed shape.
            eb = cfg.get("expert_bits") or bits
            for m in (head.block, head.mixer):
                nn.quantize(m, class_predicate=_mixed_predicate(bits, eb, gs))
        head.block.update(tree_unflatten(
            [(k[len("block."):], v) for k, v in w.items()
             if k.startswith("block.")]))
        head.mixer.update(tree_unflatten(
            [(k[len("mixer."):], v) for k, v in w.items()
             if k.startswith("mixer.")]))
        head.norm_e = head._norm(head.D, w["norm_e.weight"])
        head.norm_h = head._norm(head.hc * head.D, w["norm_h.weight"],
                                 group_size=head.D)
        head.fc = w["fc.weight"]
        mx.eval(head.block.parameters(), head.mixer.parameters())
        return head

    # --------------------------------------------------------------- draft
    def _trunk(self, h_row, nxt_id, cache=None):
        """(trunk hidden at t, token t+1) -> the head's output activation.

        T > 1 is a real case, not just prefill: aligning the head's cache to
        one row per COMMITTED token means advancing it two positions per
        speculative step. The mask MUST therefore be built the way the trunk
        builds its own (`create_attention_mask`) — passing None is only
        correct at T == 1, and at T > 1 it silently lets each position attend
        forwards, which shows up as degraded acceptance rather than an error.
        """
        core, D, hc = self.core, self.D, self.hc
        B, T = nxt_id.shape
        e = self.norm_e(core.embed_tokens(nxt_id))
        hs = self.norm_h(h_row).reshape(B, T, hc, D)
        es = mx.broadcast_to(e[:, :, None, :], (B, T, hc, D))
        cat = mx.concatenate([es, hs], axis=-1)
        hin = (cat @ self.fc.T.astype(cat.dtype)).reshape(B, T, hc * D)
        idx = cache.indexer if (cache is not None
                                and hasattr(cache, "indexer")) else None
        mask = create_attention_mask(hin, cache) if T > 1 else None
        out = self.block(hin, self.rope, mask, None, cache, idx, nxt_id, None)
        return self.mixer(out)

    def draft_logits(self, h_row, nxt_id, cache=None):
        """(trunk hidden at t, token t+1) -> logits for token t+2."""
        out = self._trunk(h_row, nxt_id, cache)
        return (self.core.embed_tokens.as_linear(out) if self.tie
                else self.lm_head(out))

    def advance(self, h_row, nxt_id, cache):
        """Fill the head's cache for these positions without projecting to
        the vocabulary. Used to seed the head over the prompt, where the
        logits are never read and the lm_head matmul over the whole prompt
        would be the single largest cost of the seed."""
        self._trunk(h_row, nxt_id, cache)
        return cache
