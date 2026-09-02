"""The qwen3_5 / qwen3_5_moe multi-token-prediction head: build, quantize,
save, load.

Structurally far simpler than the qwen4_exp head (vqlab/mtp_head.py): this
architecture has a single residual stream, so there are no hyper-connections
and no per-stream norm statistics, and the head's transformer block is a
stock `DecoderLayer` that owns its own rope.

Wiring, from the checkpoint's own key set:

    h     -> RMSNorm(D)  (pre_fc_norm_hidden)     trunk hidden at position t
    e     -> RMSNorm(D)  (pre_fc_norm_embedding)  embedding of token t+1
    concat -> fc  [D, 2D]  -> D
          -> ONE full-attention DecoderLayer (its own MoE, if the trunk is MoE)
          -> RMSNorm(D)  (the head's own final `norm`)
          -> the shared lm_head

THREE details are not determined by the key set, and each one is a silent
zero-acceptance failure if guessed wrong. All three are therefore flags, and
`vqlab mtp-probe35` sweeps them rather than trusting a guess:

  norm_shift  This family stores RMSNorm gains as DELTAS: mlx-lm's
              `TextModel.sanitize` adds 1.0 to every norm it recognizes, but
              only when the checkpoint still carries unsanitized conv1d
              weights, and it drops every `mtp.*` key before it gets there.
              So a hand-loaded head owns the convention. Measured on the
              source 397B checkpoint (2026-08-31): conv1d.weight is
              [12288, 1, 4], i.e. unsanitized, so the trunk norms ARE shifted
              and the head's must be too. This agrees with the dense 27B
              measurement (0.0000 -> 0.7285 with the shift), and was
              confirmed directly: with the shift the 27B head drafts at
              0.6562, without it at exactly 0.0000 in all four other
              wirings. Default 1.0.

  fc_order    The checkpoint fuses the two input projections into ONE
              `fc.weight` of [D, 2D], so unlike qwen4_exp (separate
              fc_embedding / fc_hidden tensors) the concat order is not
              recoverable from the file. Measured on the dense 27B
              (2026-08-31, 512 positions): "eh" = [embedding | hidden] gives
              0.6562 acceptance, "he" gives 0.0020 -- i.e. chance. Default
              "eh".

  h_source    Whether the head consumes the trunk's hidden state before or
              after the trunk's own final norm. Measured: pre_norm 0.6562 vs
              post_norm 0.6582 over 512 positions -- a ONE-token difference,
              so this flag is not resolved by that experiment and probably
              cannot be: `pre_fc_norm_hidden` is applied immediately after,
              and an RMSNorm of an already-normed vector is close to
              idempotent. Default "pre_norm", which is what the head carrying
              its own hidden norm argues for.

Head precision cannot affect output quality -- the trunk verifies every
drafted token, so a coarser head costs a rejection, never a wrong token.
"""
from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

SIDECAR_NAME = "mtp-head-q6.safetensors"

# The MoE expert stack dominates the head's weight; the router and the
# shared-expert gate are tiny and stay full precision, because a router error
# changes which experts run, not just how precisely they run.
_EXPERT_SUBSTR = "switch_mlp"
_KEEP_FP = ("mlp.gate", "shared_expert_gate")

# Which of the head's norms the +1.0 delta convention applies to. mlx-lm's
# own sanitize shifts exactly these suffixes on the trunk; the head's fc
# norms are shifted too because the convention is a property of how the
# checkpoint was written, not of which module reads the tensor.
_NORM_KEYS = (
    "norm.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.k_norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)


def _mixed_predicate(bits, expert_bits, group_size):
    def predicate(path, mod):
        if not hasattr(mod, "to_quantized"):
            return False
        if any(path.endswith(s) for s in _KEEP_FP):
            return False
        b = expert_bits if _EXPERT_SUBSTR in path else bits
        return {"group_size": group_size, "bits": b}
    return predicate


class MTPHeadQwen35:
    """One drafting head bound to a loaded qwen3_5 / qwen3_5_moe trunk."""

    def __init__(self, model, arch, *, fc_order="eh", h_source="pre_norm",
                 norm_shift=1.0):
        if fc_order not in ("he", "eh"):
            raise ValueError(f"fc_order must be 'he' or 'eh', got {fc_order!r}")
        if h_source not in ("pre_norm", "post_norm"):
            raise ValueError(f"h_source must be 'pre_norm' or 'post_norm', "
                             f"got {h_source!r}")
        text = getattr(model, "language_model", model)
        self.model = model
        self.text = text
        self.core = text.model
        self.arch = arch
        self.args = args = text.args
        self.fc_order = fc_order
        self.h_source = h_source
        self.norm_shift = float(norm_shift)
        self.tie = bool(args.tie_word_embeddings)
        self.lm_head = None if self.tie else text.lm_head
        self.D = args.hidden_size
        self.eps = getattr(args, "rms_norm_eps", 1e-6)
        # A full-attention layer index: DecoderLayer derives is_linear from
        # (idx + 1) % full_attention_interval, so this is the head's block.
        self.fa_idx = args.full_attention_interval - 1
        self.block = arch.DecoderLayer(args, self.fa_idx)
        if self.block.is_linear:      # cheap, but the whole head hangs on it
            raise RuntimeError(
                f"fa_idx {self.fa_idx} produced a linear-attention layer; the "
                f"MTP head must be full attention")
        self.norm_e = None
        self.norm_h = None
        self.norm_out = None
        self.fc = None

    # ---------------------------------------------------------------- build
    def _norm(self, w, shift):
        n = nn.RMSNorm(self.D, eps=self.eps)
        n.weight = w.astype(mx.float32) + shift
        return n

    def load_graft(self, g):
        """Fill from an upstream bf16 `mtp.*` graft (the `mtp.` prefix already
        stripped). The 397B checkpoint stores the head's experts UNFUSED, one
        tensor per expert per projection (512 x 3 = 1536 tensors), where the
        trunk stores them fused; stack them into the SwitchGLU layout."""
        shift = self.norm_shift
        layer_w = {}
        experts = {}
        for k, v in g.items():
            if not k.startswith("layers.0."):
                continue
            k = k[len("layers.0."):]
            if k.endswith("mlp.experts.gate_up_proj"):        # fused variant
                mid = v.shape[-2] // 2
                layer_w["mlp.switch_mlp.gate_proj.weight"] = v[..., :mid, :]
                layer_w["mlp.switch_mlp.up_proj.weight"] = v[..., mid:, :]
            elif k.endswith("mlp.experts.down_proj"):
                layer_w["mlp.switch_mlp.down_proj.weight"] = v
            elif k.startswith("mlp.experts."):                # unfused variant
                _, _, idx, proj, _ = k.split(".", 4)
                experts.setdefault(proj, {})[int(idx)] = v
            else:
                layer_w[k] = v
        for proj, byidx in experts.items():
            n = max(byidx) + 1
            if len(byidx) != n:
                raise SystemExit(f"FAIL: expert {proj} has {len(byidx)} "
                                 f"tensors but indices reach {n - 1}")
            layer_w[f"mlp.switch_mlp.{proj}.weight"] = mx.stack(
                [byidx[i] for i in range(n)], axis=0)

        for k in list(layer_w):
            if shift and any(("layers.0." + k).endswith(s) for s in _NORM_KEYS):
                layer_w[k] = layer_w[k].astype(mx.float32) + shift

        slots = {k for k, _ in tree_flatten(self.block.parameters())}
        unmatched = sorted(set(layer_w) - slots)
        if unmatched:
            raise SystemExit(f"FAIL: {len(unmatched)} graft keys found no "
                             f"parameter slot, e.g. {unmatched[:4]}")
        missing = sorted(slots - set(layer_w))
        if missing:
            print(f"note: {len(missing)} module params not in graft (left "
                  f"init), e.g. {missing[:4]}", flush=True)
        self.block.load_weights(list(layer_w.items()), strict=False)

        self.norm_e = self._norm(g["pre_fc_norm_embedding.weight"], shift)
        self.norm_h = self._norm(g["pre_fc_norm_hidden.weight"], shift)
        self.norm_out = self._norm(g["norm.weight"], shift)
        self.fc = g["fc.weight"]
        if self.fc.shape != (self.D, 2 * self.D):
            raise SystemExit(f"FAIL: fc.weight is {self.fc.shape}, expected "
                             f"{(self.D, 2 * self.D)}")
        mx.eval(self.block.parameters())
        return self

    def quantize(self, bits=6, group_size=32, expert_bits=None):
        eb = bits if expert_bits is None else expert_bits
        nn.quantize(self.block, class_predicate=_mixed_predicate(
            bits, eb, group_size))
        mx.eval(self.block.parameters())
        return self

    # ------------------------------------------------------------- sidecar
    def _config(self, bits, group_size, expert_bits):
        return {"bits": bits, "group_size": group_size,
                "expert_bits": expert_bits, "fa_idx": self.fa_idx,
                "fc_order": self.fc_order, "h_source": self.h_source,
                # Already applied to the stored tensors; recorded so a sidecar
                # says which convention produced it, never re-applied on load.
                "norm_shift": self.norm_shift}

    def save(self, path, bits, group_size=32, expert_bits=None):
        flat = {f"block.{k}": v
                for k, v in tree_flatten(self.block.parameters())}
        flat["norm_e.weight"] = self.norm_e.weight
        flat["norm_h.weight"] = self.norm_h.weight
        flat["norm_out.weight"] = self.norm_out.weight
        flat["fc.weight"] = self.fc
        mx.save_safetensors(str(path), flat, metadata={
            "format": "mlx",
            "vqlab_mtp": json.dumps(self._config(bits, group_size,
                                                 expert_bits)),
        })
        return flat

    @classmethod
    def from_sidecar(cls, model, arch, path):
        w = mx.load(str(path))
        meta = mx.load(str(path), return_metadata=True)[1]
        cfg = json.loads(meta.get("vqlab_mtp", "{}"))
        head = cls(model, arch,
                   fc_order=cfg.get("fc_order", "eh"),
                   h_source=cfg.get("h_source", "pre_norm"),
                   norm_shift=cfg.get("norm_shift", 1.0))
        bits, gs = cfg.get("bits"), cfg.get("group_size", 32)
        if bits:
            # Replay the recipe exactly: a mixed-bit sidecar rebuilt as
            # uniform gives every expert tensor the wrong packed shape.
            eb = cfg.get("expert_bits") or bits
            nn.quantize(head.block,
                        class_predicate=_mixed_predicate(bits, eb, gs))
        head.block.update(tree_unflatten(
            [(k[len("block."):], v) for k, v in w.items()
             if k.startswith("block.")]))
        # The shift is baked into the stored tensors; pass 0.0 so loading a
        # sidecar can never double-count it.
        head.norm_e = head._norm(w["norm_e.weight"], 0.0)
        head.norm_h = head._norm(w["norm_h.weight"], 0.0)
        head.norm_out = head._norm(w["norm_out.weight"], 0.0)
        head.fc = w["fc.weight"]
        mx.eval(head.block.parameters())
        return head

    # --------------------------------------------------------------- draft
    def _trunk(self, h_row, nxt_id, cache=None):
        """(trunk hidden at t, token t+1) -> the head's output activation.

        T > 1 is a real case, not just prefill: aligning the head's cache to
        one row per COMMITTED token advances it two positions per speculative
        step. The mask must therefore be built the way the trunk builds its
        own; passing None is correct only at T == 1, and at T > 1 it silently
        lets each position attend forwards, which shows up as degraded
        acceptance rather than as an error.
        """
        if self.h_source == "post_norm":
            h_row = self.core.norm(h_row)
        e = self.norm_e(self.core.embed_tokens(nxt_id))
        h = self.norm_h(h_row)
        cat = mx.concatenate([h, e] if self.fc_order == "he" else [e, h],
                             axis=-1)
        hin = cat @ self.fc.T.astype(cat.dtype)
        mask = (self.arch.create_attention_mask(hin, cache)
                if hin.shape[1] > 1 else None)
        return self.norm_out(self.block(hin, mask, cache))

    def draft_logits(self, h_row, nxt_id, cache=None):
        """(trunk hidden at t, token t+1) -> logits for token t+2."""
        out = self._trunk(h_row, nxt_id, cache)
        return (self.core.embed_tokens.as_linear(out) if self.tie
                else self.lm_head(out))

    def advance(self, h_row, nxt_id, cache):
        """Fill the head's cache for these positions without projecting to the
        vocabulary -- over a whole prompt the lm_head matmul would be the
        single largest cost of seeding."""
        self._trunk(h_row, nxt_id, cache)
        return cache
