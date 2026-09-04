"""Per-family registration for MTP speculative decoding.

Adding a family is a table entry here plus a head module. Nothing else in the
package names an architecture; the loop, the caches and the sampler are all
family-agnostic.

A `FamilySpec` says four things, and every one of them is a place where
architectures genuinely differ:

  head             where the drafting head lives ("module:Class"). The class
                   must expose `from_sidecar(model, arch, path)` and
                   `draft_logits(h_row, next_ids, cache)`.
  capture          dotted path, relative to the trunk core, of the submodule
                   whose INPUT is the pre-lm_head activation the head drafts
                   from. There is no public mlx-lm hook for this, so we wrap
                   that one module for the duration of the generation (see
                   capture.py) rather than monkeypatching the class.
  draft_cache      the attribute on the architecture module that builds the
                   head's own KV cache.
  cache_semantics  "reassign" or "copy" — see caches.py. qwen4_exp reassigns
                   its recurrent cache slots rather than mutating them, which
                   makes snapshots free; that is an implementation accident of
                   that arch, NOT a contract, so new families start at "copy"
                   and only move to "reassign" once
                   `caches.check_snapshot_semantics` has been run against them.

Families that ship an MTP head upstream but are NOT registered here, because
nothing in this repo can test them today:

  deepseek_v3 DeepSeek's MTP module is a different shape again (its own
              embed/norm/head rather than a shared lm_head).

(glm5_next graduated 2026-09-03, vendored from VQLab: the old blocker was
"mlx-lm has no glm5_next class", but the trunk runs under mlx_vlm's class —
which exo also uses — so the head binds to THAT arch module.)

Registering either means writing its head module and running
`caches.check_snapshot_semantics` plus the acceptance probe first. A table
entry without a measured acceptance number is not evidence of anything.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class FamilySpec:
    name: str
    head: str
    capture: str
    draft_cache: str
    sidecar_name: str = "mtp-head-q6.safetensors"
    cache_semantics: str = "copy"

    def head_cls(self):
        mod, _, attr = self.head.partition(":")
        return getattr(importlib.import_module(mod), attr)

    def arch_module(self, model):
        """The module the trunk's classes were defined in. Artifacts ship a
        `model.py` that subclasses the registry arch, so walk to the core.

        Vision-capable artifacts (the 397B's `custom_model.Model`, the GLM
        VLM wrapper) have no `.model` of their own -- the core hangs off
        `.language_model` -- so walk that first when it is there. The bound
        text model is what every head class binds to anyway."""
        text = getattr(model, "language_model", model)
        core = getattr(text, "model", None)
        if core is None:
            raise RuntimeError(
                f"family {self.name}: {type(model).__name__} exposes neither "
                f"`.model` nor `.language_model.model`; cannot resolve the "
                f"architecture module")
        return importlib.import_module(type(core).__module__)

    def make_draft_cache(self, arch):
        try:
            return getattr(arch, self.draft_cache)()
        except AttributeError as e:
            raise RuntimeError(
                f"family {self.name}: architecture module {arch.__name__} has "
                f"no {self.draft_cache!r}; the registry entry is stale against "
                f"the installed mlx-lm") from e


FAMILIES: dict[str, FamilySpec] = {}


def register(spec: FamilySpec, *, replace: bool = False) -> FamilySpec:
    if spec.name in FAMILIES and not replace:
        raise ValueError(f"family already registered: {spec.name}")
    if spec.cache_semantics not in ("reassign", "copy"):
        raise ValueError(f"cache_semantics must be 'reassign' or 'copy', "
                         f"got {spec.cache_semantics!r}")
    FAMILIES[spec.name] = spec
    return spec


def unregister(name: str) -> None:
    FAMILIES.pop(name, None)


def model_type_of(model) -> str | None:
    """mlx-lm keeps the resolved config on `model.args`; multimodal configs
    nest the text half, and the top-level type is the one that names the
    architecture module."""
    for obj in (getattr(model, "args", None), model):
        mt = getattr(obj, "model_type", None)
        if isinstance(mt, str):
            return mt
    return None


def resolve(model, family: str | None = None) -> FamilySpec:
    if family is not None:
        if family not in FAMILIES:
            raise KeyError(f"unknown MTP family {family!r}; registered: "
                           f"{sorted(FAMILIES)}")
        return FAMILIES[family]
    mt = model_type_of(model)
    if mt in FAMILIES:
        return FAMILIES[mt]
    raise KeyError(
        f"no MTP family registered for model_type {mt!r}; registered: "
        f"{sorted(FAMILIES)}. Adding one is a FamilySpec in "
        f"exo/worker/engines/mlx/mtp/registry.py plus a head module — read "
        f"that docstring.")


# ------------------------------------------------------------------ builtins
register(FamilySpec(
    name="qwen4_exp",
    head="exo.worker.engines.mlx.mtp.heads.qwen4_exp:MTPHead",
    # The head drafts from the trunk activation that goes INTO the hyper-
    # connection mixer, i.e. the last thing before the final norm + lm_head.
    capture="hyper_connection_mixer",
    draft_cache="_AttnCache",
    sidecar_name="mtp-head-q6.safetensors",
    # Measured: every qwen4_exp cache slot is REASSIGNED (cache[0] = ...),
    # never mutated, and mlx arrays are immutable, so holding the old
    # references is a free snapshot. Verified by
    # caches.check_snapshot_semantics in tests/test_mtp_caches.py.
    cache_semantics="reassign",
))


# Qwen3.5 / Qwen3.8 (`qwen3_5`, and the MoE conditional-generation wrapper
# `qwen3_5_moe`, which is the 397B). One residual stream, so the head drafts
# from the activation going INTO the trunk's final norm -- the same place in
# the graph as qwen4_exp's pre-mixer row, just reached by a different name.
#
# cache_semantics="reassign", and this one is load-bearing for SPEED, not just
# tidiness. The trunk's cache list is mostly recurrent: 48 ArraysCache to 16
# KVCache on the dense 27B, 45 to 15 on the 397B. Attention caches snapshot as
# an offset and cost nothing either way, but under "copy" every one of those
# recurrent GatedDeltaNet states is deep-copied ONCE PER SPECULATIVE STEP --
# hundreds of MB of pure copying per token, which does not fail, it just makes
# generation crawl (measured: a 12-prompt run made no visible progress in 30
# minutes). GatedDeltaNet reassigns its slots (`cache[0] = ...`, `cache[1] =
# state`) and mlx arrays are immutable, so the cheap path is correct here.
# Verified, not assumed: caches.check_snapshot_semantics returned True against
# a loaded 27B (2026-08-31).
for _qwen35_name in ("qwen3_5", "qwen3_5_moe"):
    register(FamilySpec(
        name=_qwen35_name,
        head="exo.worker.engines.mlx.mtp.heads.qwen35:MTPHeadQwen35",
        capture="norm",
        draft_cache="KVCache",
        sidecar_name="mtp-head-q6.safetensors",
        cache_semantics="reassign",
    ))


# GLM-5.3 (glm5_next, via mlx_vlm's classes — the head binds the
# LanguageModel, not the VLM wrapper; `arch_module` above does that walk).
# The head is upstream `layers.45`: a plain-residual DeepSeek-style block
# (NoPE MLA + DSA indexer + 288-expert MoE + its own shared_head.norm) —
# see heads/glm5.py for why it is NOT the trunk's hc DecoderLayer.
# capture="norm": the trunk's final-norm INPUT is the mean-collapsed
# (B, S, D) hidden, which is what hnorm/eh_proj consume.
# draft_cache is vestigial here — the head class provides
# make_draft_cache() (CacheList(main-KV, indexer-KV)) and loop.py prefers
# that; the attribute name is kept non-empty so the spec stays valid.
# cache_semantics="reassign": VQLab's check_snapshot_semantics returned
# True against the loaded 2.7bpw trunk (M4, 2026-09-02).
#
# Registered under both names: the VLM wrapper's config says glm5_next,
# the TextConfig on the bound LanguageModel says glm5_next_text.
#
# NUMBERS MEASURED IN VQLAB ON ONE BOX, NOT IN EXO AND NOT ON A CLUSTER:
# acceptance 0.8516 pooled (12 prompts x 128 tokens, q6 head, 2.7bpw
# trunk, M4, 2026-09-02) and 1.05x end-to-end WITHOUT the absorbed-MLA
# shim. exo installs that shim (glm5_shim.py) whenever this family's head
# loads; its effect on exo's end-to-end rate is UNMEASURED.
for _glm_name in ("glm5_next", "glm5_next_text"):
    register(FamilySpec(
        name=_glm_name,
        head="exo.worker.engines.mlx.mtp.heads.glm5:MTPHeadGlm5",
        capture="norm",
        draft_cache="KVCache",
        sidecar_name="mtp-head-q6.safetensors",
        cache_semantics="reassign",
    ))
