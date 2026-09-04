"""GLM-5.3 (glm5_next) MTP: registry entry, head sidecar load, the
absorbed-MLA shim, and the wrapper walk the loop needs.

What each group here is actually pinning down:

  registry      `resolve` on a glm5_next model must NOT raise. The live
                failure this file was written against was
                `KeyError: no MTP family registered for model_type
                'glm5_next'`, raised inside `plan_mtp -> _load_head`.
  arch walk     GLM artifacts are image-text-to-text, so the object exo
                holds is the VLM wrapper: no `.model`, only
                `.language_model.model`. Both `FamilySpec.arch_module` and
                the loop's capture point have to walk that, and BOTH
                walks are exercised below (the second through the real
                `mtp_stream_generate`).
  head          `from_sidecar` round-trips against a stand-in arch module,
                so the load path is covered without a 100 GiB artifact.
  shim          the L <= 8 absorbed route is selected for a 2-token
                verify, and the shim is inert unless a glm5_next head
                actually loads.
  stock path    EXO_MTP unset -> no plan, no shim, nothing patched.

NOT covered here, and it cannot be without the cluster: acceptance,
end-to-end tok/s, and whether the real `Glm5NextSparseAttention` weights
produce sane drafts. The stand-in arch below is shaped like the real one,
not equal to it.
"""

from __future__ import annotations

import sys
import types

import mlx.core as mx
import mlx.nn as nn
import pytest

from exo.worker.engines.mlx.mtp import glm5_shim, registry
from exo.worker.engines.mlx.mtp.heads.glm5 import MTPHeadGlm5
from exo.worker.engines.mlx.mtp.loop import load_mtp_head, mtp_stream_generate

D = 8
VOCAB = 16


# ------------------------------------------------------- stand-in arch module
#
# These live in THIS module on purpose: `FamilySpec.arch_module` resolves the
# arch as `importlib.import_module(type(core).__module__)`, so defining the
# trunk core here makes this test module the arch module, and the names the
# head asks for (`Glm5NextSparseAttention`, `DeepseekV32MoE`, `CacheList`,
# `KVCache`, `create_attention_mask`) are found here.


class KVCache:
    """Attention-cache shape `caches.snapshot`/`restore` require: an offset,
    a `keys` attribute, and a `trim` that moves the offset back."""

    def __init__(self) -> None:
        self.offset = 0
        self.keys = None

    def trim(self, n: int) -> int:
        self.offset -= n
        return n


class CacheList(list):
    def __init__(self, *caches) -> None:  # noqa: ANN002
        super().__init__(caches)

    @property
    def offset(self) -> int:
        return self[0].offset


def create_attention_mask(x, cache=None, return_array: bool = False):  # noqa: ANN001
    return None


class Glm5NextSparseAttention(nn.Module):
    """Shape-compatible stand-in: one linear, and it advances the cache the
    way the real block does, because the head's cache offset is what the
    committed alignment scheme asserts on."""

    def __init__(self, cfg) -> None:  # noqa: ANN001
        super().__init__()
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def __call__(self, x, mask=None, cache=None):  # noqa: ANN001
        if cache is not None:
            for c in cache:
                c.offset += int(x.shape[1])
        return self.o_proj(x)


class DeepseekV32MoE(nn.Module):
    def __init__(self, cfg) -> None:  # noqa: ANN001
        super().__init__()
        self.down = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def __call__(self, x):  # noqa: ANN001
        return self.down(x)


class _Cfg:
    hidden_size = D
    rms_norm_eps = 1e-6
    tie_word_embeddings = True
    num_attention_heads = 2
    qk_nope_head_dim = 2
    v_head_dim = 2
    n_routed_experts = 2


class _Core(nn.Module):
    """Stands in for Glm5NextModel: an embedding table and a final `norm`,
    which is this family's capture point."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _Cfg()
        self.embed_tokens = nn.Embedding(VOCAB, D)
        self.norm = nn.RMSNorm(D, eps=1e-6)


class _Args:
    model_type = "glm5_next"
    hidden_size = D


class _LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Core()
        self.args = _Args()


class _VLMWrapper(nn.Module):
    """What exo actually holds for a GLM-5.3-Flash-VQ artifact: an
    image-text-to-text wrapper with no `.model` of its own."""

    def __init__(self) -> None:
        super().__init__()
        self.language_model = _LanguageModel()
        self.args = _Args()


# ------------------------------------------------------------------ registry


@pytest.mark.parametrize("name", ["glm5_next", "glm5_next_text"])
def test_registry_has_the_glm_family(name: str):
    spec = registry.FAMILIES[name]
    assert spec.head == "exo.worker.engines.mlx.mtp.heads.glm5:MTPHeadGlm5"
    assert spec.capture == "norm"
    assert spec.sidecar_name == "mtp-head-q6.safetensors"
    assert spec.cache_semantics == "reassign"
    assert spec.head_cls() is MTPHeadGlm5


def test_resolve_no_longer_raises_for_glm5_next():
    """The exact live failure: resolve(model) raised KeyError under
    plan_mtp -> _load_head, and the runner logged 'MTP head load failed'."""
    spec = registry.resolve(_VLMWrapper())
    assert spec.name == "glm5_next"


def test_arch_module_walks_the_vlm_wrapper():
    spec = registry.FAMILIES["glm5_next"]
    assert spec.arch_module(_VLMWrapper()) is sys.modules[__name__]
    # And the plain LanguageModel still resolves the same way.
    assert spec.arch_module(_LanguageModel()) is sys.modules[__name__]


def test_arch_module_says_so_when_neither_attribute_exists():
    class _Neither(nn.Module):
        pass

    with pytest.raises(RuntimeError, match="language_model"):
        registry.FAMILIES["glm5_next"].arch_module(_Neither())


# ---------------------------------------------------------------------- head


def _fresh_head(model=None):
    model = model or _VLMWrapper()
    return MTPHeadGlm5(model, sys.modules[__name__]), model


def test_head_round_trips_through_a_sidecar(tmp_path):
    head, model = _fresh_head()
    path = tmp_path / "mtp-head-q6.safetensors"
    # bits=None: no quantization to replay, so this exercises the load path
    # without needing a real q6 pack.
    head.save(path, bits=None)

    loaded = MTPHeadGlm5.from_sidecar(model, sys.modules[__name__], path)
    for name in ("enorm", "hnorm", "eh_proj", "final_norm"):
        a = getattr(head, name).parameters()
        b = getattr(loaded, name).parameters()
        for k in a:
            assert mx.allclose(a[k], b[k]), f"{name}.{k} did not round-trip"


def test_load_mtp_head_uses_the_registry_end_to_end(tmp_path):
    """The real loader: resolve -> arch_module -> head_cls().from_sidecar,
    against the VLM wrapper rather than a bare text model."""
    head, model = _fresh_head()
    path = tmp_path / "mtp-head-q6.safetensors"
    head.save(path, bits=None)

    loaded, spec = load_mtp_head(model, sidecar=path)
    assert spec.name == "glm5_next"
    assert isinstance(loaded, MTPHeadGlm5)


def test_head_drafts_a_row_per_position(tmp_path):
    head, _ = _fresh_head()
    cache = head.make_draft_cache()
    assert isinstance(cache, CacheList) and len(cache) == 2

    h = mx.zeros((1, 2, D))
    ids = mx.array([[3, 4]])
    logits = head.draft_logits(h, ids, cache)
    assert logits.shape == (1, 2, VOCAB)
    # One cache row per position consumed -- the committed-alignment
    # invariant the loop's rollback arithmetic depends on.
    assert cache[0].offset == 2

    head.advance(mx.zeros((1, 1, D)), mx.array([[5]]), cache)
    assert cache[0].offset == 3


# ---------------------------------------------------------------------- shim


class _RecordingAttn:
    """Just enough of `Glm5NextSparseAttention` for the shim's `self`."""

    def __init__(self) -> None:
        self.scale = 1.0
        self.num_heads = 2
        self.q_head_dim = 4
        self.embed_q_calls: list[tuple] = []
        self.unembed_calls: list[tuple] = []

    # the projections the two routes disagree about
    def embed_q(self, x, transpose: bool = True):  # noqa: ANN001
        self.embed_q_calls.append((tuple(x.shape), transpose))
        return x

    def unembed_out(self, x):  # noqa: ANN001
        self.unembed_calls.append(tuple(x.shape))
        return x

    # the parts the shim copies verbatim from upstream
    def q_a_layernorm(self, x):  # noqa: ANN001
        return x

    def q_a_proj(self, x):  # noqa: ANN001
        return x

    def q_b_proj(self, x):  # noqa: ANN001
        batch, width, _ = x.shape
        return mx.zeros((batch, width, self.num_heads * self.q_head_dim))

    def kv_a_proj_with_mqa(self, x):  # noqa: ANN001
        return x

    def kv_a_layernorm(self, x):  # noqa: ANN001
        return x

    def indexer(self, x, qr, mask, cache=None):  # noqa: ANN001
        return None

    def o_proj(self, x):  # noqa: ANN001
        return x


@pytest.fixture()
def _stub_sdpa(monkeypatch):
    """`_patched_call` imports mlx_vlm lazily; stub the module so this test
    does not depend on mlx_vlm being installed in the test env."""
    mod = types.ModuleType("mlx_vlm.models.base")

    def sdpa(q, k, v, cache=None, scale=1.0, mask=None):  # noqa: ANN001
        # Shape-only stand-in: [B, H, L, dk] out.
        return mx.zeros(q.shape)

    mod.scaled_dot_product_attention = sdpa
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.models", types.ModuleType("mlx_vlm.models"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.base", mod)
    return mod


def _route(width: int) -> str:
    """Run the patched __call__ at query width `width`; report which route ran."""
    attn = _RecordingAttn()
    x = mx.zeros((1, width, D))
    glm5_shim._patched_call(attn, x, None, None)
    # Absorbed: embed_q is applied to the QUERY [B, H, L, dk] and unembed_out
    # to the OUTPUT. Unabsorbed: embed_q is applied to the latent cache with
    # transpose=False, and unembed_out to the latent cache.
    absorbed = not any(not t for _, t in attn.embed_q_calls)
    return "absorbed" if absorbed else "unabsorbed"


@pytest.mark.usefixtures("_stub_sdpa")
@pytest.mark.parametrize("width", [1, 2, 3, 8])
def test_shim_takes_the_absorbed_route_for_small_windows(width: int):
    """L == 2 is the MTP verify forward. Upstream falls off the absorbed
    route at L > 1 and pays a latent-cache expansion VQLab measured at up
    to ~40x at long Kv -- which would make drafting a net loss."""
    assert _route(width) == "absorbed"


@pytest.mark.usefixtures("_stub_sdpa")
@pytest.mark.parametrize("width", [9, 64])
def test_shim_leaves_wide_forwards_unabsorbed(width: int):
    """Prefill chunks stay on the upstream route: the expansion amortises
    over many queries there, which is why upstream chose it."""
    assert _route(width) == "unabsorbed"


def test_install_is_idempotent_and_reversible(monkeypatch):
    """install()/uninstall() against a stand-in mlx_vlm module, so the test
    does not patch a real class out from under other tests."""

    class _Victim:
        def __call__(self, x, mask=None, cache=None):  # noqa: ANN001
            return "stock"

    lang = types.ModuleType("mlx_vlm.models.glm5_next.language")
    lang.Glm5NextSparseAttention = _Victim
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.models", types.ModuleType("mlx_vlm.models"))
    monkeypatch.setitem(
        sys.modules, "mlx_vlm.models.glm5_next", types.ModuleType("mlx_vlm.models.glm5_next")
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.glm5_next.language", lang)
    monkeypatch.setattr(glm5_shim, "_INSTALLED", False)
    monkeypatch.setattr(glm5_shim, "_ORIG", None)

    stock = _Victim.__call__
    assert glm5_shim.install() is True
    assert glm5_shim.is_installed()
    assert _Victim.__call__ is glm5_shim._patched_call
    assert glm5_shim.install() is False          # idempotent
    assert glm5_shim.uninstall() is True
    assert _Victim.__call__ is stock
    assert glm5_shim.uninstall() is False


# ------------------------------------------------------- the gate, for real
#
# `_load_head` is the function that raised in the live run, so these call it
# rather than the monkeypatched stand-in the stage-1 gate tests use.


@pytest.fixture(autouse=True)
def _clean_head_caches():
    from exo.worker.engines.mlx.mtp import speculative

    speculative._HEAD_CACHE.clear()
    speculative._HEAD_FAILED.clear()
    yield
    speculative._HEAD_CACHE.clear()
    speculative._HEAD_FAILED.clear()


def _sidecar_dir(monkeypatch, tmp_path):
    from exo.worker.engines.mlx.mtp import speculative

    head, model = _fresh_head()
    head.save(tmp_path / "mtp-head-q6.safetensors", bits=None)
    monkeypatch.setattr(speculative, "build_model_path", lambda _id: tmp_path)
    return speculative, model


def test_load_head_resolves_glm_and_installs_the_shim(monkeypatch, tmp_path):
    spec_mod, model = _sidecar_dir(monkeypatch, tmp_path)
    installs: list[int] = []
    monkeypatch.setattr(glm5_shim, "install", lambda *a, **k: installs.append(1) or True)

    head = spec_mod._load_head(model, "TheDrainFlorist/GLM-5.3-Flash-VQ-2.7bpw")
    assert isinstance(head, MTPHeadGlm5)
    assert installs == [1], "the absorbed-MLA shim was not installed for glm5_next"


def test_the_shim_is_not_installed_for_other_families(monkeypatch):
    from exo.worker.engines.mlx.mtp import speculative

    installs: list[int] = []
    monkeypatch.setattr(glm5_shim, "install", lambda *a, **k: installs.append(1) or True)
    assert (
        speculative._maybe_install_glm5_shim(registry.FAMILIES["qwen3_5"]) is False
    )
    assert installs == []


def test_stock_path_is_untouched_without_the_env_var(monkeypatch, tmp_path):
    """EXO_MTP unset: no plan, and nothing gets monkeypatched — the whole
    point of the default-off gate."""
    from exo.worker.engines.mlx.mtp import speculative

    monkeypatch.delenv("EXO_MTP", raising=False)
    installs: list[int] = []
    monkeypatch.setattr(glm5_shim, "install", lambda *a, **k: installs.append(1) or True)
    monkeypatch.setattr(speculative, "build_model_path", lambda _id: tmp_path)

    assert speculative.plan_mtp(_VLMWrapper(), "glm", None, has_vision=False) is None
    assert installs == []
    assert not glm5_shim.is_installed()


@pytest.mark.usefixtures("_clean_head_caches")
def test_a_text_only_request_on_a_vision_capable_model_may_draft(
    monkeypatch, tmp_path
):
    """The gate refuses a vision REQUEST, not a vision-capable MODEL.

    Every GLM-5.3-Flash-VQ artifact is image-text-to-text, so gating on the
    model's capability would refuse the requests the head exists for.
    `generator/generate.py` passes `vision is not None`, and `prepare_vision`
    returns None whenever the request carries no images.
    """
    spec_mod, model = _sidecar_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("EXO_MTP", "1")
    monkeypatch.setattr(glm5_shim, "install", lambda *a, **k: True)

    plan = spec_mod.plan_mtp(model, "glm", None, has_vision=False)
    assert plan is not None and plan.drafts and plan.stage == 0

    # The same model, with images attached to the request, is refused.
    assert spec_mod.plan_mtp(model, "glm", None, has_vision=True) is None


# ------------------------------------------- the loop, through a GLM wrapper
#
# The standing lesson from stage 1: unit tests that never ran the real decode
# path shipped a crash. These run `mtp_stream_generate` itself against a
# GLM-SHAPED trunk (a VLM wrapper with no `.model`), which is what the
# capture-point walk and the head-provided draft cache exist for.


class _ToyGlmTrunk(_VLMWrapper):
    """A GLM-shaped trunk: logits are a deterministic function of the input
    token, so the greedy continuation is scriptable."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_widths: list[int] = []

    def make_cache(self):
        return [KVCache() for _ in range(3)]

    def __call__(self, tokens, cache=None):  # noqa: ANN001
        steps = int(tokens.shape[1])
        self.forward_widths.append(steps)
        # The capture point must see a real call, with the trunk's hidden
        # width, or `capture_input`'s getter raises.
        self.language_model.model.norm(mx.zeros((1, steps, D)))
        for c in cache or []:
            c.offset += steps
        rows = [[0.0] * VOCAB for _ in range(steps)]
        for s in range(steps):
            rows[s][_true_next(int(tokens[0, s].item()))] = 10.0
        return mx.array([rows])


def _true_next(t: int) -> int:
    return (t * 5 + 1) % VOCAB


class _ScriptedHead:
    """Drafts correctly, and owns its cache shape the way MTPHeadGlm5 does
    (CacheList(main-KV, indexer-KV)) — which is the branch in loop.py that a
    single registry attribute name cannot express."""

    def __init__(self) -> None:
        self.draft_cache_built = 0

    def make_draft_cache(self):
        self.draft_cache_built += 1
        return CacheList(KVCache(), KVCache())

    def advance(self, h, ids, cache):  # noqa: ANN001
        for c in cache:
            c.offset += int(ids.shape[1])

    def draft_logits(self, h, ids, cache):  # noqa: ANN001
        last = int(ids[0, -1].item())
        row = [0.0] * VOCAB
        row[_true_next(last)] = 10.0
        return mx.array([[list(row) for _ in range(int(ids.shape[1]))]])


class _ToyTokenizer:
    eos_token_id = None
    eos_token_ids: set[int] = set()

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


class _StubTokenizer(_ToyTokenizer):
    """Enough of a TokenizerWrapper for `mlx_generate`'s prompt handling."""

    has_thinking = False
    detokenizer = None

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        return [3, 4, 5, 6, 7]


def test_mlx_generate_drafts_end_to_end_on_a_glm_model(monkeypatch, tmp_path):
    """The REAL request path — `mlx_generate` -> `plan_mtp` -> `_load_head`
    -> `mtp_responses` — on a glm5_next model, with a real MTPHeadGlm5 built
    from a real sidecar.

    This exists because of the standing lesson from stage 1: the unit tests
    never exercised `mlx_generate`, and the gap shipped a crash. Everything
    here is real except the trunk weights, the tokenizer, and
    `glm5_shim.install` (stubbed so the test never monkeypatches a live
    mlx_vlm class out from under the rest of the suite).

    Draft QUALITY is meaningless here — the head is randomly initialised, so
    it will mostly be rejected. What is asserted is that the request takes
    the speculative path and produces the trunk's own greedy continuation,
    which is the correctness claim of exact verification.
    """
    from exo.shared.types.text_generation import (
        InputMessage,
        TextGenerationTaskParams,
    )
    from exo.worker.engines.mlx.generator.generate import mlx_generate
    from exo.worker.engines.mlx.mtp import speculative

    model = _ToyGlmTrunk()
    model.layers = [None, None, None]      # make_kv_cache asserts on this
    head, _ = _fresh_head(model)
    head.save(tmp_path / "mtp-head-q6.safetensors", bits=None)

    monkeypatch.setenv("EXO_MTP", "1")
    monkeypatch.setattr(speculative, "build_model_path", lambda _id: tmp_path)
    installs: list[int] = []
    monkeypatch.setattr(
        glm5_shim, "install", lambda *a, **k: installs.append(1) or True
    )

    task = TextGenerationTaskParams(
        model="TheDrainFlorist/GLM-5.3-Flash-VQ-2.7bpw",
        input=[InputMessage(role="user", content="hi")],
        max_output_tokens=6,
        temperature=0.0,
    )
    out = list(
        mlx_generate(
            model=model, tokenizer=_StubTokenizer(), task=task, prompt="hi",
            kv_prefix_cache=None, group=None,
        )
    )

    assert installs == [1], "the glm5 shim was not installed on the real path"
    assert len(out) == 6
    want, t = [], 7
    for _ in range(6):
        t = _true_next(t)
        want.append(t)
    assert [r.token for r in out] == want
    # Two tokens per step, so a 6-token request is 3 verify forwards plus the
    # prefill forwards; a step that never took the speculative path would
    # show up here as 6 single-token forwards.
    assert 2 in model.forward_widths


def test_the_loop_runs_against_a_vlm_wrapper_with_no_dot_model():
    """Before the walk was ported, this raised AttributeError on
    `model.model` inside `capture_input` — a crash, on the real decode path,
    that no gate test could see."""
    model = _ToyGlmTrunk()
    head = _ScriptedHead()
    prompt = mx.array([[3, 4, 5, 6, 7]])
    cache = model.make_cache()

    toks = [
        r.token
        for r in mtp_stream_generate(
            model, _ToyTokenizer(), prompt, head, family="glm5_next",
            max_tokens=8, temp=0.0, prompt_cache=cache,
        )
        if not r.tail
    ]

    assert head.draft_cache_built == 1, "the head's own cache shape was ignored"
    want, t = [], 7
    for _ in range(8):
        t = _true_next(t)
        want.append(t)
    assert toks == want
    # Every accepted step commits two tokens for one forward.
    assert [c.offset for c in cache] == [5 + 8] * 3
