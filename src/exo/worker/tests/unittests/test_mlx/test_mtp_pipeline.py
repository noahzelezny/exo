"""Stage-1 MTP: topology gate, control broadcasts, and lockstep rollback.

The test that matters here is `test_two_shard_*`: two "shards" holding
DIFFERENT numbers of KV caches, running the real `mtp_stream_generate` in two
threads against a toy trunk, exchanging real control broadcasts through a
barrier. It is a pure-Python/small-tensor stand-in for a 2-node pipeline, and
it fails in the two ways the cluster would:

  - a rank that ran a different number of collectives blocks on the barrier,
    so a divergence is a TIMEOUT, not a hang forever;
  - a rank that trimmed the wrong number of positions ends with a cache
    offset that no longer equals `prompt + 2 * steps`, which is exactly the
    silent corruption this whole protocol exists to prevent. The all-reject
    case is the one that catches it: every step over-advances the cache by 2
    and must roll back by 2, so a fixed-count or missed trim shows up as
    linear drift within a handful of steps.

The toy trunk is deterministic (`_true_next`) so acceptance can be scripted
exactly, and the head is scripted rather than learned — this file is testing
the loop's control flow and cache arithmetic, not draft quality.
"""

from __future__ import annotations

import threading

import mlx.core as mx
import mlx.nn as nn
import pytest

from exo.worker.engines.mlx.mtp import registry
from exo.worker.engines.mlx.mtp.loop import mtp_stream_generate
from exo.worker.engines.mlx.mtp.pipeline import (
    LocalCoordinator,
    PipelineCoordinator,
    is_pipeline_model,
    make_coordinator,
)

VOCAB = 32
FAMILY = "mtp_stage1_toy"
BARRIER_TIMEOUT = 20.0


def _true_next(t: int) -> int:
    """The toy trunk's greedy continuation. Deterministic and cycle-free
    enough over the short runs here that no step accidentally hits EOS."""
    return (t * 7 + 1) % VOCAB


def _one_hot_rows(token: int, steps: int) -> mx.array:
    row = [0.0] * VOCAB
    row[token] = 10.0
    return mx.array([[list(row) for _ in range(steps)]])


# --------------------------------------------------------------- toy trunk


class _CapturePoint(nn.Module):
    """Stands in for the family's pre-lm_head module; capture.py wraps it."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


class ToyDraftCache:
    """The head's own cache. Never rolled back under align='committed'."""

    def __init__(self) -> None:
        self.offset = 0


class ToyAttnCache:
    """Enough of an attention cache for `caches.is_attention` and `restore`:
    an offset, a `keys` attribute, and a `trim` that moves the offset."""

    def __init__(self) -> None:
        self.offset = 0
        self.keys = None

    def trim(self, n: int) -> int:
        self.offset -= n
        return n


class ToyInner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = _CapturePoint()


class ToyModel(nn.Module):
    """One pipeline shard's view of a toy trunk.

    Every rank computes the SAME logits from the same token ids, which is the
    property `PipelineLastLayer`'s decode-time all_gather provides on the real
    thing. `n_caches` differs per shard on purpose: the rollback must be
    driven by each cache's own offset delta, not by a shared count.
    """

    def __init__(self, n_caches: int) -> None:
        super().__init__()
        self.model = ToyInner()
        self.n_caches = n_caches
        self.forward_widths: list[int] = []

    def make_cache(self) -> list[ToyAttnCache]:
        return [ToyAttnCache() for _ in range(self.n_caches)]

    def __call__(self, tokens: mx.array, cache=None) -> mx.array:
        steps = int(tokens.shape[1])
        self.forward_widths.append(steps)
        # The captured "hidden state" is just the token values; the toy head
        # ignores it, but capture.py must see a real call at the capture
        # point or the loop raises.
        self.model.norm(tokens.astype(mx.float32)[..., None])
        for c in cache or []:
            c.offset += steps
        rows = [
            [0.0] * VOCAB for _ in range(steps)
        ]
        for s in range(steps):
            rows[s][_true_next(int(tokens[0, s].item()))] = 10.0
        return mx.array([rows])


class ToyHead:
    """Drafts correctly or incorrectly on a script.

    `draft_logits` is handed the ids whose LAST entry is the token the trunk
    will be asked to continue, so a correct draft is `_true_next(last)` and a
    wrong one is anything else.
    """

    def __init__(self, accepts: list[bool]) -> None:
        self.accepts = accepts
        self.calls = 0

    def advance(self, h, ids, cache) -> None:  # noqa: ANN001
        cache.offset += int(ids.shape[1])

    def draft_logits(self, h, ids, cache):  # noqa: ANN001
        want_ok = self.accepts[min(self.calls, len(self.accepts) - 1)]
        self.calls += 1
        last = int(ids[0, -1].item())
        tok = _true_next(last) if want_ok else (_true_next(last) + 3) % VOCAB
        return _one_hot_rows(tok, int(ids.shape[1]))


class ToyTokenizer:
    eos_token_id = None
    eos_token_ids: set[int] = set()

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


@pytest.fixture(autouse=True)
def _toy_family():
    registry.register(
        registry.FamilySpec(
            name=FAMILY,
            head=f"{__name__}:ToyHead",
            capture="norm",
            draft_cache="ToyDraftCache",
            cache_semantics="reassign",
        ),
        replace=True,
    )
    yield
    registry.unregister(FAMILY)


# ------------------------------------------------------- the fake pipeline


class ThreadCoordinator:
    """`PipelineCoordinator` over a barrier instead of `mx.distributed`.

    Same contract, same last-rank-wins semantics. The barrier timeout is the
    point: if the two ranks ever reach a different NUMBER of broadcasts, this
    raises `BrokenBarrierError` instead of deadlocking the test suite.

    The payload crosses the thread boundary as Python ints, not as an
    `mx.array`. That is forced — MLX streams are thread-local, so handing a
    live array to another thread raises "no Stream(gpu, N) in current
    thread" — and it is also the more faithful model: a real broadcast
    serializes, and a rank that shares an array object with its peer is
    testing something the cluster will never do.
    """

    def __init__(self, rank: int, world: int, barrier, slot: list) -> None:
        self._rank = rank
        self._world = world
        self._barrier = barrier
        self._slot = slot

    @property
    def is_last(self) -> bool:
        return self._rank == self._world - 1

    def broadcast(self, values: mx.array) -> mx.array:
        if values.ndim != 1:
            raise ValueError("control broadcasts are 1-D")
        if self.is_last:
            self._slot[0] = [int(v) for v in values.tolist()]
        self._barrier.wait()
        payload = self._slot[0]
        # Second rendezvous so a fast rank cannot overwrite the slot before a
        # slow one has read it.
        self._barrier.wait()
        return mx.array(payload, dtype=mx.int32)


def _run_shard(model, head, coord, prompt, max_tokens, out: dict, key: str):
    cache = model.make_cache()
    tokens: list[int] = []
    try:
        for r in mtp_stream_generate(
            model,
            ToyTokenizer(),
            prompt,
            head,
            family=FAMILY,
            max_tokens=max_tokens,
            temp=0.0,
            prompt_cache=cache,
            coord=coord,
        ):
            if not r.tail:
                tokens.append(r.token)
        out[key] = {
            "tokens": tokens,
            "offsets": [c.offset for c in cache],
            "widths": list(model.forward_widths),
        }
    except BaseException as e:  # noqa: BLE001 - reported to the main thread
        out[key] = {"error": e}


def _two_shard_run(accepts: list[bool], max_tokens: int, prompt_len: int = 5):
    """Run a 2-shard pipeline in two threads; return both ranks' results."""
    prompt = mx.array([[3 + i for i in range(prompt_len)]])
    # Deliberately different shard sizes: 3 layers on rank 0, 5 on rank 1.
    shard0, shard1 = ToyModel(3), ToyModel(5)
    barrier = threading.Barrier(2, timeout=BARRIER_TIMEOUT)
    slot: list = [None]
    out: dict = {}
    threads = [
        threading.Thread(
            target=_run_shard,
            args=(
                shard0,
                None,  # a non-last rank never loads a head
                ThreadCoordinator(0, 2, barrier, slot),
                prompt,
                max_tokens,
                out,
                "r0",
            ),
        ),
        threading.Thread(
            target=_run_shard,
            args=(
                shard1,
                ToyHead(accepts),
                ThreadCoordinator(1, 2, barrier, slot),
                prompt,
                max_tokens,
                out,
                "r1",
            ),
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=BARRIER_TIMEOUT + 10)
        assert not t.is_alive(), "a shard never finished: the ranks diverged"
    for key in ("r0", "r1"):
        if "error" in out[key]:
            raise AssertionError(f"{key} raised: {out[key]['error']!r}") from out[
                key
            ]["error"]
    return out["r0"], out["r1"], prompt_len


# ------------------------------------------------------------------ tests


@pytest.mark.parametrize(
    ("name", "accepts"),
    [
        ("all_accepted", [True]),
        ("all_rejected", [False]),
        ("alternating", [True, False]),
        ("reject_then_accept", [False, True, True, False]),
    ],
)
def test_two_shard_rollback_is_lockstep(name: str, accepts: list[bool]):
    """Both shards emit the same tokens and land on the same cache offsets.

    The offset identity `prompt + 2 * steps` is the assertion with teeth: a
    rejection over-advances every cache by exactly 2 and must trim exactly 2
    back, so a wrong trim shows as drift that grows with the step count.
    """
    max_tokens = 12
    # Cycle the script out to a length the run cannot exhaust.
    r0, r1, prompt_len = _two_shard_run(
        [accepts[i % len(accepts)] for i in range(64)], max_tokens
    )

    assert r0["tokens"] == r1["tokens"], "shards emitted different text"
    assert len(r0["tokens"]) == max_tokens

    steps = max_tokens // 2
    expected = prompt_len + 2 * steps
    for rank, res in (("rank0", r0), ("rank1", r1)):
        assert res["offsets"] == [expected] * len(res["offsets"]), (
            f"{rank} cache offsets drifted: {res['offsets']} != {expected}"
        )


def test_two_shard_greedy_matches_the_trunk():
    """Speculation changes the schedule, never the text.

    Whatever the head drafts, the emitted sequence must be the toy trunk's
    own greedy continuation — that is the whole correctness claim of exact
    verification, and it has to survive being split across shards.
    """
    accepted, _, prompt_len = _two_shard_run([True] * 64, 10)
    rejected, _, _ = _two_shard_run([False] * 64, 10)
    assert accepted["tokens"] == rejected["tokens"]

    first = 3 + prompt_len - 1
    want, t = [], first
    for _ in range(10):
        t = _true_next(t)
        want.append(t)
    assert accepted["tokens"] == want


def test_rejection_costs_an_extra_forward_on_every_shard():
    """The replay forward is not a last-rank-only event.

    If only the drafting rank replayed, the pipeline would deadlock on the
    next send. Counting forwards on both shards is the cheap proof that the
    rollback path is symmetric.
    """
    all_ok, all_ok_r1, _ = _two_shard_run([True] * 64, 10)
    none_ok, none_ok_r1, _ = _two_shard_run([False] * 64, 10)

    assert len(none_ok["widths"]) > len(all_ok["widths"])
    assert len(none_ok["widths"]) == len(none_ok_r1["widths"])
    assert len(all_ok["widths"]) == len(all_ok_r1["widths"])
    # Every decode forward verifies a 2-token window — the T-wide verify.
    assert set(none_ok["widths"][-4:]) == {2}


def test_two_shard_stops_together_on_eos():
    """EOS inside a drafted pair ends the request on both ranks at once.

    Termination is derived from the emitted tokens, which the broadcasts make
    identical, so no separate stop message is needed — but if that ever
    stopped being true, one rank would keep calling collectives and the
    barrier here would time out.
    """
    prompt = mx.array([[3, 4, 5, 6, 7]])
    first = _true_next(7)
    eos = _true_next(_true_next(first))  # lands inside a drafted pair

    class EosTokenizer(ToyTokenizer):
        eos_token_ids = {eos}

    shard0, shard1 = ToyModel(3), ToyModel(5)
    barrier = threading.Barrier(2, timeout=BARRIER_TIMEOUT)
    slot: list = [None]
    out: dict = {}

    def run(model, head, rank, key):
        cache = model.make_cache()
        toks = []
        try:
            for r in mtp_stream_generate(
                model, EosTokenizer(), prompt, head, family=FAMILY,
                max_tokens=64, temp=0.0, prompt_cache=cache,
                coord=ThreadCoordinator(rank, 2, barrier, slot),
            ):
                if not r.tail:
                    toks.append(r.token)
            out[key] = toks
        except BaseException as e:  # noqa: BLE001
            out[key] = e

    threads = [
        threading.Thread(target=run, args=(shard0, None, 0, "r0")),
        threading.Thread(target=run, args=(shard1, ToyHead([True] * 64), 1, "r1")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=BARRIER_TIMEOUT + 10)
        assert not t.is_alive()

    assert not isinstance(out["r0"], BaseException), out["r0"]
    assert out["r0"] == out["r1"]
    assert out["r0"][-1] == eos
    assert eos not in out["r0"][:-1], "generation continued past EOS"


def _single_node_run(accepts: list[bool], max_tokens: int, prompt_len: int = 5):
    """Stage 0: the same loop, default coordinator, one rank."""
    prompt = mx.array([[3 + i for i in range(prompt_len)]])
    model = ToyModel(8)
    cache = model.make_cache()
    toks = []
    for r in mtp_stream_generate(
        model, ToyTokenizer(), prompt, ToyHead(accepts), family=FAMILY,
        max_tokens=max_tokens, temp=0.0, prompt_cache=cache,
    ):
        if not r.tail:
            toks.append(r.token)
    return toks, [c.offset for c in cache]


def test_stage_0_default_path_is_unchanged():
    """No coordinator argument means LocalCoordinator means stage 0.

    Guards the restructuring itself: the seam that made the loop shardable
    must be a no-op when nothing shards it.
    """
    toks, offsets = _single_node_run([True] * 64, 12)
    assert len(toks) == 12
    assert offsets == [5 + 12] * 8


@pytest.mark.parametrize("accepts", [[True], [False], [True, False]])
def test_pipeline_output_matches_single_node(accepts: list[bool]):
    """The cluster smoke's claim, in miniature.

    A 2-shard pipeline and a single node running the same greedy prompt must
    emit the same tokens. This is the property the orchestrator will check
    against a real Flash pipeline; having it hold on a toy trunk first means
    a failure there is about the model or the transport, not about the
    protocol in loop.py.
    """
    script = [accepts[i % len(accepts)] for i in range(64)]
    single, _ = _single_node_run(script, 10)
    shard0, shard1, _ = _two_shard_run(script, 10)
    assert single == shard0["tokens"] == shard1["tokens"]


# ------------------------------------------------------- coordinator units


def test_local_coordinator_is_the_identity():
    c = LocalCoordinator()
    assert c.is_last
    v = mx.array([7, 9], dtype=mx.int32)
    assert c.broadcast(v) is v


class _FakeGroup:
    def __init__(self, rank: int, size: int) -> None:
        self._r, self._s = rank, size

    def rank(self) -> int:
        return self._r

    def size(self) -> int:
        return self._s


def test_pipeline_coordinator_reports_the_last_rank():
    assert PipelineCoordinator(_FakeGroup(1, 2)).is_last
    assert not PipelineCoordinator(_FakeGroup(0, 2)).is_last
    assert PipelineCoordinator(_FakeGroup(2, 3)).is_last


def test_pipeline_coordinator_refuses_wide_payloads():
    """Control broadcasts carry ids and flags; hidden states go through the
    graph's send/recv. Sending one here would be a silent bandwidth cliff."""
    c = PipelineCoordinator(_FakeGroup(0, 2))
    with pytest.raises(ValueError, match="1-D"):
        c.broadcast(mx.zeros((2, 4), dtype=mx.int32))


def test_make_coordinator_collapses_single_rank():
    assert isinstance(make_coordinator(None), LocalCoordinator)
    assert isinstance(make_coordinator(_FakeGroup(0, 1)), LocalCoordinator)
    assert isinstance(make_coordinator(_FakeGroup(0, 2)), PipelineCoordinator)


# --------------------------------------------------------- the gate itself


def _pipeline_wrapped_model():
    from exo.worker.engines.mlx.auto_parallel import PipelineFirstLayer

    class _Layer(nn.Module):
        def __call__(self, x, *a, **k):  # noqa: ANN001
            return x

    model = ToyModel(2)
    model.layers = [PipelineFirstLayer(_Layer(), 1, group=_FakeGroup(1, 2))]
    return model


def _plain_model():
    class _Layer(nn.Module):
        def __call__(self, x, *a, **k):  # noqa: ANN001
            return x

    model = ToyModel(2)
    model.layers = [_Layer(), _Layer()]
    return model


def test_is_pipeline_model_asks_the_object():
    assert is_pipeline_model(_pipeline_wrapped_model())
    assert not is_pipeline_model(_plain_model())
    assert not is_pipeline_model(ToyModel(2))  # no .layers at all


@pytest.fixture()
def _mtp_on(monkeypatch):
    monkeypatch.setenv("EXO_MTP", "1")


_SENTINEL_HEAD = object()


def _patch_head_load(monkeypatch, head=_SENTINEL_HEAD):
    from exo.worker.engines.mlx.mtp import speculative

    monkeypatch.setattr(speculative, "_load_head", lambda *a, **k: head)
    return speculative


@pytest.mark.usefixtures("_mtp_on")
def test_gate_single_node_takes_stage_0(monkeypatch):
    spec = _patch_head_load(monkeypatch)
    plan = spec.plan_mtp(_plain_model(), "m", None, has_vision=False)
    assert plan is not None
    assert plan.stage == 0
    assert isinstance(plan.coord, LocalCoordinator)
    assert plan.drafts


@pytest.mark.usefixtures("_mtp_on")
def test_gate_pipeline_multi_node_takes_stage_1(monkeypatch):
    spec = _patch_head_load(monkeypatch)
    # `_FakeGroup` is not an mx.distributed.Group, so the real all_gather in
    # the agreement broadcast cannot run here. The agreement itself is
    # exercised by the two-shard tests above; what this test is pinning down
    # is the topology DECISION.
    monkeypatch.setattr(PipelineCoordinator, "broadcast", lambda self, v: v)
    plan = spec.plan_mtp(
        _pipeline_wrapped_model(), "m", _FakeGroup(1, 2), has_vision=False
    )
    assert plan is not None
    assert plan.stage == 1
    assert plan.coord.is_last
    assert plan.drafts


@pytest.mark.usefixtures("_mtp_on")
def test_gate_tensor_sharding_stays_refused(monkeypatch):
    """No rank in a tensor-sharded instance holds a whole final hidden state,
    so there is nothing for the head to draft from. Refused, not degraded."""
    spec = _patch_head_load(monkeypatch)
    assert (
        spec.plan_mtp(_plain_model(), "m", _FakeGroup(0, 2), has_vision=False)
        is None
    )


@pytest.mark.usefixtures("_mtp_on")
def test_gate_refuses_vision_and_missing_model_id(monkeypatch):
    spec = _patch_head_load(monkeypatch)
    assert spec.plan_mtp(_plain_model(), "m", None, has_vision=True) is None
    assert spec.plan_mtp(_plain_model(), None, None, has_vision=False) is None


def test_gate_is_off_without_the_env_var(monkeypatch):
    monkeypatch.delenv("EXO_MTP", raising=False)
    spec = _patch_head_load(monkeypatch)
    assert spec.plan_mtp(_plain_model(), "m", None, has_vision=False) is None


@pytest.mark.usefixtures("_mtp_on")
def test_gate_falls_back_when_the_head_will_not_load(monkeypatch):
    spec = _patch_head_load(monkeypatch, head=None)
    assert spec.plan_mtp(_plain_model(), "m", None, has_vision=False) is None
