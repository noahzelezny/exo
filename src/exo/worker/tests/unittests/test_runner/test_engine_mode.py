"""The engine-mode switch: sequential+MTP for one voice, batch for many, flipped
at a task boundary without a reload (exo/worker/runner/engine_mode.py)."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field

import pytest

from exo.shared.types.tasks import LoadModel, StartWarmup, TaskId, TaskStatus, TextGeneration
from exo.shared.types.events import TaskStatusUpdated
from exo.shared.types.worker.runner_response import FinishedResponse
from exo.utils.channels import mp_channel
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner import engine_mode as em
from exo.worker.runner.runner import Runner

from ...constants import (
    INSTANCE_1_ID,
    LOAD_TASK_ID,
    MODEL_A_ID,
    NODE_A,
    RUNNER_1_ID,
    WARMUP_TASK_ID,
)
from ..conftest import get_bound_mlx_ring_instance
from .test_event_ordering import CHAT_PARAMS, EventCollector, MockLoadOutput, nothin


# ----------------------------------------------------------------- decision

@pytest.mark.parametrize(
    "mode,waiting,mtp,current,want",
    [
        (None, 5, True, "sequential", None),            # no file: launch rule stands
        ("auto", 0, True, "batch", "sequential"),       # a lone chat drafts
        ("auto", 0, True, "sequential", None),
        ("auto", 1, True, "sequential", "batch"),       # anyone waiting: batch
        ("auto", 5, True, "batch", None),
        ("auto", 0, False, "sequential", "batch"),      # can't draft: never sit sequential
        ("sequential", 9, True, "batch", "sequential"),
        ("sequential", 0, False, "batch", None),        # honored only when drafting exists
        ("batch", 0, True, "sequential", "batch"),
        ("batch", 0, True, "batch", None),
    ],
)
def test_resolve_target(mode, waiting, mtp, current, want):
    assert em.resolve_target(mode, waiting=waiting, mtp_available=mtp, current=current) == want


def test_read_mode_file(tmp_path, monkeypatch):
    f = tmp_path / "engine-mode"
    monkeypatch.setenv("EXO_ENGINE_MODE_FILE", str(f))
    assert em.read_mode() is None                      # absent
    f.write_text(" Auto \n")
    assert em.read_mode() == "auto"                    # case/whitespace tolerant
    f.write_text("turbo")
    assert em.read_mode() is None                      # a typo is not a mode


# ----------------------------------------------------------------- runner swap

@dataclass(eq=False)
class FakeEngine(Engine):
    """`sequential`: starts one task per step and finishes it on the NEXT step,
    so the runner observes a real boundary (finished + others waiting).
    `batch`: finishes everything pending in one step."""
    engine_mode: str
    kv_prefix_cache: object = field(default_factory=object)
    _cancelled_tasks: set[TaskId] = field(default_factory=set)
    pending: list[TextGeneration] = field(default_factory=list)
    started: TextGeneration | None = None
    submitted: list[TaskId] = field(default_factory=list)
    closed: bool = False
    warmed: bool = False

    def warmup(self) -> None:
        self.warmed = True

    def submit(self, task) -> None:
        self.submitted.append(task.task_id)
        self.pending.append(task)

    def in_flight(self) -> bool:
        return self.started is not None

    def step(self):
        done = FinishedResponse()
        if self.engine_mode == "batch":
            out = [(t.task_id, done) for t in self.pending]
            self.pending.clear()
            return out
        if self.started is not None:
            t, self.started = self.started, None
            return [(t.task_id, done)]
        if self.pending:
            self.started = self.pending.pop(0)
        return []

    def close(self) -> None:
        self.closed = True

    def serve_prefill(self, request, wfile) -> None: ...


class FakeBuilder(Builder):
    supports_engine_modes = True

    def __init__(self, *, mtp: bool = True, world: int = 1, first: str = "sequential"):
        self.mtp = mtp
        self.first = first
        self.group = None if world == 1 else type("G", (), {"size": lambda s: world})()
        self.builds: list[dict] = []
        self.engines: list[FakeEngine] = []

    def connect(self, bound_instance) -> None: ...

    def load(self, bound_instance) -> Generator:
        yield MockLoadOutput(1, 1)

    def mtp_available(self) -> bool:
        return self.mtp

    def build(self, engine_mode=None, kv_prefix_cache=None) -> Engine:
        self.builds.append({"engine_mode": engine_mode, "kv_prefix_cache": kv_prefix_cache})
        e = FakeEngine(engine_mode=engine_mode or self.first)
        if kv_prefix_cache is not None:
            e.kv_prefix_cache = kv_prefix_cache
        self.engines.append(e)
        return e

    def close(self) -> None: ...


def _chat(i: int) -> TextGeneration:
    return TextGeneration(
        task_id=TaskId(f"t{i}"), command_id=TaskId(f"c{i}"),
        task_params=CHAT_PARAMS, instance_id=INSTANCE_1_ID,
    )


def _ready_runner(monkeypatch, builder: FakeBuilder) -> tuple[Runner, EventCollector]:
    bound = get_bound_mlx_ring_instance(
        instance_id=INSTANCE_1_ID, model_id=MODEL_A_ID, runner_id=RUNNER_1_ID, node_id=NODE_A,
    )
    _sender, receiver = mp_channel()
    receiver.close = nothin
    receiver.join = nothin
    monkeypatch.setattr(Runner, "_start_prefill_server", lambda _self: None)
    events = EventCollector()
    r = Runner(bound, builder, events, receiver)  # pyright: ignore[reportArgumentType]
    r.handle_first_task(LoadModel(task_id=LOAD_TASK_ID, instance_id=INSTANCE_1_ID))
    r.handle_first_task(StartWarmup(task_id=WARMUP_TASK_ID, instance_id=INSTANCE_1_ID))
    return r, events


def _completed(events) -> set[TaskId]:
    return {
        e.task_id for e in events.events
        if isinstance(e, TaskStatusUpdated) and e.task_status == TaskStatus.Complete
    }


def test_auto_moves_a_waiting_fanout_onto_the_batch_engine(monkeypatch, tmp_path):
    f = tmp_path / "engine-mode"; f.write_text("auto")
    monkeypatch.setenv("EXO_ENGINE_MODE_FILE", str(f))
    b = FakeBuilder()
    r, events = _ready_runner(monkeypatch, b)
    assert r._engine_mode == "sequential"
    seq = b.engines[0]

    for i in (2, 3, 4):
        r._work_queue.put(_chat(i))
    r.handle_generation_tasks(starting_task=_chat(1))

    # Switched exactly once, to batch, at the first finish with someone
    # waiting (t2), over the SAME prefix cache; t2 was re-submitted there and
    # t3/t4 arrived on the batch engine directly. t1 ran sequential and was
    # never re-submitted.
    assert [x["engine_mode"] for x in b.builds] == [None, "batch"]
    assert b.builds[1]["kv_prefix_cache"] is seq.kv_prefix_cache
    batch = b.engines[1]
    assert batch.warmed and seq.closed
    assert set(batch.submitted) == {TaskId("t2"), TaskId("t3"), TaskId("t4")}
    assert seq.submitted == [TaskId("t1"), TaskId("t2")]
    assert r._engine_mode == "batch"
    assert {TaskId(f"t{i}") for i in (1, 2, 3, 4)} <= _completed(events)


def test_auto_flips_back_to_sequential_for_a_lone_request(monkeypatch, tmp_path):
    f = tmp_path / "engine-mode"; f.write_text("auto")
    monkeypatch.setenv("EXO_ENGINE_MODE_FILE", str(f))
    b = FakeBuilder(first="batch")
    r, events = _ready_runner(monkeypatch, b)
    assert r._engine_mode == "batch"
    r.handle_first_task(_chat(1))          # the idle boundary in main()
    assert [x["engine_mode"] for x in b.builds] == [None, "sequential"]
    assert r._engine_mode == "sequential"
    assert TaskId("t1") in _completed(events)


def test_no_mode_file_means_no_switch(monkeypatch, tmp_path):
    monkeypatch.setenv("EXO_ENGINE_MODE_FILE", str(tmp_path / "missing"))
    b = FakeBuilder()
    r, _ = _ready_runner(monkeypatch, b)
    for i in (2, 3, 4):
        r._work_queue.put(_chat(i))
    r.handle_generation_tasks(starting_task=_chat(1))
    assert [x["engine_mode"] for x in b.builds] == [None]


def test_sharded_instance_is_never_switched(monkeypatch, tmp_path):
    f = tmp_path / "engine-mode"; f.write_text("batch")
    monkeypatch.setenv("EXO_ENGINE_MODE_FILE", str(f))
    b = FakeBuilder(world=2)
    r, _ = _ready_runner(monkeypatch, b)
    assert r._maybe_switch_engine(waiting=0) is False
    assert len(b.builds) == 1


def test_switch_refuses_while_a_task_is_mid_generation(monkeypatch, tmp_path):
    f = tmp_path / "engine-mode"; f.write_text("batch")
    monkeypatch.setenv("EXO_ENGINE_MODE_FILE", str(f))
    b = FakeBuilder()
    r, _ = _ready_runner(monkeypatch, b)
    b.engines[0].started = _chat(9)
    assert r._maybe_switch_engine(waiting=3) is False
    assert len(b.builds) == 1


def test_builder_without_modes_is_left_alone(monkeypatch, tmp_path):
    f = tmp_path / "engine-mode"; f.write_text("batch")
    monkeypatch.setenv("EXO_ENGINE_MODE_FILE", str(f))
    b = FakeBuilder()
    b.supports_engine_modes = False
    r, _ = _ready_runner(monkeypatch, b)
    assert r._maybe_switch_engine(waiting=5) is False
