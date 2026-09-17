"""A request the chat template refuses fails THAT request, not the runner.

2026-09-17: Qwen3.8's template raises TemplateError on reasoning_effort="high";
the runner re-raised, died, was rebuilt, and after five retries the instance was
deleted -- 2,864 times on one node. Now the task gets an ErrorChunk plus a
FinishedResponse and the runner keeps serving.
"""

from __future__ import annotations

import jinja2
import pytest

import exo.worker.runner.llm_inference.batch_generator as mlx_batch_generator
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.events import ChunkGenerated, RunnerStatusUpdated, TaskStatusUpdated
from exo.shared.types.tasks import TaskStatus, TextGeneration
from exo.shared.types.text_generation import InputMessage, InputMessageContent, TextGenerationTaskParams
from exo.shared.types.worker.runners import RunnerShutdown

from ...constants import COMMAND_1_ID, INSTANCE_1_ID, MODEL_A_ID
from .test_event_ordering import (
    CHAT_TASK,
    INIT_TASK,
    LOAD_TASK,
    SHUTDOWN_TASK,
    WARMUP_TASK,
    _run,
    patch_out_mlx,  # noqa: F401  (fixture)
)

BAD_ID = "bad-request-task"
BAD_TASK = TextGeneration(
    task_id=BAD_ID, command_id=COMMAND_1_ID, instance_id=INSTANCE_1_ID,
    task_params=TextGenerationTaskParams(
        model=MODEL_A_ID, stream=True, max_output_tokens=4, temperature=0.0,
        input=[InputMessage(role="user", content=InputMessageContent("REFUSE ME"))],
    ),
)


def _refusing_template(monkeypatch):
    def apply(_tok, params):
        if params.input and "REFUSE ME" in (params.input[0].content or ""):
            raise jinja2.exceptions.TemplateError("Unexpected reasoning effort high.")
        return "test prompt"
    monkeypatch.setattr(mlx_batch_generator, "apply_chat_template", apply)


def _check(events):
    errs = [e for e in events if isinstance(e, ChunkGenerated) and isinstance(e.chunk, ErrorChunk)]
    assert errs and "reasoning effort" in errs[0].chunk.error_message
    done = {e.task_id for e in events if isinstance(e, TaskStatusUpdated) and e.task_status == TaskStatus.Complete}
    assert BAD_ID in done, "refused task never finished -> runner would spin on it"
    assert CHAT_TASK.task_id in done, "the good task behind the refused one did not run"
    assert isinstance(events[-1], RunnerStatusUpdated) and isinstance(events[-1].runner_status, RunnerShutdown)


def test_batch_engine_refuses_the_request_and_keeps_serving(patch_out_mlx, monkeypatch):  # noqa: F811
    monkeypatch.delenv("EXO_NO_BATCH", raising=False)
    _refusing_template(monkeypatch)
    _check(_run([INIT_TASK, LOAD_TASK, WARMUP_TASK, BAD_TASK, CHAT_TASK, SHUTDOWN_TASK]))


def test_sequential_engine_refuses_the_request_and_keeps_serving(patch_out_mlx, monkeypatch):  # noqa: F811
    monkeypatch.setenv("EXO_NO_BATCH", "1")
    _refusing_template(monkeypatch)
    # The good task must actually generate on the sequential path: stub the
    # generator to one token then stop.
    from exo.shared.types.worker.runner_response import GenerationResponse

    def fake_generate(**_kw):
        yield GenerationResponse(text="hi", token=0, finish_reason="stop", usage=None)
    monkeypatch.setattr(mlx_batch_generator, "mlx_generate", fake_generate)
    _check(_run([INIT_TASK, LOAD_TASK, WARMUP_TASK, BAD_TASK, CHAT_TASK, SHUTDOWN_TASK]))
