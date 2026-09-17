import contextlib
import os
from collections.abc import Generator
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.cache import ChunkedKVCache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    SequentialGenerator,
)
from exo.worker.runner.llm_inference.tool_parsers import (
    make_llama4_parser,
    make_mlx_parser,
)

from .cache import KVPrefixCache
from .types import Model
from .utils_mlx import (
    initialize_mlx,
    load_mlx_items,
)
from .vision import VisionProcessor


def _mtp_available(model: Model, model_id: ModelId) -> bool:
    """True when this model COULD draft: family registered and the sidecar on
    disk. Environment-agnostic on purpose — the runner's engine-mode switch
    asks this to decide whether a sequential engine is worth building at all
    (sequential without drafting is the measured 4x decode loss)."""
    try:
        from exo.download.download_utils import build_model_path
        from exo.worker.engines.mlx.mtp.registry import resolve

        spec = resolve(model)
        return (build_model_path(model_id) / spec.sidecar_name).exists()
    except Exception:
        return False


def _mtp_would_engage(model: Model, model_id: ModelId) -> bool:
    """True when MTP drafting would actually activate for this model:
    EXO_MTP set, the family registered, and the sidecar on disk. Used to
    auto-select the sequential engine (the only path with an MTP loop) so
    the operator sets ONE knob, not two. Deliberately mirrors plan_mtp's
    per-request gates minus the topology ones — topology cannot flip
    between engine choice and the first request on this runner."""
    if os.environ.get("EXO_MTP") != "1":
        return False
    return _mtp_available(model, model_id)


def _has_unbatchable_cache(model: Model) -> bool:
    """mlx_lm's batch engine rejects some cache types ("ChunkedKVCache does not
    yet support batching with history" — llama4 being the current case). Probe
    the model's cache layout; such models run on the sequential engine. The
    prefix pool stays on — chunked-unsafe rollbacks are rejected per-hit by
    chunked_rollback_ok in cache.py.
    """
    make_cache = getattr(model, "make_cache", None)
    if make_cache is None:
        return False
    try:
        return any(isinstance(c, ChunkedKVCache) for c in make_cache())
    except Exception:
        return False


@dataclass
class MlxBuilder(Builder):
    # The runner may rebuild this builder's engine between tasks with a
    # different `engine_mode` (see exo/worker/runner/engine_mode.py). Image
    # builders do not advertise this and are never switched.
    supports_engine_modes = True

    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    inference_model: Model | None = None
    tokenizer: TokenizerWrapper | None = None
    group: mx.distributed.Group | None = None
    vision_processor: VisionProcessor | None = None

    def connect(self, bound_instance: BoundInstance) -> None:
        self.group = initialize_mlx(bound_instance)

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        (
            self.inference_model,
            self.tokenizer,
            self.vision_processor,
        ) = yield from load_mlx_items(bound_instance, self.group)

    def close(self) -> None:
        with contextlib.suppress(NameError, AttributeError):
            del self.inference_model
        with contextlib.suppress(NameError, AttributeError):
            del self.tokenizer
        with contextlib.suppress(NameError, AttributeError):
            del self.group

    def mtp_available(self) -> bool:
        assert self.inference_model
        return _mtp_available(self.inference_model, self.model_id)

    def _single_node(self) -> bool:
        return self.group is None or self.group.size() == 1

    def batch_drafts(self) -> bool:
        """Whether a batch engine built here drafts (mtp/batch_loop.py):
        the model can draft and the instance is single-node. Read by the
        runner's engine-mode switch."""
        return self._single_node() and self.mtp_available()

    def build(
        self,
        engine_mode: str | None = None,
        kv_prefix_cache: KVPrefixCache | None = None,
    ) -> Engine:
        """Build the engine over the already-loaded model.

        ``engine_mode`` None keeps the launch-time rule (EXO_NO_BATCH / cache
        type / EXO_MTP). "sequential" or "batch" is the runner's engine-mode
        switch choosing for THIS build; drafting is set to follow the engine
        (sequential drafts when the model can, batch never does), because the
        batch engine has no MTP path and a drafting request on it would be a
        request that never drafts anyway. ``kv_prefix_cache`` lets a rebuild
        keep the pooled prefixes of the engine it replaces — the whole point of
        switching without a reload is that nothing warm is thrown away.
        """
        assert self.inference_model
        assert self.tokenizer

        vision_processor = self.vision_processor

        tool_parser = None
        logger.info(
            f"model has_tool_calling={self.tokenizer.has_tool_calling} using tokens {self.tokenizer.tool_call_start}, {self.tokenizer.tool_call_end}"
        )
        if (
            self.tokenizer.tool_call_start
            and self.tokenizer.tool_call_end
            and self.tokenizer.tool_parser  # type: ignore
        ):
            tool_parser = make_mlx_parser(
                self.tokenizer.tool_call_start,
                self.tokenizer.tool_call_end,
                self.tokenizer.tool_parser,  # type: ignore
            )
        elif getattr(self.inference_model, "model_type", "") == "llama4":
            # mlx_lm's TokenizerWrapper carries no tool tokens for llama4;
            # without a parser its <|python_start|>{...}<|python_end|> calls
            # stream to the client as raw text.
            logger.info("using llama4 <|python_start|> tool parser")
            tool_parser = make_llama4_parser()

        if kv_prefix_cache is None:
            kv_prefix_cache = KVPrefixCache(self.group)

        device_rank = 0 if self.group is None else self.group.rank()
        unbatchable = _has_unbatchable_cache(self.inference_model)
        batch_head = None
        if engine_mode is None:
            will_draft = _mtp_would_engage(self.inference_model, self.model_id)
            sequential = bool(os.environ.get("EXO_NO_BATCH")) or unbatchable or will_draft
            reason = (
                "chunked KV cache; batch engine disabled"
                if unbatchable
                else "EXO_MTP drafting engages; batch engine has no MTP path"
                if will_draft
                else "batching disabled"
            )
        else:
            from exo.worker.engines.mlx.mtp.speculative import set_mtp_runtime

            if engine_mode not in ("sequential", "batch"):
                raise ValueError(f"unknown engine_mode {engine_mode!r}")
            # An unbatchable cache layout is a hard refusal, not a preference.
            sequential = engine_mode == "sequential" or unbatchable
            will_draft = False
            if self.mtp_available():
                if sequential:
                    will_draft = True
                elif self._single_node():
                    # The batch engine drafts too (mtp/batch_loop.py), on
                    # single-node instances: the head loads once and is
                    # shared with the sequential path's cache.
                    from exo.worker.engines.mlx.mtp.speculative import load_batch_head

                    batch_head = load_batch_head(self.inference_model, self.model_id)
                    will_draft = batch_head is not None
            set_mtp_runtime(will_draft)
            reason = (
                "chunked KV cache; batch engine disabled"
                if unbatchable
                else f"engine mode {engine_mode!r}"
                + (" (drafting)" if will_draft else " (no drafting)")
            )
        engine: Engine
        if sequential:
            logger.info(f"using SequentialGenerator ({reason})")
            engine = SequentialGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
            )
        else:
            logger.info(f"using BatchGenerator ({reason})" if engine_mode else "using BatchGenerator")
            engine = BatchGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
                mtp_head=batch_head,
            )
        # Read by the runner's engine-mode switch (runner/engine_mode.py).
        engine.engine_mode = "sequential" if sequential else "batch"  # type: ignore[attr-defined]
        return engine
