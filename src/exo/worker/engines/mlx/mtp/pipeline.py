"""Stage-1 control plane: coordinating a speculative step across pipeline shards.

The finding that shapes this file: exo's pipeline parallelism is SPMD *inside
the MLX graph*, not an orchestrator driving workers. Every rank runs the same
Python decode loop; `PipelineFirstLayer` swaps its input for a
`mx.distributed.recv_like` from rank r-1, and `PipelineLastLayer` sends its
output to r+1 (auto_parallel.py). There is no per-step control channel to
extend, because there is no per-step control channel at all — ranks stay in
lockstep by *executing the same code*.

That is the whole reason stage 1 is small. A speculative step needs no new
collective for the verify forward (requirement 2 falls out for free: a T-wide
trunk forward is just `model(two_tokens)`, the same shape prefill already
sends) and no rollback *message* in the usual sense. What it needs is for
every rank to reach the same accept/reject verdict, and then each rank trims
its own KV caches by the delta it measured — `caches.restore` already trims by
the offset delta rather than a fixed count, so a shard holding 20 layers and a
shard holding 4 both roll back correctly from the same verdict.

Two things break the "same code, same result" symmetry, and this file exists
to repair exactly those two:

1. **The draft head lives on the last rank only.** During prefill,
   `PipelineLastLayer` skips its all_gather (`is_prefill=True`), so only the
   last rank's captured activation is the true final hidden state — the other
   ranks capture their own shard's intermediate output, which is garbage to
   draft from. Seeding the head anywhere else would need an all_gather of
   `[1, chunk, H]` per prefill chunk (tens of MB per chunk); broadcasting one
   int per step instead is four orders of magnitude cheaper.

2. **Sampling draws randomness.** The last rank samples the draft token, so
   its `mx.random` stream advances past the others' and every subsequent
   independent sample would diverge. Once the last rank samples *anything*
   for a step, it must sample *everything* for that step and broadcast.

So: two tiny broadcasts per step, both from the last rank.

    B1  [d2]                 before the verify forward — rank 0 must embed
                             the drafted token, so it needs the real id.
    B2  [ok, t2, t_next]     after the verify forward — the verdict that
                             drives every rank's rollback, plus the two
                             tokens that actually commit.

Ordering is fixed and unconditional: B1 and B2 happen on every step, on every
rank, in that order, whether or not the draft is accepted. A verdict-dependent
number of collectives would deadlock the moment two ranks disagreed, which is
precisely the failure this is guarding against.

Failure behavior is inherited, not invented. Both broadcasts are
`mx.distributed.all_gather` over the same group the hidden-state send/recv
already uses, so a dead or wedged rank hangs or raises here exactly as it
would in `PipelineFirstLayer.recv_like`. Stage 1 adds no new failure mode and
no new recovery path; the runner's existing instance-level supervision is
still the thing that notices.

`LocalCoordinator` is the single-node identity of all of this, which is what
lets stage 0 and stage 1 be *one loop* rather than two. Stage 0 passes a
coordinator whose broadcasts are the identity function and whose `is_last` is
True, and the loop reduces line-for-line to what it did before.
"""

from __future__ import annotations

from typing import Protocol

import mlx.core as mx

__all__ = [
    "Coordinator",
    "LocalCoordinator",
    "PipelineCoordinator",
    "is_pipeline_model",
    "make_coordinator",
]


def is_pipeline_model(model) -> bool:
    """Is this model instance pipeline-sharded?

    Asks the object, not the placement metadata. `pipeline_auto_parallel`
    wraps the shard's first and last decoder layer; a tensor-sharded or
    single-node model has neither wrapper. That makes this a direct test of
    the property stage 1 actually depends on — the all_gather in
    `PipelineLastLayer` that puts the final hidden state on the last rank —
    rather than a claim about how the instance was *meant* to be built.
    """
    # Imported here: auto_parallel pulls in a large slice of mlx_lm's model
    # zoo, and the mtp package is imported at module scope by generate.py.
    from exo.worker.engines.mlx.auto_parallel import (
        PipelineFirstLayer,
        PipelineLastLayer,
    )

    layers = getattr(model, "layers", None)
    if layers is None:
        return False
    return any(
        isinstance(layer, (PipelineFirstLayer, PipelineLastLayer))
        for layer in layers
    )


class Coordinator(Protocol):
    """What the speculative loop needs from its execution topology."""

    @property
    def is_last(self) -> bool:
        """Does this rank hold the true final hidden state (and the head)?"""
        ...

    def broadcast(self, values: mx.array) -> mx.array:
        """Return the LAST rank's `values` on every rank.

        `values` must have identical shape and dtype on every rank; non-last
        ranks supply a placeholder that is discarded.
        """
        ...


class LocalCoordinator:
    """Single node: every broadcast is the identity, this rank is the last."""

    __slots__ = ()

    @property
    def is_last(self) -> bool:
        return True

    def broadcast(self, values: mx.array) -> mx.array:
        return values

    def __repr__(self) -> str:
        return "LocalCoordinator()"


class PipelineCoordinator:
    """Pipeline-sharded: broadcast from the last rank over the model's group.

    The broadcast idiom is `all_gather(...)[-n:]` — take the last rank's block
    out of the concatenation. That is not a clever trick invented here; it is
    the same expression `PipelineLastLayer` already uses to put the final
    hidden state on every rank during decode, so it is proven on this
    transport. Reusing it keeps stage 1 from introducing a second, differently
    -behaved notion of "broadcast" alongside the one in the graph.

    The collective runs on the CPU stream, matching `mx_barrier` and `mx_any`
    in utils_mlx: these are a handful of ints, and the GPU stream carries a
    watchdog timeout that a control message has no business sitting under.
    """

    __slots__ = ("_group", "_rank", "_size")

    def __init__(self, group: mx.distributed.Group):
        self._group = group
        self._rank = group.rank()
        self._size = group.size()

    @property
    def group(self) -> mx.distributed.Group:
        return self._group

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._size

    @property
    def is_last(self) -> bool:
        return self._rank == self._size - 1

    def broadcast(self, values: mx.array) -> mx.array:
        if self._size == 1:
            return values
        if values.ndim != 1:
            raise ValueError(
                f"control broadcasts are 1-D vectors of ids/flags; got shape "
                f"{values.shape}. Anything wider is a hidden state, and hidden "
                f"states travel through the graph's send/recv, not here."
            )
        n = values.shape[0]
        gathered = mx.distributed.all_gather(
            values,
            group=self._group,
            stream=mx.default_stream(mx.Device(mx.cpu)),
        )
        out = gathered[-n:]
        mx.eval(out)
        return out

    def __repr__(self) -> str:
        return (
            f"PipelineCoordinator(rank={self._rank}/{self._size}, "
            f"is_last={self.is_last})"
        )


def make_coordinator(group: mx.distributed.Group | None) -> Coordinator:
    """The coordinator for this instance's topology."""
    if group is None or group.size() <= 1:
        return LocalCoordinator()
    return PipelineCoordinator(group)
