"""Engine-mode switch: sequential+MTP for one voice, the batch engine for many.

The two mlx engines want opposite things. `SequentialGenerator` is the only
path with an MTP loop, so a lone chat gets speculative drafting there; but it
serves ONE request at a time, so a six-worker swarm behind it is a queue.
`BatchGenerator` serves them concurrently and cannot draft. Until now the
choice was made once, at engine build, from EXO_MTP — and changing it meant
restarting exo and reloading ~100 GB of weights.

Both engines are built over the same loaded model, tokenizer and pooled
prefix cache, so the runner can replace one with the other at a task
boundary (nothing mid-generation) in the time of a warmup, keeping every
warm byte. This module holds the DECISION; the runner holds the swap.

Control plane, stage 0: a per-node mode file — `~/.exo/engine-mode`
(override with EXO_ENGINE_MODE_FILE) — holding one word:

    sequential   one request at a time, drafting when the model can
    batch        concurrent serving; drafting too on a single-node instance
                 (mtp/batch_loop.py), plain batching on a sharded one
    auto         where the batch engine drafts: batch, always. Otherwise batch
                 as soon as another request is WAITING at a boundary and
                 sequential when a request starts alone (the contextual mode:
                 a sub-agent or swarm flips it, the next lone chat flips back)

Absent or unreadable file = the launch-time rule, exactly as before. The file
is read at each task boundary, so an operator (or Scout) writes one word and
the next boundary applies it — no restart, no reload. Single-node instances
only: a switch is a local decision and the ranks of a sharded instance have
no shared file to agree on, so a group of size >1 is never switched.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

EngineMode = Literal["sequential", "batch"]
MODES = ("sequential", "batch", "auto")
# `auto` picks the batch engine when at least this many OTHER requests are
# waiting at a boundary. 1, measured 2026-09-17 on Qwen3.8-Flash-Next-VQ-4.4bpw
# (M4, 500-token generations): sequential+MTP alone 26.3 tok/s; batch alone
# 19.6; batch x2 16.6 each (33.2 aggregate, two done in 30s vs 38s serial);
# batch x3 15.0 each (44.9 aggregate). A waiting sub-agent therefore costs the
# chat ~1/3 of its decode while both run, but starts at once instead of
# queueing ~20s -- the trade delegation exists for. Each switch is ~3-4s.
AUTO_BATCH_THRESHOLD = 1


def mode_file() -> Path:
    return Path(os.environ.get("EXO_ENGINE_MODE_FILE") or Path.home() / ".exo" / "engine-mode")


def read_mode(path: Path | None = None) -> str | None:
    """The requested mode, or None when no (valid) file is present. Never raises:
    a garbage file logs nothing and behaves like no file, because a typo must
    not take a serving node off its launch-time rule."""
    try:
        text = (path or mode_file()).read_text(encoding="utf-8").strip().lower()
    except OSError:
        return None
    return text if text in MODES else None


def resolve_target(
    mode: str | None,
    *,
    waiting: int,
    mtp_available: bool,
    current: EngineMode,
    batch_drafts: bool = False,
    current_drafts: bool = True,
) -> EngineMode | None:
    """The engine the runner should be on for the next task(s), or None to stay.

    `waiting` counts OTHER requests queued at this boundary: 0 at the idle
    boundary where one request is about to start alone, len(queue) at a
    finish boundary where the rest are still waiting. A sequential
    engine for a model that cannot draft is the measured 4x decode loss with
    nothing bought, so `auto` never picks it for such a model and an explicit
    "sequential" is honored only when drafting is available.

    `current_drafts`: whether the engine the runner is on drafts now — the
    launch rule builds a plain batch engine, and the first boundary must
    rebuild it with the head when `batch_drafts` says one is available.

    `batch_drafts`: the batch engine itself drafts (mtp/batch_loop.py, single-
    node instances). Then `auto` has nothing to flip for: a lone request on
    the drafting batch engine decodes as the sequential loop would (measured
    2026-09-17, Qwen3.8-Flash-Next-VQ-2.1bpw on the M3: 23.1 vs 22 tok/s,
    identical tokens) and a waiter simply joins the batch, with no ~3s rebuild
    at either boundary. An explicit "sequential" is still honored.
    """
    if mode is None:
        return None
    if mode == "auto":
        want: EngineMode = (
            "batch"
            if batch_drafts or waiting >= AUTO_BATCH_THRESHOLD or not mtp_available
            else "sequential"
        )
    elif mode == "sequential":
        want = "sequential" if mtp_available else "batch"
    else:
        want = "batch"
    if want == current:
        # A batch engine built by the launch rule (EXO_MTP unset) has no head;
        # if the batch engine here CAN draft, rebuild it once so it does.
        if want == "batch" and batch_drafts and not current_drafts:
            return "batch"
        return None
    return want
