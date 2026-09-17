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
    batch        concurrent serving, no drafting
    auto         batch when ≥2 requests are waiting, sequential otherwise
                 (the contextual mode: a swarm flips it, a chat flips it back)

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
# ≥ this many requests waiting at a boundary and `auto` picks the batch engine.
AUTO_BATCH_THRESHOLD = 2


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
    pending: int,
    mtp_available: bool,
    current: EngineMode,
) -> EngineMode | None:
    """The engine the runner should be on for the next task(s), or None to stay.

    `pending` counts requests that are waiting (none started). A sequential
    engine for a model that cannot draft is the measured 4x decode loss with
    nothing bought, so `auto` never picks it for such a model and an explicit
    "sequential" is honored only when drafting is available.
    """
    if mode is None:
        return None
    if mode == "auto":
        want: EngineMode = (
            "batch"
            if pending >= AUTO_BATCH_THRESHOLD or not mtp_available
            else "sequential"
        )
    elif mode == "sequential":
        want = "sequential" if mtp_available else "batch"
    else:
        want = "batch"
    return None if want == current else want
