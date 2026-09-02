"""Snapshot and rollback for a speculative decode step.

Two cache kinds appear in one `model.make_cache()` list and they roll back
completely differently:

  attention caches   expose keys/values and a `trim(n)`. They MUST be trimmed
                     by the offset DELTA, not by a fixed count. Trimming a
                     hardcoded 1 leaves a stale key behind while the recurrent
                     caches roll back 2, and the two streams then drift
                     silently — correct-looking text that is not what the
                     trunk would have produced. This bug was hit once already.

  recurrent caches   expose a `cache` list of state arrays. Rollback is
                     whatever the snapshot held.

For the recurrent kind there is a real hazard behind an implementation
accident. qwen4_exp REASSIGNS its slots (`cache[0] = ...`) and mlx arrays are
immutable, so keeping the old references is a free, correct snapshot. An
architecture that instead writes in place (`cache[0][..., i] = k`, which mlx
does support) would corrupt the snapshot through the very reference we saved.
So the copying policy is a per-family field, defaulting to "copy", and
`check_snapshot_semantics` is the measurement that earns a family the cheap
"reassign" path.
"""
from __future__ import annotations

import mlx.core as mx


def is_attention(c) -> bool:
    return hasattr(c, "keys") and hasattr(c, "trim")


def snapshot(caches, *, copy: bool = True) -> list:
    snaps = []
    for c in caches:
        if is_attention(c):
            snaps.append(("attn", c.offset, None))
        elif hasattr(c, "cache"):
            state = list(c.cache)
            if copy:
                state = [mx.array(x) if isinstance(x, mx.array) else x
                         for x in state]
            snaps.append(("state", getattr(c, "offset", None), state))
        else:
            raise TypeError(
                f"cache {type(c).__name__} is neither an attention cache "
                f"(keys/trim) nor a state cache (.cache); speculative "
                f"rollback cannot be proven correct for it")
    return snaps


def restore(caches, snaps) -> None:
    """Back to exactly where the snapshot was taken."""
    for c, s in zip(caches, snaps):
        kind, offset, state = s
        if kind == "attn":
            n = c.offset - offset
            if n > 0:
                c.trim(n)
            elif n < 0:
                raise RuntimeError(
                    f"attention cache went BACKWARDS since the snapshot "
                    f"({c.offset} < {offset}); rollback would corrupt it")
        else:
            c.cache = list(state)
            if offset is not None:
                c.offset = offset


def check_snapshot_semantics(caches, advance) -> bool:
    """Does `copy=False` snapshotting actually hold for these caches?

    Snapshot without copying, deep-copy the same arrays separately, run
    `advance()` (one forward through the model), and check the snapshot's
    arrays are still what they were. True means the family may use
    cache_semantics="reassign"; False means it writes state in place and must
    copy. Returns True for an all-attention cache list (nothing to alias).

    This is the gate that lets a new family claim the cheap path, and it fails
    on an in-place cache by construction — see tests/test_mtp_caches.py, where
    it is run against both a reassigning and a mutating cache.
    """
    snaps = snapshot(caches, copy=False)
    witness = [[mx.array(x) if isinstance(x, mx.array) else x for x in s[2]]
               for s in snaps if s[0] != "attn"]
    for w in witness:
        mx.eval(*[x for x in w if isinstance(x, mx.array)])
    advance()
    held = [s[2] for s in snaps if s[0] != "attn"]
    for kept, ref in zip(held, witness):
        for a, b in zip(kept, ref):
            if not isinstance(a, mx.array) or not isinstance(b, mx.array):
                continue
            if a.shape != b.shape or not bool(mx.all(a == b).item()):
                return False
    return True
