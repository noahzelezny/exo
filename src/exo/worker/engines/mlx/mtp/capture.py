"""Per-family capture of the pre-lm_head activation.

The MTP head drafts token t+2 from the trunk's hidden state at t. mlx-lm's
public contract is `model(tokens, cache=cache) -> logits`; it hands back
logits and nothing else, so the hidden state has to be taken from inside.

The technique is a wrapper module that records its own input and forwards.
That much is unavoidable. What is avoidable is doing it as a permanent
monkeypatch on a hardcoded attribute name: here the attribute is a registry
field, and the wrap is a context manager, so the trunk is left exactly as it
was found even if generation raises.

The wrapper is an nn.Module so that the wrapped submodule stays reachable from
`model.parameters()` while it is installed. The captured array is kept in a
closure cell rather than as an attribute, so it never enters the module tree.
"""
from __future__ import annotations

from contextlib import contextmanager

import mlx.nn as nn


def _resolve(root, path: str):
    """(owner, attr) for a dotted path, so nested capture points work."""
    obj = root
    parts = path.split(".")
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj, parts[-1]


def _spy(inner, sink):
    class _Capture(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner

        def __call__(self, x, *args, **kwargs):
            sink[:] = [x]
            return self.inner(x, *args, **kwargs)

    return _Capture()


@contextmanager
def capture_input(core, path: str):
    """Yield a getter for the most recent input to `core.<path>`.

    The getter raises if nothing has been captured yet, which is the honest
    failure for a registry entry that names the wrong module: a silently stale
    hidden state would show up only as degraded acceptance."""
    owner, attr = _resolve(core, path)
    inner = getattr(owner, attr)
    sink: list = []
    setattr(owner, attr, _spy(inner, sink))
    try:
        def get():
            if not sink:
                raise RuntimeError(
                    f"nothing captured at {path!r}: the trunk forward did not "
                    f"call it. Check the family's capture path against the "
                    f"installed architecture.")
            return sink[0]

        yield get
    finally:
        setattr(owner, attr, inner)
