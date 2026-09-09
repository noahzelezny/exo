"""The event-log replay batch must be large enough that catch-up is not
dominated by round trips.

2026-09-09: the master answered RequestEventLog with a hardcoded 1000 events.
Each batch is one round trip gated by the replica's nack backoff (0.5s base,
10s cap), so a 400k-event log needed 400+ trips and several minutes. Long
enough that a restart mid-replay restarted the replay, which livelocked a
node in "preparing" indefinitely.
"""
import os, re, pathlib

SRC = pathlib.Path(os.path.expanduser("~/exo/src/exo/master/main.py")).read_text()


def test_replay_batch_is_not_hardcoded_1000():
    assert "command.since_idx + 1000" not in SRC, (
        "replay batch is hardcoded at 1000 -- 400+ round trips for a 400k log")


def test_replay_batch_is_configurable():
    assert "EXO_EVENT_LOG_REPLAY_BATCH" in SRC, "no env override for the batch size"


def test_replay_batch_default_is_large():
    m = re.search(r'EXO_EVENT_LOG_REPLAY_BATCH", "(\d+)"', SRC)
    assert m, "cannot find the default"
    n = int(m.group(1))
    assert n >= 10000, f"default {n} still needs {400000//n}+ round trips for a 400k log"


def test_the_handler_uses_the_constant():
    assert "command.since_idx + _EVENT_LOG_REPLAY_BATCH" in SRC, (
        "the constant exists but the handler does not use it")
