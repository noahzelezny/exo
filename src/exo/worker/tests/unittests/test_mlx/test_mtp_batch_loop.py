"""Batched MTP (mtp/batch_loop.py) over a toy trunk with REAL mlx-lm caches.

The toy trunk is deterministic — the next token is a function of the current
one — so the correct output of every row is known in closed form, and the
head drafts right or wrong on a rule of the content, which makes rows in the
same batch accept and reject at different steps. The caches are mlx-lm's own
(KVCache -> BatchKVCache on merge, ArraysCache for a recurrent slot) so the
lockstep trim, the whole-batch replay, and the merge / extend / filter
plumbing run the code the real model runs, not a stand-in.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from exo.worker.engines.mlx.mtp.batch_loop import MTPBatch, RowParams, admit
from exo.worker.engines.mlx.mtp.capture import capture_input
from exo.worker.engines.mlx.mtp.sampling import make_distribution

VOCAB = 17
EOS = 16


def _true_next(tok: int) -> int:
    return (tok * 7 + 3) % (VOCAB - 1)  # never EOS on its own


def _chain(prompt: list[int], n: int) -> list[int]:
    out, t = [], prompt[-1]
    for _ in range(n):
        t = _true_next(t)
        out.append(t)
    return out


class _CapturePoint(nn.Module):
    def __call__(self, x):
        return x


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = _CapturePoint()


class ToyTrunk(nn.Module):
    """Logits at every position = one-hot of `_true_next(input)`, with a
    weight big enough that sampling at temperature is the argmax too.

    Cache 0 is an attention cache (batched: BatchKVCache) fed one key per
    token, cache 1 a recurrent ArraysCache whose slot is REASSIGNED each
    forward (the property `cache_semantics="reassign"` names)."""

    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.forwards: list[tuple[int, int]] = []  # (B, S) per forward

    def make_cache(self):
        return [KVCache(), ArraysCache(1)]

    def __call__(self, ids, cache=None):
        B, S = ids.shape
        self.forwards.append((B, S))
        kv = mx.zeros((B, 1, S, 1))
        cache[0].update_and_fetch(kv, kv)
        cache[1][0] = ids[:, -1:].astype(mx.float32)
        # the "hidden state": the ids themselves, so the head can see them
        h = ids.astype(mx.float32)[..., None]
        self.model.norm(h)
        nxt = (ids * 7 + 3) % (VOCAB - 1)
        logits = mx.zeros((B, S, VOCAB))
        logits = mx.put_along_axis(logits, nxt[..., None], mx.array(100.0), axis=-1)
        return logits


class ToyHead:
    """Drafts `_true_next(last)` unless `rule(last)` says to be wrong."""

    def __init__(self, rule):
        self.rule = rule
        self.calls = 0

    def _kv(self, ids):
        B, T = ids.shape
        return mx.zeros((B, 1, T, 1))

    def advance(self, h, ids, cache):
        cache.update_and_fetch(self._kv(ids), self._kv(ids))

    def draft_logits(self, h, ids, cache):
        self.calls += 1
        cache.update_and_fetch(self._kv(ids), self._kv(ids))
        last = ids[:, -1]
        good = (last * 7 + 3) % (VOCAB - 1)
        wrong = (good + 5) % (VOCAB - 1)
        bad = self.rule(last)
        tok = mx.where(bad, wrong, good)
        logits = mx.zeros((*ids.shape, VOCAB))
        col = mx.broadcast_to(tok[:, None], ids.shape)
        return mx.put_along_axis(logits, col[..., None], mx.array(100.0), axis=-1)


def _params(max_tokens, *, temp=0.0, drafts=True):
    return RowParams(
        max_tokens=max_tokens,
        dist=make_distribution(temp=temp),
        processors=[],
        eos={EOS},
        drafts=drafts,
    )


@pytest.fixture
def rig():
    trunk = ToyTrunk()
    head = ToyHead(rule=lambda last: (last % 3) == 0)  # wrong on a third of tokens
    with capture_input(trunk.model, "norm") as get_h:
        batch = MTPBatch(trunk, head, get_h, copy_caches=False, draft_max_rows=8)
        yield trunk, head, batch, get_h


def _admit(rig, uid, prompt, max_tokens, **kw):
    trunk, head, batch, get_h = rig
    return admit(
        trunk, head, get_h, mx.array(prompt), _params(max_tokens, **kw),
        uid=uid, make_draft_cache=KVCache, prefill_step_size=3,
    )


def _run(batch, uids):
    """Drive the batch until these rows finish; return emitted tokens per uid."""
    got = {u: [] for u in uids}
    fin = {}
    for _ in range(200):
        if not batch.uids:
            break
        for rs in batch.step():
            for e in rs.tokens:
                got[rs.uid].append(e.token)
                if e.finish:
                    fin[rs.uid] = e.finish
    return got, fin


def test_rows_of_different_length_all_follow_the_trunk(rig):
    trunk, head, batch, _ = rig
    prompts = {0: [1, 2, 3, 4, 5], 1: [9], 2: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]}
    batch.extend([_admit(rig, u, p, 12) for u, p in prompts.items()])
    got, fin = _run(batch, prompts)
    for u, p in prompts.items():
        assert got[u] == _chain(p, 12), u
        assert fin[u] == "length"
    assert batch.uids == []
    # a mix of accepts and rejects actually happened
    assert head.calls > 0
    assert any(t == 2 and b == 3 for b, t in trunk.forwards)   # verify at B=3
    assert trunk.forwards.count((3, 2)) > 6                       # some replays


def test_rows_leave_at_different_steps_and_survivors_stay_correct(rig):
    trunk, head, batch, _ = rig
    batch.extend([_admit(rig, 0, [1, 2], 3), _admit(rig, 1, [5, 6, 7], 9),
                  _admit(rig, 2, [8], 16)])
    trunk.forwards.clear()          # drop the admission prefills
    got, fin = _run(batch, [0, 1, 2])
    assert got[0] == _chain([1, 2], 3) and fin[0] == "length"
    assert got[1] == _chain([5, 6, 7], 9)
    assert got[2] == _chain([8], 16)
    # the batch shrank as rows finished: forwards at B=3, then 2, then 1
    bs = [b for b, _ in trunk.forwards if _ == 2]
    assert 3 in bs and 2 in bs and 1 in bs
    assert bs == sorted(bs, reverse=True)


def test_a_row_admitted_mid_stream_joins_correctly(rig):
    trunk, head, batch, _ = rig
    batch.extend([_admit(rig, 0, [1, 2, 3], 20)])
    got = {0: [], 7: []}
    for _ in range(3):
        for rs in batch.step():
            got[rs.uid] += [e.token for e in rs.tokens]
    assert len(got[0]) == 6
    batch.extend([_admit(rig, 7, [11, 12], 8)])
    more, fin = _run(batch, [0, 7])
    got[0] += more[0]
    got[7] += more[7]
    assert got[0] == _chain([1, 2, 3], 20)
    assert got[7] == _chain([11, 12], 8)


def test_eos_stops_a_row_and_its_text_is_final(rig):
    trunk, head, batch, _ = rig
    # Make the trunk emit EOS after token 2: patch _true_next via the rule of
    # the toy — simplest is a prompt whose chain hits a token we relabel.
    row = _admit(rig, 0, [1, 2, 3], 50)
    batch.extend([row])
    # Force EOS into the params' view: the loop compares against params.eos.
    chain = _chain([1, 2, 3], 6)
    batch.params[0].eos = {chain[3]}
    got, fin = _run(batch, [0])
    assert got[0] == chain[:4]
    assert fin[0] == "stop"


def test_non_drafting_row_rides_along(rig):
    trunk, head, batch, _ = rig
    batch.extend([_admit(rig, 0, [1, 2, 3], 10), _admit(rig, 1, [4, 5], 10, drafts=False)])
    got, fin = _run(batch, [0, 1])
    assert got[0] == _chain([1, 2, 3], 10)
    assert got[1] == _chain([4, 5], 10)
    # a non-drafting row forces the replay on every step: 2 forwards per step
    assert trunk.forwards.count((2, 2)) == 10   # 5 steps x (verify + replay)


def test_remove_drops_a_row_and_keeps_the_rest_aligned(rig):
    trunk, head, batch, _ = rig
    batch.extend([_admit(rig, u, [u + 1, u + 2], 10) for u in range(3)])
    batch.step()
    batch.remove([1])
    assert batch.uids == [0, 2]
    got, fin = _run(batch, [0, 2])
    assert [0, 2] and got[0] == _chain([1, 2], 10)[2:]
    assert got[2] == _chain([3, 4], 10)[2:]


def test_temperature_rows_mix_with_greedy(rig):
    trunk, head, batch, _ = rig
    batch.extend([_admit(rig, 0, [1, 2], 8), _admit(rig, 1, [3, 4], 8, temp=0.8)])
    got, fin = _run(batch, [0, 1])
    # the toy's logits are a 100-vs-0 point mass, so sampling == argmax and
    # rejection sampling accepts exactly the correct drafts
    assert got[0] == _chain([1, 2], 8)
    assert got[1] == _chain([3, 4], 8)


def test_caches_track_every_committed_position(rig):
    trunk, head, batch, _ = rig
    prompts = {0: [1, 2, 3], 1: [4]}
    batch.extend([_admit(rig, u, p, 6) for u, p in prompts.items()])
    for _ in range(2):
        batch.step()
    attn = batch.cache[0]
    # per-row true offsets: prompt + 4 emitted
    assert attn.offset.tolist() == [3 + 4, 1 + 4]
    # the head holds one row per committed token (prompt seed 0..P-2, the
    # bootstrap row at P-1, then two per step), so its offset tracks the
    # trunk's exactly -- which is what keeps its rotary positions true
    assert batch.hcache.offset.tolist() == attn.offset.tolist()


def test_wide_batch_takes_plain_steps_and_drafts_again_when_it_narrows(rig):
    trunk, head, batch, _ = rig
    batch.draft_max_rows = 2
    batch.extend([_admit(rig, 0, [1, 2], 4), _admit(rig, 1, [3, 4, 5], 12),
                  _admit(rig, 2, [6], 12)])
    trunk.forwards.clear()
    got = {0: [], 1: [], 2: []}
    # 3 rows > ceiling: one token per row per step, single-token forwards
    for _ in range(4):
        for rs in batch.step():
            got[rs.uid] += [e.token for e in rs.tokens]
    assert trunk.forwards == [(3, 1)] * 4
    assert batch.uids == [1, 2]                      # row 0 hit max_tokens=4
    assert got[0] == _chain([1, 2], 4)
    # back under the ceiling: 2-token speculative steps resume, aligned
    more, fin = _run(batch, [1, 2])
    assert all(s == 2 for _, s in trunk.forwards[4:])
    assert got[1] + more[1] == _chain([3, 4, 5], 12)
    assert got[2] + more[2] == _chain([6], 12)
    assert batch.hcache is None                      # everything finished


def test_adaptive_rule_drafts_while_acceptance_to_the_rows_clears_the_floor():
    trunk = ToyTrunk()
    head = ToyHead(rule=lambda last: last < 0)   # always right
    with capture_input(trunk.model, "norm") as get_h:
        b = MTPBatch(trunk, head, get_h, copy_caches=False, min_accept_prob=0.5,
                     acceptance_prior=0.8)
        assert b.drafting_pays(1) and b.drafting_pays(3)      # 0.8**3 = 0.51
        assert not b.drafting_pays(4)                          # 0.8**4 = 0.41
        b.acc_est = 0.95
        assert b.drafting_pays(13) and not b.drafting_pays(14)
        # a fixed ceiling overrides the rule
        b.draft_max_rows = 2
        assert b.drafting_pays(2) and not b.drafting_pays(3)


def test_acceptance_estimate_tracks_the_head(rig):
    trunk, head, batch, _ = rig
    batch.extend([_admit(rig, 0, [1, 2, 3], 40)])
    _run(batch, [0])
    # the toy head is wrong on a third of tokens; the EMA settles near 2/3
    assert 0.5 < batch.acc_est < 0.85
