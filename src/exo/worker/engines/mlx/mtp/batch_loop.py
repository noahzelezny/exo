"""MTP speculative decoding over a BATCH of sequences.

loop.py runs one sequence: draft one token with the head, verify it inside a
2-token trunk forward, roll back and replay on rejection. This module runs
that same step over B rows at once, which is what lets the batch engine keep
drafting when several requests are in flight instead of trading the head
away for batching.

What makes it batchable at all is a property of the 1-token draft: EVERY row
advances by exactly two positions per step, accepted or not.

  accepted -> the verify forward committed [t1, d2]; the caches are right.
  rejected -> the caches hold [t1, d2] with the wrong second token; the
              trunk's own t2 came out of the same forward, so the step still
              emits two tokens, but position offset-1 has to be rewritten.

Because the row count in the KV cache is lockstep, the batched caches
(mlx-lm's BatchKVCache, one offset per row, shared write index) never need
per-row trimming. Rejection is handled as ONE whole-batch trim(2) and ONE
whole-batch replay forward with the tokens that actually committed — for a
row that accepted, the replay writes back exactly what was there. That costs
an extra forward whenever ANY row rejects, i.e. with acceptance a and B rows
the expected forwards per step are 1 + (1 - a^B) for 2B tokens, against B
tokens per forward without drafting. It is a decaying win, not a free one:
measure it (see the numbers in the runner's engine_mode.py) before assuming
it beats plain batching at a given B.

A row that cannot draft (a request carrying images, whose head seeding the
vision embedding patch would corrupt) rides along: it pays the replay every
step and gets the trunk's own tokens, never a drafted one.

Sampling is per row — temperature, top-p, processors, and the acceptance
test are the row's own — so a batch may mix greedy and sampled requests. At
temperature the verdict is the same exact rejection sampling loop.py uses
(sampling.rejection_correct); at temperature 0 it is `draft == argmax`. One
deliberate difference from loop.py: a row's logits processors (repetition
penalty, eos ban) are applied to the TRUNK row that verifies the draft, not
only to the draft and to t1 — otherwise a penalised token could be committed
through the verify path that sampling would have refused.

Single node only. The stage-1 pipeline seam (pipeline.py's Coordinator) is
not threaded through here; the batch engine is built for single-node
instances and the builder gates batched drafting on `group is None`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional

import mlx.core as mx
from mlx_lm.generate import _extend_cache, _merge_caches

from .caches import restore, snapshot
from .sampling import Distribution, rejection_correct

__all__ = ["RowParams", "Row", "Emitted", "RowStep", "MTPBatch", "admit",
           "default_draft_max_rows"]


def default_draft_max_rows() -> int:
    """Rows up to which the batch drafts; above it, plain one-token steps.

    The whole-batch replay costs a forward whenever ANY row rejects, so the
    speculative win decays with rows: measured 2026-09-17 on the M3 with
    Qwen3.8-Flash-Next-VQ-2.1bpw (acceptance ~0.75), drafting beat plain
    batching at 1 and 2 rows (23 vs 18, 27-33 vs 25 tok/s aggregate) and lost
    at 3 (32 vs 34). Override with EXO_MTP_BATCH_MAX_ROWS; a higher-acceptance
    head earns a higher ceiling — measure, then raise it.
    """
    try:
        return max(1, int(os.environ.get("EXO_MTP_BATCH_MAX_ROWS", "2")))
    except ValueError:
        return 2


@dataclass
class RowParams:
    """Everything about a request the loop needs, fixed at admission."""

    max_tokens: int
    #: None = greedy (temperature 0). Otherwise logits [1, V] -> Distribution.
    dist: Optional[Callable[[mx.array], Distribution]]
    processors: List[Callable[[mx.array, mx.array], mx.array]]
    eos: set
    #: False for a row the head must not draft for; it still decodes correctly.
    drafts: bool = True


@dataclass
class Row:
    """One admitted sequence: prefilled, first token sampled, head seeded."""

    uid: int
    params: RowParams
    cache: list                    # single-row trunk caches, full prompt inside
    hcache: Any                    # head cache (an EMPTY one when not drafting)
    t1: mx.array                   # [1] int32, the first token to commit
    row_t1: mx.array               # [1, V] the logits t1 was sampled from
    draft_row: Optional[mx.array]  # [1, V] the head's draft of t2, or None
    n_prompt: int
    drafts: bool


@dataclass
class Emitted:
    token: int
    from_draft: bool
    finish: Optional[str]
    logits: mx.array               # [V] the trunk row that produced it


@dataclass
class RowStep:
    uid: int
    tokens: List[Emitted]
    #: running per-row acceptance: (accepted drafts, speculative steps)
    accepted: int
    steps: int


def _apply(row: mx.array, procs, emitted: List[int]) -> mx.array:
    for proc in procs:
        row = proc(mx.array(emitted), row)
    return row


def _pick(row: mx.array, p: RowParams, emitted: List[int]) -> mx.array:
    """row: [1, V] -> token [1]."""
    row = _apply(row, p.processors, emitted)
    if p.dist is None:
        return mx.argmax(row, axis=-1)
    return p.dist(row).sample()


def admit(
    model,
    head,
    get_h: Callable[[], mx.array],
    ids: mx.array,
    params: RowParams,
    *,
    uid: int,
    make_draft_cache: Callable[[], Any],
    prefill_step_size: int = 2048,
    prefill_ctx=None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Row:
    """Prefill one prompt and seed the head for it: loop.py's prefill, kept
    as a standalone so the batch can admit rows between steps.

    `prefill_ctx` (a zero-arg context-manager factory) wraps the chunked
    prompt forwards only — a vision request's embedding patch goes there,
    and the final single-token forward that produces the first logits runs
    outside it, as it does on the stock path.

    The head is seeded from position 0 (see loop.py on alignment), which is
    why an admitted row always starts from a FRESH trunk cache: a reused
    prefix would shift every rotary position the head sees.
    """
    import contextlib

    if ids.ndim == 1:
        ids = ids[None]
    n = int(ids.shape[1])
    cache = model.make_cache()
    drafts = params.drafts and head is not None
    dcache = make_draft_cache()

    h_chunks: List[mx.array] = []
    ctx = prefill_ctx() if prefill_ctx is not None else contextlib.nullcontext()
    with ctx:
        for i in range(0, n - 1, prefill_step_size):
            end = min(i + prefill_step_size, n - 1)
            chunk = ids[:, i:end]
            if not chunk.shape[1]:
                continue
            model(chunk, cache=cache)
            # Evaluate the cache and the captured hidden state, never the
            # logits: forcing them would materialise the lm_head projection
            # for every prompt position (loop.py says the same).
            want = [c.state for c in cache if hasattr(c, "state")]
            if drafts:
                h = get_h()
                h_chunks.append(h)
                want.append(h)
            mx.eval(want)
            mx.clear_cache()
            if on_progress is not None:
                on_progress(end, n)

    logits = model(ids[:, max(n - 1, 0):], cache=cache)
    row_t1 = logits[:, -1]
    t1 = _pick(row_t1, params, []).astype(mx.int32)

    draft_row = None
    if drafts:
        h_chunks.append(get_h())
        h_last = h_chunks[-1][:, -1:]
        mx.eval(t1, h_last)
        if n >= 2:
            # Positions 0..P-2, so the head enters decoding with the trunk's
            # history and cache.offset == P-1 == the true position.
            h_all = mx.concatenate(h_chunks, axis=1)
            head.advance(h_all[:, : n - 1], ids[:, 1:n], dcache)
            del h_all
        h_chunks.clear()
        mx.clear_cache()
        # Bootstrap draft at position P-1: input (h_{P-1}, x_P).
        draft_row = head.draft_logits(h_last, t1[None], dcache)[:, -1]
        mx.eval(draft_row)
    else:
        mx.eval(t1)

    return Row(
        uid=uid, params=params, cache=cache, hcache=dcache, t1=t1,
        row_t1=row_t1, draft_row=draft_row, n_prompt=n, drafts=drafts,
    )


class MTPBatch:
    """The batched speculative state: B rows advancing two tokens a step.

    Rows enter through `extend` (between steps) and leave when a token
    finishes them or through `remove`. Row order is `uids`; every per-row
    structure below is indexed the same way and filtered together.
    """

    def __init__(self, model, head, get_h: Callable[[], mx.array], *,
                 copy_caches: bool, draft_max_rows: int | None = None):
        self.model = model
        self.head = head
        self.get_h = get_h
        self.copy_caches = copy_caches
        self.draft_max_rows = (
            draft_max_rows if draft_max_rows is not None else default_draft_max_rows()
        )

        self.uids: List[int] = []
        self.params: List[RowParams] = []
        self.emitted: List[List[int]] = []
        self.drafts: List[bool] = []
        self.accepted: List[int] = []
        self.steps: List[int] = []

        self.cache: list = []               # batched trunk caches
        self.hcache: Any = None             # batched head cache
        self.t1: Optional[mx.array] = None          # [B] int32
        self.row_t1: Optional[mx.array] = None      # [B, V]
        self.draft_row: Optional[mx.array] = None   # [B, V]

    def __len__(self) -> int:
        return len(self.uids)

    @property
    def any_drafting(self) -> bool:
        return self.head is not None and any(self.drafts)

    # ------------------------------------------------------------ membership

    def extend(self, rows: Iterable[Row]) -> None:
        rows = list(rows)
        if not rows:
            return
        self.cache = _extend_cache(self.cache, _merge_caches([r.cache for r in rows]))
        if self.head is not None:
            hcs = [r.hcache for r in rows]
            merged = type(hcs[0]).merge(hcs)
            if self.hcache is None:
                self.hcache = merged
            else:
                self.hcache.extend(merged)

        V = int(rows[0].row_t1.shape[-1])
        t1 = mx.concatenate([r.t1 for r in rows]).astype(mx.int32)
        row_t1 = mx.concatenate([r.row_t1 for r in rows], axis=0)
        drafts = [
            r.draft_row if r.draft_row is not None
            else mx.zeros((1, V), dtype=row_t1.dtype)
            for r in rows
        ]
        draft_row = mx.concatenate(drafts, axis=0) if self.head is not None else None

        if self.t1 is None:
            self.t1, self.row_t1, self.draft_row = t1, row_t1, draft_row
        else:
            self.t1 = mx.concatenate([self.t1, t1])
            self.row_t1 = mx.concatenate([self.row_t1, row_t1], axis=0)
            if draft_row is not None:
                self.draft_row = mx.concatenate([self.draft_row, draft_row], axis=0)

        for r in rows:
            self.uids.append(r.uid)
            self.params.append(r.params)
            self.emitted.append([])
            self.drafts.append(r.drafts)
            self.accepted.append(0)
            self.steps.append(0)
        # A row's arrays are consumed by the first step; nothing to eval here
        # that the step will not force anyway.

    def filter(self, keep: List[int]) -> None:
        """Keep only these row indices, in this order."""
        if keep == list(range(len(self.uids))):
            return
        self.uids = [self.uids[i] for i in keep]
        self.params = [self.params[i] for i in keep]
        self.emitted = [self.emitted[i] for i in keep]
        self.drafts = [self.drafts[i] for i in keep]
        self.accepted = [self.accepted[i] for i in keep]
        self.steps = [self.steps[i] for i in keep]
        if not keep:
            self.cache = []
            self.hcache = None
            self.t1 = self.row_t1 = self.draft_row = None
            return
        for c in self.cache:
            c.filter(keep)
        if self.hcache is not None:
            self.hcache.filter(keep)
        idx = mx.array(keep)
        self.t1 = self.t1[idx]
        self.row_t1 = self.row_t1[idx]
        if self.draft_row is not None:
            self.draft_row = self.draft_row[idx]

    def remove(self, uids: Iterable[int]) -> None:
        drop = set(uids)
        self.filter([i for i, u in enumerate(self.uids) if u not in drop])

    # ------------------------------------------------------------------ step

    def step(self) -> List[RowStep]:
        """One speculative step: every live row commits two tokens.

        Returns one RowStep per row in batch order (rows that finish are
        removed from the batch before returning, so `uids` afterwards lists
        only the survivors).
        """
        B = len(self.uids)
        if B == 0:
            return []
        assert self.t1 is not None and self.row_t1 is not None
        if B > self.draft_max_rows:
            return self._plain_step()

        # A row drafts this step iff it is a drafting row AND the head has
        # produced a draft for it (the batch may hold only non-drafting rows).
        live = [self.drafts[i] and self.draft_row is not None for i in range(B)]

        # --- draft d2 per row -------------------------------------------
        d2_rows: List[mx.array] = []
        qs: List[Optional[Distribution]] = []
        for i in range(B):
            p = self.params[i]
            if not live[i]:
                # Placeholder that is always "rejected" below; the replay
                # supplies the trunk's own token.
                d2_rows.append(self.t1[i:i + 1])
                qs.append(None)
                continue
            row = _apply(self.draft_row[i:i + 1], p.processors, self.emitted[i])
            if p.dist is None:
                d2_rows.append(mx.argmax(row, axis=-1))
                qs.append(None)
            else:
                q = p.dist(row)
                d2_rows.append(q.sample())
                qs.append(q)
        d2 = mx.concatenate(d2_rows).astype(mx.int32)

        # --- verify: one 2-wide forward over the batch -------------------
        csnap = snapshot(self.cache, copy=self.copy_caches)
        lg2 = self.model(mx.stack([self.t1, d2], axis=1), cache=self.cache)

        # --- verdicts ----------------------------------------------------
        oks: List[Any] = [None] * B
        t2_rows: List[Any] = [None] * B
        lazy: List[mx.array] = []
        for i in range(B):
            p = self.params[i]
            row = _apply(lg2[i:i + 1, 0], p.processors, self.emitted[i])
            if not live[i]:
                # Non-drafting row: the trunk's own token, committed through
                # the replay below.
                t2 = mx.argmax(row, axis=-1) if p.dist is None else p.dist(row).sample()
                oks[i] = False
                t2_rows[i] = t2
                lazy.append(t2)
            elif p.dist is None:
                true_t2 = mx.argmax(row, axis=-1)
                ok = true_t2 == d2[i:i + 1]
                oks[i] = ok
                t2_rows[i] = mx.where(ok, d2[i:i + 1], true_t2)
                lazy += [ok, t2_rows[i]]
            else:
                pt = p.dist(row)
                acc, t2 = rejection_correct(pt.probs, qs[i].probs, d2[i:i + 1])
                oks[i] = acc
                t2_rows[i] = t2
                lazy += [acc, t2]
        mx.eval(*lazy)
        ok_flags = [bool(o.item()) if isinstance(o, mx.array) else bool(o) for o in oks]
        t2 = mx.concatenate(t2_rows).astype(mx.int32)

        for i in range(B):
            if live[i]:
                self.steps[i] += 1
                self.accepted[i] += int(ok_flags[i])

        # --- rollback + replay if anyone rejected ------------------------
        if not all(ok_flags):
            restore(self.cache, csnap)
            lg2 = self.model(mx.stack([self.t1, t2], axis=1), cache=self.cache)

        # --- emit --------------------------------------------------------
        t1_list = self.t1.tolist()
        t2_list = t2.tolist()
        out: List[RowStep] = []
        keep: List[int] = []
        for i in range(B):
            p = self.params[i]
            em = self.emitted[i]
            toks: List[Emitted] = []
            finish: Optional[str] = None
            for tok, from_draft, row in (
                (t1_list[i], False, self.row_t1[i]),
                (t2_list[i], ok_flags[i], lg2[i, 0]),
            ):
                em.append(tok)
                if tok in p.eos:
                    finish = "stop"
                elif len(em) >= p.max_tokens:
                    finish = "length"
                toks.append(Emitted(token=tok, from_draft=from_draft,
                                    finish=finish, logits=row))
                if finish is not None:
                    break
            out.append(RowStep(uid=self.uids[i], tokens=toks,
                               accepted=self.accepted[i], steps=self.steps[i]))
            if finish is None:
                keep.append(i)

        # --- next-step state for the survivors ---------------------------
        row_t1 = lg2[:, 1]
        h_pair = self.get_h() if self.any_drafting else None
        t_next_rows = [
            _pick(row_t1[i:i + 1], self.params[i], self.emitted[i]) for i in keep
        ]
        self.filter(keep)
        if not keep:
            return out
        idx = mx.array(keep)
        row_t1 = row_t1[idx]
        t2k = t2[idx]
        t_next = mx.concatenate(t_next_rows).astype(mx.int32)

        if self.any_drafting:
            assert h_pair is not None
            # Advance the head over the two committed positions of every
            # surviving row: (h_i, x_{i+1}) and (h_{i+1}, x_{i+2}). The
            # second output drafts x_{i+3}, next step's speculative token.
            pair_ids = mx.stack([t2k, t_next], axis=1)
            self.draft_row = self.head.draft_logits(h_pair[idx], pair_ids, self.hcache)[:, -1]
            mx.eval(t_next, self.draft_row)
        else:
            self.draft_row = None if self.head is None else self.draft_row
            mx.eval(t_next)
        self.t1 = t_next
        self.row_t1 = row_t1
        return out

    def _plain_step(self) -> List[RowStep]:
        """One stock decode step: every row commits t1 and samples the next.

        Taken when the batch is too wide for drafting to pay (draft_max_rows).
        The head still advances one position per row so its cache stays one
        row per committed token, and the next draft is ready the moment the
        batch shrinks back under the ceiling.
        """
        B = len(self.uids)
        assert self.t1 is not None and self.row_t1 is not None
        lg = self.model(self.t1[:, None], cache=self.cache)     # [B, 1, V]
        t1_list = self.t1.tolist()
        out: List[RowStep] = []
        keep: List[int] = []
        for i in range(B):
            p = self.params[i]
            em = self.emitted[i]
            tok = t1_list[i]
            em.append(tok)
            finish: Optional[str] = None
            if tok in p.eos:
                finish = "stop"
            elif len(em) >= p.max_tokens:
                finish = "length"
            out.append(RowStep(
                uid=self.uids[i],
                tokens=[Emitted(token=tok, from_draft=False, finish=finish,
                                logits=self.row_t1[i])],
                accepted=self.accepted[i], steps=self.steps[i]))
            if finish is None:
                keep.append(i)

        row_t1 = lg[:, 0]
        h = self.get_h() if self.any_drafting else None
        t_next_rows = [
            _pick(row_t1[i:i + 1], self.params[i], self.emitted[i]) for i in keep
        ]
        self.filter(keep)
        if not keep:
            return out
        idx = mx.array(keep)
        row_t1 = row_t1[idx]
        t_next = mx.concatenate(t_next_rows).astype(mx.int32)
        if self.any_drafting:
            assert h is not None
            # (h_i, x_{i+1}) for the one position that committed; its output
            # drafts x_{i+2}, which is next step's speculative token.
            self.draft_row = self.head.draft_logits(
                h[idx], t_next[:, None], self.hcache)[:, -1]
            mx.eval(t_next, self.draft_row)
        else:
            mx.eval(t_next)
        self.t1 = t_next
        self.row_t1 = row_t1
        return out
