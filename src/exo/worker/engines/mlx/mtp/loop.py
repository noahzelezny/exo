"""The MTP speculative decode loop.

This is a decode *strategy*, not a runtime. mlx-lm loads the model and owns
the architecture; the only contract we depend on is

    model(tokens, cache=cache) -> logits          and    model.make_cache()

plus one per-family capture point for the pre-lm_head activation (capture.py).
Everything else here is the loop.

The head drafts token t+2 from (trunk hidden at t, embedding of t+1), so each
step verifies exactly one speculative token inside a single 2-token trunk
forward:

  accepted -> two tokens for one trunk forward.
  rejected -> the trunk's own t+2 came out of that same forward, so the step
              still emits two tokens; the caches roll back and the accepted
              pair is replayed so the state matches what was emitted.

A rejection therefore costs one extra forward, never a wrong token. At
temperature the acceptance test is exact rejection sampling with residual
correction (sampling.rejection_correct), so the emitted sequence is
distributed exactly as plain sampling from the trunk would be, for any draft
quality. At temperature 0 that rule degenerates to `draft == argmax`, which is
the greedy branch below.

What this does NOT give is bit-identical output against single-token decoding,
and demanding it is a false gate: MLX's chunked and single-token kernels
disagree at genuine near-ties (measured top-2 logit gaps of 0.25 and exactly
0.00 against a median of 3.625), and verification always happens inside a
2-token forward. Every correct speculative implementation on this runtime
inherits that. The gate is "divergence confined to near-ties" — `vqlab
mtp-bench` measures it directly.

Head-cache alignment. The head is a transformer block with its own KV cache,
and qwen4_exp derives its ROTARY POSITIONS from that cache's offset
(`Attention.__call__`: `offset = cache.offset`, then `rope(_positions(offset,
S))`). The offset is not bookkeeping — it IS the position signal.

`align="committed"` (default) keeps one head row per COMMITTED token, which
makes `cache.offset` equal the true sequence position by construction:

  - the head is seeded over the prompt at prefill, positions 0..P-2;
  - each step advances it two positions, over the pair that actually
    committed, AFTER verification — so the head only ever consumes committed
    tokens and needs no rollback of its own.

Measured worth: +5.92pp acceptance for q6 (0.8171 vs 0.7578), paired over 12
independent prompts, t=6.34, better on 12/12 — and it replicates on two other
head recipes. See `vqlab mtp-accept`; a single-prompt comparison has no power
to see this and initially read it as neutral.

`align="legacy"` is the original loop: one head forward per step, drafted
before verification. Two tokens commit per step while the head advances one,
so its rotary positions come out compressed 2x and shifted by the prompt
length, an error that grows with generation length, and its cache holds an
entry for only every other position. Kept solely so the two schemes can be
A/B'd on one build; it is not a supported configuration.

This matches the invariant MTPLX documents for the same architecture ("the
draft head's history cache (qwen4_exp: one QSA layer's KV + indexer streams)
grows one row per committed token during decode"). MTPLX reaches it by
overriding the rope offset explicitly; making the offset true by construction
gets the same positions without re-implementing the attention layer.

NOT handled yet, and both need the offset decoupled from the cache the way
MTPLX does it: windowing the prompt seed, and resetting the head cache on
long generations. MTPLX measured an uncapped draft cache decaying 86 -> 25
tok/s within a single 34k-token request. Head-cache state conditions
acceptance only — never correctness — so both are safe to add later.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Union

import mlx.core as mx

from . import registry
from .capture import capture_input
from .caches import restore, snapshot
from .sampling import Distribution, make_distribution, rejection_correct

__all__ = ["MTPResponse", "load_mtp_head", "mtp_generate", "mtp_stream_generate"]


@dataclass
class MTPResponse:
    """One emitted token, plus the running speculative statistics."""
    text: str
    token: int
    from_draft: bool          # emitted as an ACCEPTED draft (a free token)
    finish_reason: Optional[str]
    steps: int
    accepted: int
    acceptance: float         # accepted / steps
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    tail: bool = False        # trailing detokenizer flush, not a new token
    logprobs: Optional[mx.array] = None   # normalized, over the full vocab


def load_mtp_head(model, sidecar=None, family: Optional[str] = None,
                  model_path=None):
    """Load a drafting head for `model`.

    `sidecar` may be a file; if omitted, the family's sidecar name is looked
    for in `model_path`. Returns (head, spec).

    The head is optional by construction: sidecars are named outside mlx-lm's
    `model*.safetensors` glob, so a model directory carrying one still loads
    normally through the stock loader.
    """
    import pathlib

    spec = registry.resolve(model, family)
    if sidecar is None:
        if model_path is None:
            raise ValueError("pass either sidecar= or model_path=")
        sidecar = pathlib.Path(model_path) / spec.sidecar_name
    sidecar = pathlib.Path(sidecar)
    if not sidecar.exists():
        raise FileNotFoundError(
            f"no MTP sidecar at {sidecar}; build one with `vqlab mtp-pack`. "
            f"The head is optional — without it the model decodes normally.")
    arch = spec.arch_module(model)
    return spec.head_cls().from_sidecar(model, arch, sidecar), spec


def _encode(tokenizer, prompt) -> mx.array:
    if isinstance(prompt, mx.array):
        ids = prompt
    elif isinstance(prompt, str):
        ids = mx.array(tokenizer.encode(prompt))
    else:
        ids = mx.array(list(prompt))
    if ids.ndim == 1:
        ids = ids[None]
    if ids.shape[0] != 1:
        raise ValueError("speculative decoding here is single-sequence; "
                         f"got a batch of {ids.shape[0]}")
    return ids


def _eos_ids(tokenizer) -> set:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return set(ids)
    tid = getattr(tokenizer, "eos_token_id", None)
    return {tid} if tid is not None else set()


class _Detok:
    """mlx-lm's streaming detokenizer when there is one, whole-text decode
    otherwise, so a bare HF tokenizer still works."""

    def __init__(self, tokenizer):
        self._d = getattr(tokenizer, "detokenizer", None)
        self._tok = tokenizer
        self._ids: List[int] = []
        if self._d is not None:
            self._d.reset()

    def add(self, token: int) -> str:
        if self._d is not None:
            self._d.add_token(token)
            return self._d.last_segment
        self._ids.append(token)
        prev = self._text if hasattr(self, "_text") else ""
        self._text = self._tok.decode(self._ids)
        return self._text[len(prev):]

    def finalize(self) -> str:
        if self._d is not None:
            self._d.finalize()
            return self._d.last_segment
        return ""


def mtp_stream_generate(
    model,
    tokenizer,
    prompt,
    head,
    *,
    family: Optional[str] = None,
    max_tokens: int = 256,
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = 0,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: List[int] = [],
    logits_processors: Optional[List[Callable]] = None,
    prefill_step_size: int = 2048,
    align: str = "committed",
    prompt_cache=None,
    want_logprobs: bool = False,
) -> Iterator[MTPResponse]:
    """Stream tokens from `model`, drafting each second token with `head`.

    Sampling parameters mirror `mlx_lm.sample_utils.make_sampler`; the
    resulting output distribution is the one mlx-lm would produce for the same
    parameters, preserved through verification by rejection sampling.

    `align` selects the head-cache scheme: "committed" (default) or "legacy".
    See the module docstring -- "legacy" exists for A/B measurement only.

    `prompt_cache` lets a caller (a server) own the trunk cache. It must be
    EMPTY: the head is seeded from this prompt starting at position 0, so a
    cache carrying a reused prefix would shift every rotary position.
    """
    if align not in ("committed", "legacy"):
        raise ValueError(f"align must be 'committed' or 'legacy', got {align!r}")
    spec = registry.resolve(model, family)
    dist = make_distribution(temp, top_p, min_p, min_tokens_to_keep, top_k,
                             xtc_probability, xtc_threshold, xtc_special_tokens)
    copy_caches = spec.cache_semantics != "reassign"
    eos = _eos_ids(tokenizer)
    detok = _Detok(tokenizer)
    ids = _encode(tokenizer, prompt)
    arch = spec.arch_module(model)

    emitted: List[int] = []
    processors = list(logits_processors or [])

    def pick(row: mx.array):
        """row: [1, V] -> (token [1], Distribution or None)."""
        for proc in processors:
            row = proc(mx.array(emitted), row)
        if dist is None:
            return mx.argmax(row, axis=-1), None
        d = dist(row)
        return d.sample(), d

    with capture_input(model.model, spec.capture) as get_h:
        cache = model.make_cache() if prompt_cache is None else prompt_cache
        # A caller-supplied cache that already holds a prefix would put the
        # head's positions back exactly where the alignment fix took them
        # from: the head is seeded from THIS prompt starting at position 0,
        # so a trunk cache at offset O silently shifts every rotary position
        # by O. Refuse rather than degrade -- a server reusing prompt caches
        # must either pair a head cache with each trunk cache or disable
        # reuse (see vqlab/serve.py).
        used = max((getattr(c, "offset", 0) or 0) for c in cache) if cache else 0
        if used and align == "committed":
            raise ValueError(
                f"prompt_cache already holds {used} positions; the MTP head "
                f"cannot be aligned to a reused trunk cache yet. Pass a fresh "
                f"cache, or use align='legacy' (which is misaligned anyway).")
        dcache = spec.make_draft_cache(arch)

        # prefill. The per-chunk hidden states are kept when the head is
        # being seeded, because the head's input at position j is
        # (h_j, embed(x_{j+1})) for EVERY prompt position, not just the last.
        n = ids.shape[1]
        h_chunks = []
        for i in range(0, n - 1, prefill_step_size):
            chunk = ids[:, i:min(i + prefill_step_size, n - 1)]
            if chunk.shape[1]:
                model(chunk, cache=cache)
                # Evaluate the CACHE (and the captured hidden state, which the
                # head is seeded from) -- never the logits. MLX is lazy, so
                # forcing the returned logits here would materialize the whole
                # lm_head projection for every prefill position: at a 2048
                # chunk and a ~151k vocabulary that is a large matmul per
                # chunk whose result is thrown away. mlx-lm's own prefill
                # evaluates `[c.state for c in cache]` for exactly this
                # reason. Only the final single-token forward below needs
                # logits.
                want = [c.state for c in cache if hasattr(c, "state")]
                if align == "committed":
                    h = get_h()
                    h_chunks.append(h)
                    want.append(h)
                mx.eval(want)
                mx.clear_cache()
        logits = model(ids[:, max(n - 1, 0):], cache=cache)
        h_chunks.append(get_h())
        h_last = h_chunks[-1][:, -1:]
        row_t1 = logits[:, -1]
        t1, _ = pick(row_t1)
        mx.eval(t1, h_last)

        if align == "committed":
            # Seed positions 0..P-2 so the head enters decoding with the same
            # history the trunk has, and with cache.offset == P-1. Position
            # P-1 is not seeded: its input needs x_P, the first sampled
            # token, and it is the bootstrap draft below.
            if n >= 2:
                h_all = mx.concatenate(h_chunks, axis=1)
                head.advance(h_all[:, :n - 1], ids[:, 1:n], dcache)
                del h_all
            h_chunks.clear()
            mx.clear_cache()
            # Bootstrap draft: position P-1, input (h_{P-1}, x_P).
            draft_row = head.draft_logits(h_last, t1[None], dcache)[:, -1]
            mx.eval(draft_row)

        steps = accepted = 0
        start = time.perf_counter()
        finish: Optional[str] = None

        while finish is None:
            steps += 1
            if align == "legacy":
                # Drafted BEFORE verification off a cache that advances one
                # position per step, so it must be rolled back on rejection.
                dsnap = snapshot([dcache], copy=copy_caches)
                draft_row = head.draft_logits(h_last, t1[None], dcache)[:, -1]
            if dist is None:
                d2 = mx.argmax(draft_row, axis=-1)
                q = None
            else:
                for proc in processors:
                    draft_row = proc(mx.array(emitted), draft_row)
                q = dist(draft_row)
                d2 = q.sample()
            mx.eval(d2)

            csnap = snapshot(cache, copy=copy_caches)
            lg2 = model(mx.concatenate([t1, d2])[None], cache=cache)

            if dist is None:
                true_t2 = mx.argmax(lg2[:, 0], axis=-1)
                mx.eval(true_t2)
                ok = bool((true_t2 == d2).item())
                t2 = d2 if ok else true_t2
            else:
                p = dist(lg2[:, 0])
                acc, t2 = rejection_correct(p.probs, q.probs, d2)
                mx.eval(acc, t2)
                ok = bool(acc.item())

            if ok:
                accepted += 1
            else:
                # Roll back to before the drafted pair and replay the tokens
                # actually emitted, so the caches match the emitted stream.
                restore(cache, csnap)
                if align == "legacy":
                    restore([dcache], dsnap)
                lg2 = model(mx.concatenate([t1, t2])[None], cache=cache)

            # The logprobs reported for each token are the TRUNK's, taken
            # from the forward that actually produced it -- for an accepted
            # draft too, since the trunk verified it in the same forward. A
            # rejected token's logprob is likewise the trunk's, which is the
            # honest number: what the target model assigned to what we
            # emitted. The head's own draft distribution is never reported.
            for tok, from_draft, row in ((t1, False, row_t1),
                                         (t2, ok, lg2[:, 0])):
                token = int(tok.item())
                emitted.append(token)
                if token in eos:
                    finish = "stop"
                elif len(emitted) >= max_tokens:
                    finish = "length"
                elapsed = time.perf_counter() - start
                lp = None
                if want_logprobs:
                    r32 = row.astype(mx.float32)
                    lp = (r32 - mx.logsumexp(r32, axis=-1, keepdims=True))[0]
                yield MTPResponse(
                    text=detok.add(token), token=token, from_draft=from_draft,
                    finish_reason=finish, steps=steps, accepted=accepted,
                    acceptance=accepted / steps, generation_tokens=len(emitted),
                    generation_tps=len(emitted) / elapsed if elapsed else 0.0,
                    peak_memory=mx.get_peak_memory() / 2**30, logprobs=lp,
                )
                if finish is not None:
                    break
            if finish is not None:
                break

            # `get_h()` is the hidden state of the forward that actually
            # committed — the replay on the rejection path, not the discarded
            # speculative one — so it is (h_i, h_{i+1}) for the emitted pair.
            h_pair = get_h()
            h_last = h_pair[:, -1:]
            row_t1 = lg2[:, 1]
            t_next, _ = pick(row_t1)
            mx.eval(t_next, h_pair)

            if align == "committed":
                # Advance the head over the two positions that just
                # committed: (h_i, x_{i+1}) and (h_{i+1}, x_{i+2}). Its cache
                # now holds one row per committed token, so cache.offset is
                # still the true position. The second output drafts x_{i+3},
                # which is exactly the next step's speculative token; the
                # first is redundant as a draft and is there to fill the row.
                pair_ids = mx.concatenate([t2, t_next])[None]
                draft_row = head.draft_logits(h_pair, pair_ids, dcache)[:, -1]
                mx.eval(draft_row)
            t1 = t_next

        tail = detok.finalize()
        if tail:
            elapsed = time.perf_counter() - start
            yield MTPResponse(
                text=tail, token=emitted[-1], from_draft=False, tail=True,
                finish_reason=finish, steps=steps, accepted=accepted,
                acceptance=accepted / max(steps, 1),
                generation_tokens=len(emitted),
                generation_tps=len(emitted) / elapsed if elapsed else 0.0,
                peak_memory=mx.get_peak_memory() / 2**30,
            )


def mtp_generate(model, tokenizer, prompt, head, *, verbose: bool = False,
                 **kwargs) -> str:
    """Generate a completion with MTP speculative drafting; return the text.

    Mirrors `mlx_lm.generate`'s shape. `mtp_stream_generate` exposes the
    per-token stream and the acceptance statistics.
    """
    parts, last = [], None
    for r in mtp_stream_generate(model, tokenizer, prompt, head, **kwargs):
        parts.append(r.text)
        if verbose:
            print(r.text, end="", flush=True)
        last = r
    if verbose and last is not None:
        print(f"\n\n{last.generation_tokens} tokens, "
              f"{last.generation_tps:.2f} tok/s, "
              f"acceptance {last.acceptance:.3f} over {last.steps} steps, "
              f"peak memory {last.peak_memory:.2f} GiB", flush=True)
    return "".join(parts)
