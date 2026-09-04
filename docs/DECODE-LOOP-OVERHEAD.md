# Decode-loop overhead: where the two stock engines lose

Date: 2026-09-03
Branch: `mtp-stage1`
Method: code analysis on the M3 checkout + single-process `mlx` microbenchmarks
(no model loads, no cluster access — a measurement campaign owned the cluster
all night).

Status of the conclusion: **Q1 (sequential's flat 4x deficit) is NOT solved.**
Six named suspects were falsified by direct measurement. **Q2 (the batch
engine's length collapse) is partially solved**: the growth term is proven
*not* to be the pipeline forward, which eliminates the whole family of
"attention/KV/interconnect gets slower with context" explanations and points
the search at the batch engine's own per-step work. The single log line that
finishes the job already exists in the tree and needs one read-only run.

---

## 0. The numbers this document has to explain

Measured 2026-09-03, 2-node M3+M4 pipeline, GLM 2.7bpw, 300-token greedy.
Converted to per-emitted-token latency, because that is the quantity the code
controls:

| path | 300 tok | 2000 tok | ms/tok @300 | ms/tok @2000 |
|---|---|---|---|---|
| batch (`BatchGenerator`, production default) | 19.5 tok/s | 5.7 tok/s | 51.3 | 175.4 |
| sequential stock (`EXO_NO_BATCH=1`) | 5.0 tok/s | — | 200.0 | — |
| sequential + stage-1 MTP | 22.3 tok/s | 11.0 tok/s | 44.8 | 90.9 |
| single-box plain mlx-lm | 20.3 tok/s | 20.3 tok/s | 49.3 | 49.3 |

Two separate pathologies, and they are not the same bug:

- sequential stock is **flat and 4x slow** — it loses ~150 ms/token at any length;
- batch is **fast then collapses** — it loses an *additional* ~124 ms/token
  between 300 and 2000.

---

## 1. MEASURED — six falsifications

All run in the live `exo` env on M3 (`/opt/anaconda3/envs/exo`, mlx
`0.32.0.dev20260622+4c8d2590`, the jaccl fork), single process, no model
weights. Scripts were scratch; the numbers are reproduced below in full so
nobody has to re-derive them to disagree.

**F1. `mx.array(growing python list)`** — the `opt_batch_gen` patch rebuilds the
whole token list into an array every token (see §3.3).
`N=100: 0.0019 ms · 300: 0.0031 · 1000: 0.0081 · 2000: 0.0152 · 4000: 0.0290`.
Linear, and **15 µs at N=2000**. Real O(N), irrelevant at these lengths.

**F2. `mx.concat([tokens, new])` growing** — `generate_step`'s token accumulator
on the sequential path.
`100: 0.1999 ms · 300: 0.1787 · 1000: 0.1809 · 2000: 0.1715 · 4000: 0.1742`.
**Flat.** Not O(N)-visible.

**F3. repetition-penalty processor at `penalty=1.0`** over a growing token array,
V=151936.
`100: 0.3168 ms · 300: 0.2862 · 1000: 0.2747 · 2000: 0.2891 · 4000: 0.2838`.
**Flat**, ~0.28 ms.

**F4. `mx.depends(cache.keys, output)` + `mx.eval`** — the construct
`PipelineLastLayer` applies to the boundary layer's KV every forward. Shaped as
GLM's boundary cache (1, 64, N, 128) fp16, i.e. 1.6 → 62.5 MiB.
`100: 0.0230 ms · 300: 0.0225 · 1000: 0.0208 · 2000: 0.0185 · 4000: 0.0211`.
**Flat, and does not copy** — `mx.depends` is a dependency edge, not a
materialisation. My prior that this was an O(context) copy per token was wrong.

**F5. `KVCache.update_and_fetch` + a forced `mx.eval(cache.keys)` per step**,
2200 steps, measured in bands:

| variant | tok 100–300 | 900–1100 | 1900–2100 |
|---|---|---|---|
| baseline | 0.244 ms | 0.213 ms | 0.213 ms |
| `+ mx.eval(cache.keys)` | 0.212 ms | 0.219 ms | 0.212 ms |
| `+ depends + eval(keys)` | 0.241 ms | 0.245 ms | 0.250 ms |

**Flat.** Forcing the KV buffer to materialise separately each step costs
nothing and does not grow.

**F6. `StreamingDetokenizer.last_segment`**, the classic O(T²) trap — mlx-lm's
own docstring advertises it. Naive vs BPE, 2400 tokens:
Naive `100–300: 0.0137 ms · 900–1100: 0.0090 · 1900–2100: 0.0060` (total 0.02 s
for 2000 tokens); BPE `~0.0010 ms` flat (total 0.002 s).
Naive is 10x worse and still **microseconds**. Not a factor either way.

**What F1–F6 establish.** The entire host-side per-token bookkeeping budget of
*both* engines is well under 1 ms, against measured per-token times of 51–200
ms. **No host-side Python/mlx bookkeeping explains either pathology.** Every
"something is quietly O(context) on the CPU" hypothesis in the brief is dead.

---

## 2. DEDUCED FROM THE MEASURED NUMBERS — the collapse is not the forward

This is the strongest result in the document and it uses only §0.

Forwards through the distributed pipeline per emitted token:

- sequential stock: **1.000** (one 1-token forward per token)
- batch: **1.000**
- MTP at the measured acceptance 0.887: one 2-wide verify forward per step,
  plus one replay forward on rejection, for two emitted tokens →
  `(1 + 0.113) / 2 =` **0.557**

Now take a 2-emitted-token unit and let `T` be the cost of one pipeline trunk
forward, `H` the MTP head forward (node-local, no pipeline), `B` the two control
broadcasts per step.

At 300 tokens:
`2T = 2 × 51.3 = 102.6 ms` → `T = 51.3 ms`
`1.113T + H + B = 2 × 44.8 = 89.7 ms` → `H + B = 32.6 ms`. Consistent.

At 2000 tokens, *if the forward were what grows*:
`2T' = 2 × 175.4 = 350.8 ms` → `T' = 175.4 ms`
`1.113T' = 195.2 ms`, but the whole MTP step measures `2 × 90.9 = 181.8 ms`.

**195.2 > 181.8.** MTP would have to complete 1.113 trunk forwards, a head
forward and two broadcasts in less time than 1.113 trunk forwards alone. The
premise is false.

**Therefore the trunk forward does not grow appreciably with context**, and
neither does the interconnect, the KV attention, or anything else inside
`model(...)`. Whatever costs the batch engine 124 ms/token at 2000 tokens is
work the batch engine does *around* the forward and the MTP path does not do.

The corroborating signature: the MTP-over-batch speed ratio is **1.14x at 300
tokens and 1.93x at 2000**. A term that scaled per *emitted token* would leave
that ratio flat, since both paths emit the same tokens. A ratio widening toward
~2 as length grows is the signature of a cost paid per batch-engine *step*.

---

## 3. INFERRED FROM CODE — what the two engines actually do per token

### 3.1 The runner wrapper is exonerated

MTP runs through the **identical** wrapper: `mtp_responses` is swapped in for
`stream_generate` *inside* `mlx_generate`
(`engines/mlx/generator/generate.py:912`), so it still goes through
`SequentialGenerator.step()` → `runner.handle_generation_tasks` → one
`event_sender.send` IPC per token → the `apply_all_parsers` chain — and it
sustains 22.3 tok/s. The sequential wrapper therefore costs at most ~45 ms/token
end to end, and stock sequential's 200 ms/token deficit lives **inside**
`stream_generate`/`generate_step`, not around it.

This kills the brief's leading suspect (`on_generation_token` →
`agree_on_cancellations`/`agree_on_tasks` per token) twice over — see 3.2.

### 3.2 Collectives per emitted token, counted

The brief asked whether the agree-on-* callbacks are blocking CPU collectives
per token. They are not, and the count runs the *wrong way*:

- **`SequentialGenerator`** — `on_generation_token` is throttled by
  `check_for_cancel_every` (`batch_generator.py:246-256`). `agree_on_tasks` is
  called from `step()` only when `_active is None`, i.e. when idle. Steady-state
  cost: ~3 collectives per 100 tokens = **0.03/token**, and they use
  `stream=mx.default_stream(mx.Device(mx.cpu))` (`utils_mlx.py:1014-1032`), so
  they are genuinely off the GPU.
- **`BatchGenerator`** — `step()` opens with
  `if not self._queue: self.agree_on_tasks()` (`batch_generator.py:387-388`).
  `_queue` is empty throughout steady decode, so this fires **every step**:
  one `mx.distributed.all_gather(mx.array([n_tasks]))` plus a `.tolist()` that
  forces a sync. Cost: **1.0/token**. And unlike `mx_any`/`mx_barrier`,
  `mx_all_gather_tasks` passes **no `stream=` argument**
  (`utils_mlx.py:1087-1090`), so this one lands on the **default GPU stream**
  while the model runs on mlx-lm's `generation_stream`.
- **MTP loop** — two `coord.broadcast` per step (B1 `[t1,d2]`, B2 `[ok,t2]`,
  `mtp/loop.py`), i.e. **1.0/token**, plus the same throttled 0.03/token from
  the shared wrapper.

**The batch engine pays ~33x more per-token collectives than sequential and is
3.9x faster.** Collectives are not the sequential pathology. They remain a
suspect for the *batch* pathology, but as a serialisation hazard rather than a
size cost — see §5.

### 3.3 Two real defects found on the way, neither of which is tonight's bug

**(i) `warmup_inference` divides by the wrong bound.**
`engines/mlx/generator/generate.py:497-499`:

```python
check_for_cancel_every = min(
    math.ceil(tokens_generated / min(time.monotonic() - t, 0.001)), 100
)
```

`min(elapsed, 0.001)` should be `max(elapsed, 0.001)` — the 0.001 is plainly a
divide-by-zero guard. As written the divisor is *always* 0.001, so the
expression is always `min(50000, 100) = 100`. The tokens/sec-adaptive intent is
dead; the value is a constant. Benign today (100 is a fine number, and it is
what makes 3.1's exoneration hold) but it is a silent constant masquerading as a
measurement, and it becomes wrong the moment anyone raises the clamp.

**(ii) The `opt_batch_gen` patch dropped upstream's bounded token context.**
Upstream `GenerationBatch._step` feeds logits processors from a rolling buffer:

```python
token_context = [tc.update_and_fetch(inputs[i:i+1]) for i, tc in enumerate(self._token_context)]
```

exo's `_patched_step` replaced that with the full, growing list
(`patches/opt_batch_gen.py:73`):

```python
sample_logits = processor(mx.array(self.tokens[e]), sample_logits)
```

That is a genuine O(N) regression on the decode critical path. F1 says it costs
15 µs at N=2000, so it is **not** the collapse — but it is free to fix and there
is no reason to carry it.

**(iii) GLM pays for a no-op penalty.** The 2.7bpw card sets
`repetition_penalty = 1.0`. `make_logits_processors` installs a penalty whenever
`penalty is not None and penalty != 0` (`sample_utils.py:122`), so 1.0 gets
installed — and `make_repetition_penalty` at 1.0 multiplies and divides by one.
GLM therefore runs a gather + `mx.where` + scatter over a 20-token window, on
*both* engines, every token, to change nothing. F3 prices it at ~0.28 ms/token —
small, but it is also what keeps (ii) and `generate_step`'s `mx.concat`
accumulator (F2) alive at all. Setting the card to `repetition_penalty = null`
removes all three at once.

### 3.4 The pipeline forces hard syncs inside the model call

`PipelineFirstLayer.__call__` (`auto_parallel.py:147-153`) does
`mx.eval(x)` → `recv_like` → `mx.eval(x)`. `PipelineLastLayer.__call__`
(`:173-207`) does `mx.eval(output)`, then on a non-last rank `send` +
`mx.depends` + `mx.eval(output)` + `mx.eval(_cache.keys)`, then in decode
`all_gather(output)` + `mx.eval(output)`. Per forward: **6 blocking evals on
rank 0, 5 on rank 1**, plus a send/recv and an all_gather.

These sit *inside* `model.__call__`, so they run during Python graph
construction. That means `generate_step`'s one-step lookahead —
`next_y, next_logprobs = _step(y); mx.async_eval(next_y, next_logprobs)` —
**cannot overlap anything**: `_step` blocks internally on the pipeline before it
ever returns. Neither engine actually pipelines across tokens on a multi-rank
pipeline; both are fully serialised per forward.

This is real and worth fixing, but note it is **symmetric** — the batch engine's
`_patched_step` calls the same `self.model(...)` and eats the same 5–6 evals. It
does not explain the sequential/batch gap.

---

## 4. NOT ESTABLISHED — Q1, sequential's flat 4x

I could not close this from code, and I am not going to invent a mechanism for
it. What is known:

- it is **not** the runner wrapper (§3.1),
- it is **not** the agree-on-* collectives (§3.2),
- it is **not** KV quantisation — `kv_bits_for` returns `None` for GLM
  (`KV_BITS_BY_FAMILY` holds only `deepseek-v4`), so `maybe_quantize_kv_cache`
  returns on its first line,
- it is **not** `wired_limit`, which I suspected as an engine asymmetry and then
  falsified: `stream_generate` sets it per request, but mlx-lm's
  `BatchGenerator.__init__` sets **the same** `max_recommended_working_set_size`
  once at construction (`generate.py:1576-1580`). Both paths decode with it set.
  (M3 reports `max_recommended_working_set_size = 90194313216`, memory 103079215104.)
- it is **not** the detokenizer (F6), the token accumulator (F2), or the
  penalty processor (F3).

Surviving candidates, in the order I would test them:

1. **Cross-rank host-work serialisation.** Because the pipeline blocks inside
   `model.__call__` (§3.4), per-token host work on rank 0 and rank 1 *adds*
   rather than overlapping — the far rank sits inside `recv`. MTP pays this once
   per two tokens; sequential pays it every token. This is the only candidate
   that is both consistent with §3.1 and length-independent.
2. **`mx.clear_cache()` cadence.** `generate_step` clears every 256 tokens
   (`generate.py:469`), `BatchGenerator._next` every 512. With GLM's buffer pool
   this is not obviously cheap.
3. **Stream crossing.** `y.item()` in `generate_step` syncs from the default
   stream onto an array produced on `generation_stream`.

**The discriminating experiment is one line of config, not code**: set
`repetition_penalty = null` in the GLM card and re-run `EXO_NO_BATCH=1` at 300
tokens. That removes the only exo-specific per-token work inside
`generate_step`. If the number does not move — and F2/F3 predict it will move by
about 0.5 ms/token, i.e. not at all — then the deficit is candidate 1 and the
fix is structural.

---

## 5. Q2, the batch length collapse — where the search now points

§2 proves the growth term is not the forward. §1 proves it is not host-side
bookkeeping in the response loop. What is left is work the batch engine does per
*step* that the MTP path does not, whose cost is not fixed:

**Leading candidate: the per-step `agree_on_tasks()` all_gather.** Per §3.2 this
is a distributed collective issued on the **default GPU stream** every single
token, immediately followed by a `.tolist()` that blocks on it — while the model
occupies a different stream. Its payload is one int and never grows, so this is
not a size argument; it is a *coupling* argument. A collective that must drain
the default stream before it can complete will cost more as the amount of
outstanding GPU work grows, and the outstanding work grows with the KV. This is
testable and it is the one thing in the batch path with the right shape.

Second: the prefix-cache pool (`_save_prefix_cache` deepcopies the full KV per
request) inflates resident memory in the batch engine, while the MTP path forces
`kv_prefix_cache = None` (`generate.py:754`). Note `use_prefix_cache` defaults to
`False` (`types/text_generation.py:123`), so this is only live if the bench
harness opts in — check before spending time on it.

**The definitive diagnostic already exists in the tree and needs no new code.**
`ExoBatchGenerator.step` (`batch_generate.py:478-484`) already logs, every 64
steps:

```
step overhead: {overhead}ms (next={next_elapsed}ms total={step_elapsed}ms)
```

`next=` is time inside `_mlx_gen.next()`; `overhead` is exo's response loop.
Run 2000 tokens on the batch engine at DEBUG and read the two columns as
length grows. If `next=` grows, the collapse is inside mlx-lm's batch step
(candidate 1, since §2 already excluded the forward). If `overhead` grows,
§1 was measured wrong and I want to know. This is read-only, costs one run, and
settles Q2 outright. **Do this before writing any patch.**

---

## 6. Recommendations

### (a) Sequential stock overhead — do not patch yet

Mechanism unestablished (§4). Two things are free and worth doing regardless:

- set `repetition_penalty = null` on the GLM 2.7/3.1/3.6bpw cards — removes the
  no-op penalty, the `mx.concat` accumulator and the `opt_batch_gen` array
  rebuild in one edit (**15 min**), and doubles as the §4 discriminating
  experiment;
- fix `min` → `max` in `warmup_inference` (**5 min**, one character, but re-check
  the clamp: today's constant 100 is load-bearing for §3.1's argument, so change
  it deliberately or not at all).

Then measure. Budget **0.5 day** of instrumented sequential runs before
proposing a real fix. If it lands on candidate 1, the fix is to stop blocking
inside `model.__call__` — deferring `PipelineLastLayer`'s evals the way
`_pending_prefill_sends` already does for prefill — which is a **3–5 day** job
with real deadlock risk and needs its own design note.

### (b) Batch collapse — one read-only run, then ~2 hours

1. DEBUG run reading `step overhead` (**30 min**, read-only, no code).
2. If `next=` is flat and `overhead` grows: §1 is wrong, re-measure.
3. If `next=` grows: hoist `agree_on_tasks()` out of the per-step path. It
   already has the right pattern next to it — gate it on the same
   `tokens_since_cancel_check` counter `on_generation_token` uses, so it fires
   every ~100 steps instead of every step, and pass
   `stream=mx.default_stream(mx.Device(mx.cpu))` into `mx_all_gather_tasks` to
   match `mx_any`/`mx_barrier`. **~15 lines, 1–2 hours**, low risk — the
   ordering guarantee that prevents TP collective deadlock is preserved because
   every rank still runs the same counter.
4. Restore upstream's bounded token context in `opt_batch_gen._patched_step`
   (**~10 lines, 1 hour**). Not the collapse (F1), but it is a straight
   regression against upstream and should not be carried.

### (c) Retiring `EXO_NO_BATCH=1` — integrate MTP, do not fix sequential

The framing in the brief has an answer that falls out of §2, and it is not the
obvious one.

Fixing sequential's overhead **does not retire the flag**. MTP lives only on the
sequential path (`mlx_generate` swaps `mtp_responses` in for `stream_generate`),
so `EXO_NO_BATCH=1` would still be required to get MTP at all. Perfecting
sequential gets it to *batch's* numbers — 19.5/5.7 — which is strictly worse
than what we already have with the flag set (22.3/11.0). It spends days to lose
the win.

MTP's advantage is **0.557 pipeline forwards per emitted token vs 1.000**. That
ratio is a property of speculative decoding, not of the engine, and §2 shows it
is the one term that survives at 2000 tokens — the gap *widens* from 1.14x to
1.93x with length. It is worth carrying into the batch engine.

Full MTP-in-batch is expensive: the loop assumes batch depth 1 throughout — one
`emitted` list, one cache, `snapshot`/`restore` over that cache, and a
verdict broadcast that is deliberately unconditional and lockstep across ranks
(the design note is explicit that a verdict-dependent collective count
deadlocks). Generalising to continuous batching means per-sequence accept/reject
with per-sequence rollback under a shared collective schedule. **2–3 weeks**,
and it should not start until (b) is done, because a collapsing batch engine
would hide the win.

**The cheap route, and my recommendation: dispatch to the MTP sequential loop
when batch depth is 1.** `BatchGenerator` already knows its occupancy
(`_active_tasks`, `EXO_MAX_CONCURRENT_REQUESTS`). A single-request path that
routes to `mtp_responses` and falls back to the batch loop the moment a second
task arrives retires the env var for the interactive single-request case — which
is the overwhelming majority of traffic on a 2-node box — without touching the
batch decode math or the collective schedule. **2–3 days.** It also makes
MTP-in-batch (the 2–3 week job) optional rather than blocking, and gives a
clean A/B on one build.

Order: **(b) step 1 tonight's leftovers → (a) card edit + measure → (b) fix →
(c) depth-1 dispatch.**

---

## Appendix: what I did not do

Read-only discipline held: nothing was started, stopped or placed via the exo
API, no supervisor or `~/.exo` was touched, and no code on the decode paths was
modified or committed. All microbenchmarks ran in-process on M3 with no model
weights loaded. This document is the only artifact.
