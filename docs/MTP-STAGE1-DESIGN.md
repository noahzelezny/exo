# MTP stage 1 — speculative decoding over pipeline-sharded instances

Status: implemented, default-off behind `EXO_MTP=1`, unit-tested, **not yet
smoke-tested on a cluster**. Stage 0 (single node) is unchanged in behaviour.

---

## The finding that shaped the design

The brief anticipated extending "whatever channel coordinates pipeline steps"
with a rollback message. There is no such channel.

exo's pipeline parallelism is **SPMD inside the MLX graph**. There is no
orchestrator and no per-step envelope. `pipeline_auto_parallel` slices
`inner_model.layers` to the shard's range and wraps the two ends
(`auto_parallel.py`):

- `PipelineFirstLayer.__call__` throws away its input and takes
  `mx.distributed.recv_like(x, r-1)` instead;
- `PipelineLastLayer.__call__` runs the layer and `mx.distributed.send`s the
  result to `r+1`.

The hidden state travels as a **bare `mx.array`** — no header, no tag, no
sequence number. Ranks stay in lockstep purely by *running the same Python*.
Every rank executes the same decode loop and issues the same number of
forwards in the same order; `recv_like` matches by shape and arrival order.

Three consequences fall straight out, and they are most of this design:

1. **The verify forward needs no new collective.** A T-wide verification is
   literally `model(two_tokens, cache=cache)` — the same call shape prefill
   already makes. Requirement 2 is satisfied by doing nothing.
2. **Rollback needs no rollback *message*, only an agreed *verdict*.**
   `mtp/caches.py:restore` already trims each cache by *its own* measured
   offset delta. Give every rank the same accept/reject bit and a shard
   holding 20 layers and a shard holding 4 both roll back correctly, with no
   count in the message at all.
3. **Only two things break the "same code, same result" symmetry**, and the
   whole control plane exists to repair exactly those two. They are below.

Two further facts, both verified in the tree rather than assumed:

- **Every node downloads the full repo.** `ResumableShardDownloader.ensure_shard`
  applies no per-layer weight filter, so `embed_tokens`, `lm_head` and the
  `mtp-head-q6.safetensors` sidecar are present on every node. The brief's
  contingency — "plan a small duplication at instance-load time if embeddings
  aren't on the last shard" — is **not needed for any family**. Only
  `layers` is sliced; the outer `embed → layers → norm → lm_head` module is
  intact on every rank. (Rank > 0 even *runs* `embed_tokens`, then discards
  the result in `PipelineFirstLayer`.)
- **During decode, every rank already has the final hidden state.**
  `PipelineLastLayer` ends a decode forward with
  `all_gather(output)[-output.shape[0]:]`, a broadcast-from-last idiom. This
  is why every rank computes identical logits today.

---

## Where the head lives, and why

**On the last rank only.**

The tempting alternative — replicate the head on every rank, since decode-time
`all_gather` puts the final hidden state everywhere — fails at **prefill**.
`set_pipeline_prefill(model, is_prefill=True)` gates that `all_gather` *off*
(correctly: only the last rank's logits matter before decode starts). So
during prompt processing, an intermediate rank's capture point sees its own
shard's intermediate activation, not the trunk's final one. Seeding a head
from that would not fail loudly — it would draft noise and quietly halve
acceptance.

Forcing the gather on during prefill would fix it and cost an `[1, chunk,
hidden]` all_gather per chunk — tens of MB per chunk, per rank. Broadcasting
one int per step instead is four orders of magnitude cheaper.

So the head loads on the last rank, is seeded there, and drafts there. On
every other rank `plan.head is None` and **stage 1 costs those nodes no extra
memory at all**.

### The second asymmetry: sampling

Because the head samples the draft token, the last rank's `mx.random` stream
advances past its peers'. `mlx_generate` seeds all ranks identically
(`mx.random.seed(seed)`), which is what makes independent per-rank sampling
work on the stock path — but one extra draw on one rank destroys that forever.

Therefore: **once the last rank samples anything for a step, it samples
everything for that step and broadcasts the results.** Non-last ranks never
call the sampler after prefill. This is not an optimization; a design that
let non-last ranks sample "the easy ones" would diverge on token two.

---

## Verify data flow

Per speculative step, T = 2 (draft one token, verify a pair):

```
last rank      head.draft_logits(h, ids, dcache) -> d2
ALL ranks      B1: broadcast [t1, d2] from last rank
ALL ranks      lg2 = model([t1, d2], cache)     <-- the verify forward
                 rank0 embeds -> layers -> send
                 rank1 recv   -> layers -> all_gather
                 every rank ends holding the same lg2
last rank      verdict: greedy `d2 == argmax`, or exact rejection sampling
                 with residual correction (sampling.rejection_correct)
ALL ranks      B2: broadcast [ok, t2] from last rank
ALL ranks      if not ok: restore(cache, csnap); replay model([t1, t2])
ALL ranks      emit t1, t2
last rank      head.advance over the two committed positions; sample t_next
                 (t_next rides the NEXT step's B1)
```

The verify forward is the existing distributed pipeline, unmodified. No new
collective ops, no new code path in `auto_parallel.py`.

---

## The rollback protocol

### Exact messages

Both are `mx.distributed.all_gather(v)[-n:]` — broadcast-from-last — over the
**same group** the hidden-state send/recv uses, on the **CPU stream** (matching
`mx_barrier` / `mx_any` in `utils_mlx.py`: these are a handful of ints and have
no business under the GPU watchdog timeout).

| msg | payload | dtype | when |
|-----|---------|-------|------|
| B1  | `[t1, d2]` | int32, shape (2,) | top of every step, before the verify forward |
| B2  | `[ok, t2]` | int32, shape (2,) | after the verify forward, before any rollback |

`B1` carries **both** tokens because rank 0 must *embed* them: the verify
forward is `model([t1, d2])` on every rank, and a rank that guessed either id
would compute a different pipeline than the one whose logits decide the
verdict. `t1` rides B1 rather than the previous step's B2 because it is only
known *after* that step's rollback-and-replay.

### Ordering and failure behaviour

- **Unconditional and fixed.** B1 then B2, every step, every rank, accepted or
  rejected. A verdict-dependent number of collectives would deadlock the first
  time two ranks disagreed — which is precisely the failure being guarded
  against.
- **No new failure mode.** Both broadcasts run on the group that already
  carries the hidden states. A dead or wedged rank hangs or raises here exactly
  as it would in `PipelineFirstLayer.recv_like`. Stage 1 adds no recovery path;
  the runner's existing instance-level supervision is still what notices.
- **Rollback is derived, not transmitted.** No rank sends a trim count. Each
  rank calls `restore(cache, csnap)`, which trims by that cache's own offset
  delta and raises if a cache moved backwards. Shards with different layer
  counts roll back correctly from one shared bit.

### One more agreement: whether to draft at all

`plan_mtp` ends with a **once-per-request** broadcast of the last rank's
"did the head load?" bit. Head loading can succeed on one node and fail on
another — a half-downloaded sidecar, a stale registry entry against a
differently-pinned mlx-lm — and ranks that disagreed about whether the request
is speculative would run different forward counts and deadlock in `recv_like`.
The last rank's answer is the instance's answer.

Everything checked *before* that point is uniform across ranks by construction
(same task, same model object), with **one operator-owned exception**:
`EXO_MTP` must be set identically on every node of the instance. Setting it on
some nodes only is a misconfiguration — and it is a **hang**, not a wrong
answer. Stage 0 shares this property; it is called out here because stage 1
makes it reachable.

---

## Boundary cases

- **All accepted.** Two tokens per step for one trunk forward. No trim. Cache
  offset advances exactly 2 per step on every rank.
- **All rejected.** The trunk's own `t+2` came out of the *same* verify
  forward, so the step still emits two tokens — a rejection costs one extra
  forward, never a wrong token. Every cache over-advances by 2 and trims back
  by 2 before the replay. The replay forward happens on **every** rank; if only
  the drafting rank replayed, the pipeline would wedge on the next send.
  (`test_rejection_costs_an_extra_forward_on_every_shard` counts forwards on
  both shards for exactly this reason.)
- **EOS inside a drafted pair.** Termination is derived from the emitted
  tokens, which the broadcasts make bit-identical on every rank, so both ranks
  break out of the loop at the same iteration with the same collective count.
  No stop message is needed. If `t1` is EOS, `t2` is not emitted; the caches
  hold a position past the end, which is harmless because the request is over.
- **Max tokens inside a pair.** Same mechanism — `len(emitted) >= max_tokens`
  is evaluated on identical `emitted` lists.

---

## What is NOT bit-identical, and why that is not the gate

Stage 0 already documents this and stage 1 inherits it verbatim: MLX's chunked
and single-token kernels disagree at genuine near-ties, and verification always
happens inside a 2-token forward. Every correct speculative implementation on
this runtime inherits that. The gate is "divergence confined to near-ties".

Stage 1 adds one more source in principle — the last rank's logits are computed
from an `all_gather`ed hidden state — but in practice this changes nothing,
because the stock multi-node decode path computes its logits the same way.

---

## The topology gate

`plan_mtp` (speculative.py) replaces `maybe_mtp_head`:

| topology | outcome |
|---|---|
| single node (no group, or `world_size == 1`) | **stage 0** — `LocalCoordinator`, loop unchanged |
| multi-node **and** pipeline-sharded | **stage 1** |
| multi-node, not pipeline-sharded (tensor) | **refused**, stock decode |
| vision request, or unregistered family, or missing sidecar | refused |
| `EXO_MTP` unset | refused |

The pipeline test asks the **model object**, not the placement metadata:
`is_pipeline_model` looks for the wrapper layers `pipeline_auto_parallel`
installs. That is deliberate — the property stage 1 depends on *is* the
`all_gather` inside `PipelineLastLayer`, and the wrappers **are** that
property. Metadata would only say how the instance was *meant* to be built.
`generate.py`'s pre-existing `_has_pipeline_communication_layer` now delegates
to the same function, so the two cannot drift.

---

## Structure: one loop, not two

`mtp_stream_generate` is **not** forked. It takes a `Coordinator`
(`mtp/pipeline.py`) that defaults to `LocalCoordinator`, whose `is_last` is
`True` and whose `broadcast` is the identity function. With that default the
loop reduces line-for-line to what stage 0 ran. `PipelineCoordinator` makes the
same loop correct on every pipeline rank.

This matters for maintenance: `loop.py` is vendored from `vqlab/mtp/` and the
package docstring says to fix bugs *there* first and re-vendor. A forked
pipeline loop would have made that impossible. The Coordinator seam is now the
one documented exception to verbatim vendoring, called out in
`mtp/__init__.py`.

---

## Explicitly out of scope

- **Tensor sharding.** Still refused, and not by an oversight that could be
  relaxed. Tensor ranks each hold a *slice of every layer*, so no rank ever
  holds a whole final hidden state to draft from. Supporting it needs a
  tensor-parallel head, not a wider gate. (Separately, qwen4_exp has no tensor
  strategy in this fork at all.)
- **The batch engine.** `BatchGenerator` / `MlxBatchGenerator` is untouched;
  MTP remains single-sequence, on the `SequentialGenerator` path only.
- **KV prefix pool reuse.** Stage 1 inherits stage 0's trade: an MTP request
  bypasses the pool, because the head must be seeded from position 0. Pairing a
  head cache with each pooled prefix is still the stage-0.5 design.
- **Overlapped pipeline prefill.** Stage 1's prompt processing uses the plain
  chunked path, not `pipeline_parallel_prefill`'s software-pipelined bubble
  schedule, because the head must observe every chunk's hidden state in order
  on the last rank. Prefill is therefore *correct but not bubble-optimized* on
  a pipeline instance. On long prompts this is a real cost and is the first
  thing to measure after the smoke passes.
- **Fusing B1 and B2 into one collective per step.** Possible (B2 of step k and
  B1 of step k+1 are adjacent), deliberately not done — two obvious broadcasts
  are worth more than one clever one until the smoke passes.
- **Head-cache windowing / reset on long generations.** Inherited stage-0 gap;
  acceptance-only, never correctness.

---

## Tests

`src/exo/worker/tests/unittests/test_mlx/test_mtp_pipeline.py` (22 tests, no
cluster, no model download):

- **Fake 2-shard pipeline** — two `ToyModel` shards with **different cache
  counts** (3 and 5, so a fixed-count trim cannot pass), running the *real*
  `mtp_stream_generate` in two threads, exchanging real control broadcasts
  through a `threading.Barrier` with a timeout. A rank that ran a different
  number of collectives fails as a **timeout**, not an infinite hang.
  Parameterized over all-accept / all-reject / alternating / mixed.
  - The assertion with teeth is the offset identity `prompt + 2 * steps`: a
    wrong trim shows up as drift that grows with step count.
  - The broadcast payload crosses the thread boundary as Python ints — forced,
    because MLX streams are thread-local, and also more faithful to a wire.
- **Speculation changes the schedule, never the text** — accept-everything and
  reject-everything runs emit identical tokens, equal to the toy trunk's own
  greedy continuation.
- **Pipeline output == single-node output** — the cluster smoke's claim in
  miniature, so a real-cluster failure points at the model or the transport,
  not at the protocol.
- **EOS inside a drafted pair** stops both ranks together.
- **Stage 0 default path unchanged** — no coordinator argument still works.
- **Gate selection** — single-node → stage 0, pipeline → stage 1, tensor → off,
  vision → off, `EXO_MTP` unset → off, head-load failure → off.
- **Coordinator units** — last-rank identification, single-rank collapse, and
  refusal of >1-D payloads (a hidden state must never ride the control path).

Full local suite: **526 passed, 3 skipped**. `ruff` clean.

---

## What the cluster smoke must exercise

Not run here, by instruction. The orchestrator should check:

1. **2-node Flash pipeline, `EXO_MTP=1` on BOTH nodes**, greedy
   (`temperature=0`), a prompt long enough for several hundred decode steps.
2. **Compare the generated text against single-node stage 0** on the same
   prompt and seed. Expect equality or divergence confined to near-ties (see
   above) — *not* bitwise equality as a hard gate.
3. **Watch acceptance**, logged per request as `MTP decode done: … acceptance
   X`. It should land near the single-node number for the same model
   (~0.75–0.82 measured for q6 heads). A collapse toward 0 means the head is
   being seeded from the wrong activation — i.e. the "head on last rank" claim
   is wrong for that family.
4. **Confirm tok/s is not *worse* than stock 2-node decode.** Two extra
   collectives per step are tiny, but the un-bubbled prefill is not; measure
   prefill and decode separately.
5. **Kill a node mid-generation** and confirm the failure looks like an
   ordinary pipeline failure (supervision notices), not a silent wrong answer.
6. **Negative case:** a tensor-sharded 2-node instance with `EXO_MTP=1` must
   log the refusal and decode normally.

---

## Known risks

1. **Head-on-last-rank assumes the capture point sees the true final
   activation on that rank.** Verified by reading `PipelineLastLayer` — the
   last rank's own output *is* the end of the pipeline — but not yet observed
   on hardware. This is what risk 3 in the smoke list detects, and it fails
   *quietly* (as low acceptance), which is why acceptance must actually be
   read, not assumed.
2. **Un-bubbled prefill on pipeline instances.** Correct, possibly slow on long
   prompts. Could make MTP a net loss for long-context agent loops even at good
   acceptance. Measure before defaulting anything on.
3. **`EXO_MTP` must be uniform across nodes.** Non-uniform is a hang. Worth a
   startup-time agreement check if stage 1 ever moves toward default-on.
4. **Family coverage — check which "Flash" before booking the smoke.**
   Registered families are exactly `qwen4_exp`, `qwen3_5`, `qwen3_5_moe`.
   - **Qwen3.8-Flash-Next** resolves to `qwen3_5` / `qwen3_5_moe` — registered,
     and the right choice for the first smoke.
   - **GLM-5.3-Flash** resolves to `glm5_next`, which is **not registered**
     (GLM-5.3 ships an MTP layer, but there is no head module for it here).
     That instance will take the refusal path and decode normally — a correct
     outcome, but it smoke-tests nothing.
   Either way the run also needs `mtp-head-q6.safetensors` present in the
   artifact dir, or the gate refuses at the sidecar check.
5. **Non-last ranks allocate an unused draft cache.** `spec.make_draft_cache`
   still runs on every rank. It is empty and cheap, but it is dead state.
6. **Thread-based tests are a model of the cluster, not the cluster.** They
   prove collective *counts* and cache *arithmetic*. They cannot prove
   `mx.distributed.all_gather` on the CPU stream interleaves correctly with the
   graph's GPU-stream send/recv. That is the single most valuable thing the
   smoke adds.
