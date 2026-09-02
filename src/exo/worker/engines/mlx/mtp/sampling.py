"""Sampling for speculative decoding: the same distribution mlx-lm would
sample from, plus exact rejection sampling with residual correction.

The samplers are mlx-lm's (`mlx_lm.sample_utils`, Apache-2.0). We do not
reimplement top-p/top-k/min-p/XTC or the penalties — we adapt the *shape* of
mlx-lm's sampler, which returns a token, into one that also returns the
normalized distribution it sampled from, because speculative verification
needs the probabilities and not just the draw.

Filter order matches `make_sampler` exactly: the filters run on unscaled
logprobs and the temperature is applied at the categorical draw. So for the
same parameters a token from `Distribution.sample` is drawn from the same
distribution `mlx_lm`'s sampler would have used.

Correction (Leviathan et al. 2023, "Fast Inference from Transformers via
Speculative Decoding"; Chen et al. 2023): given a draft x ~ q and the target p,

    accept x with probability min(1, p(x)/q(x));
    otherwise draw from the normalized residual max(p - q, 0).

The resulting draw is distributed exactly as p, for ANY q. That is what makes
a bad draft cost speed and never quality — the same guarantee greedy decoding
gets from `argmax(p) == x`, which is this rule's zero-temperature limit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import mlx.core as mx
from mlx_lm.sample_utils import (
    apply_min_p,
    apply_top_k,
    apply_top_p,
    apply_xtc,
    make_logits_processors,
)

__all__ = ["Distribution", "make_distribution", "make_logits_processors",
           "rejection_correct", "acceptance_profile"]


@dataclass(frozen=True)
class Distribution:
    """One step's distribution over the vocabulary.

    `probs` is normalized and is what verification compares. `logits` is the
    temperature-scaled, filter-masked array the draw comes from, kept so the
    draw is bit-for-bit mlx-lm's."""
    probs: mx.array          # [B, V], sums to 1
    logits: mx.array         # [B, V], scaled + masked

    def sample(self) -> mx.array:
        return mx.random.categorical(self.logits)

    def argmax(self) -> mx.array:
        return mx.argmax(self.logits, axis=-1)


def make_distribution(
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = 0,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: List[int] = [],
) -> Optional[Callable[[mx.array], Distribution]]:
    """logits [B, V] -> Distribution, or None for greedy (temp == 0).

    None is not a fallback: at temp 0 the target is a point mass, rejection
    sampling degenerates to "accept iff the draft equals the argmax", and the
    loop takes that cheaper branch. Building a one-hot vector to rediscover
    the same rule would only add a softmax per token."""
    if temp == 0:
        return None

    methods = []
    if 0 < top_p < 1.0:
        methods.append(lambda x: apply_top_p(x, top_p))
    if min_p != 0.0:
        methods.append(lambda x: apply_min_p(x, min_p, min_tokens_to_keep))
    if xtc_probability > 0.0:
        methods.append(lambda x: apply_xtc(x, xtc_probability, xtc_threshold,
                                           xtc_special_tokens))
    if top_k > 0:
        methods.append(lambda x: apply_top_k(x, top_k))

    inv_temp = 1.0 / temp

    def distribution(logits: mx.array) -> Distribution:
        lp = logits.astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        for method in methods:
            lp = method(lp)
        scaled = lp * inv_temp
        return Distribution(probs=mx.softmax(scaled, axis=-1), logits=scaled)

    return distribution


def rejection_correct(p: mx.array, q: mx.array, draft: mx.array):
    """Verify one drafted token against the target distribution.

    p, q: [B, V] normalized. draft: [B] the token drawn from q.
    Returns (accepted: bool array [B], token: [B]).

    On rejection the replacement comes from the normalized residual
    max(p - q, 0), which is what makes the pair (accept-or-residual) exactly
    p-distributed. Sampling the replacement from p itself — the tempting
    simplification — over-weights tokens the draft already had a chance to
    propose, and biases the output.
    """
    idx = draft[:, None]
    p_d = mx.take_along_axis(p, idx, axis=-1)[:, 0]
    q_d = mx.take_along_axis(q, idx, axis=-1)[:, 0]
    # q_d == 0 can only arise from underflow in a token q itself produced;
    # treat it as certainly acceptable rather than dividing by zero.
    ratio = mx.where(q_d > 0, p_d / mx.maximum(q_d, 1e-30), 1.0)
    accepted = mx.random.uniform(shape=p_d.shape) < ratio

    residual = mx.maximum(p - q, 0.0)
    total = residual.sum(axis=-1, keepdims=True)
    # total is 0 only if q dominates p everywhere, which forces ratio >= 1 and
    # acceptance; the fallback keeps the draw well-defined regardless.
    residual = mx.where(total > 0, residual / mx.maximum(total, 1e-30), p)
    replacement = mx.random.categorical(mx.log(residual))

    return accepted, mx.where(accepted, draft, replacement)


def acceptance_profile(p: mx.array, q: mx.array):
    """The exact per-token outcome probabilities of `rejection_correct`,
    computed in closed form: (accept_prob, resulting_distribution).

    Used as a test oracle — the resulting distribution must equal p for any q,
    which is the whole claim — and useful for reporting the acceptance rate a
    given draft head earns at a given temperature without sampling it."""
    a = mx.minimum(p, q)                      # P(draw x AND accept x)
    accept = a.sum(axis=-1)
    residual = mx.maximum(p - q, 0.0)
    total = residual.sum(axis=-1, keepdims=True)
    norm = mx.where(total > 0, residual / mx.maximum(total, 1e-30), p)
    return accept, a + (1.0 - accept)[:, None] * norm
