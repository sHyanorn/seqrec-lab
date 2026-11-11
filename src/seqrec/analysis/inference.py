"""The registered tests as functions of per-run values (METHODS 7.3, 7.4 and 8).

A value is one run's y = log NDCG@10 (7.1), or its time to target (8). Nothing here reads a
file or knows an arm; `seqrec.analysis.confirm` assembles the families.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from seqrec.analysis.power import ALPHA, DELTA

REJECTED, NOT_REJECTED, PENDING = "rejected", "not rejected", "pending"


@dataclass(frozen=True)
class Welch:
    diff: float  # mean(a) - mean(b)
    se: float
    df: float  # Welch-Satterthwaite


def welch(a: Sequence[float], b: Sequence[float]) -> Welch:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if min(len(a), len(b)) < 2:
        raise ValueError("a Welch test needs two runs or more per arm")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb == 0:
        raise ValueError("no variance in either arm: the Welch statistic is undefined")
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    return Welch(float(a.mean() - b.mean()), math.sqrt(va + vb), float(df))


def p_two_sided(a: Sequence[float], b: Sequence[float]) -> float:
    """Directional test: H0 mu_a = mu_b."""
    w = welch(a, b)
    return float(2 * stats.t.sf(abs(w.diff) / w.se, w.df))


def p_above(a: Sequence[float], b: Sequence[float], margin: float) -> float:
    """One-sided test of H0: mu_a - mu_b <= margin. Non-inferiority has margin -delta,
    minimum effect +delta."""
    w = welch(a, b)
    return float(stats.t.sf((w.diff - margin) / w.se, w.df))


def p_tost(a: Sequence[float], b: Sequence[float], delta: float = DELTA) -> float:
    """Equivalence: the larger p of the one-sided tests against -delta and +delta."""
    return max(p_above(a, b, -delta), p_above(b, a, -delta))


def holm(pvalues: Mapping[str, float | None], alpha: float = ALPHA) -> dict[str, str]:
    """Holm's step-down procedure over a family, every registered test of it a key. None is a
    test whose runs do not exist yet, and its status is pending. A test with a p-value is

    - rejected, if Holm rejects it with every missing p-value set to 1;
    - not rejected, if Holm does not reject it with every missing p-value set to 0;
    - pending otherwise.

    Lowering another test's p-value never turns a rejection into a non-rejection, so the
    first two hold for every value the missing tests can take. A complete family gets
    ordinary Holm."""
    known = sorted((p, name) for name, p in pvalues.items() if p is not None)
    if any(not 0 <= p <= 1 for p, _ in known):  # also true for NaN
        raise ValueError("p-values must lie in [0, 1]")
    m, k = len(pvalues), len(known)
    status = dict.fromkeys(pvalues, PENDING)
    certain, possible = True, True
    for rank, (p, name) in enumerate(known):
        certain = certain and p <= alpha / (m - rank)  # the missing tests rank last
        possible = possible and p <= alpha / (k - rank)  # the missing tests rank first
        if certain:
            status[name] = REJECTED
        elif not possible:
            status[name] = NOT_REJECTED
    return status


def mann_whitney_less(x: Sequence[float], y: Sequence[float]) -> float:
    """Exact one-sided Mann-Whitney test that x is stochastically smaller than y (METHODS 8):
    the share of all splits of the pooled values into groups of these sizes whose rank sum
    for x is at most the observed one. A run censored at +inf ranks last; ties get
    mid-ranks."""
    pooled = np.concatenate([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    if not len(x) or not len(y) or np.isnan(pooled).any():
        raise ValueError("both groups need values, and none may be NaN")
    ranks = stats.rankdata(pooled)  # mid-ranks; +inf values tie with each other, last
    observed = ranks[: len(x)].sum()
    sums = np.fromiter(
        (sum(c) for c in itertools.combinations(ranks, len(x))),
        dtype=float,
        count=math.comb(len(pooled), len(x)),
    )
    return float(np.mean(sums <= observed + 1e-9))  # rank sums are multiples of one half
