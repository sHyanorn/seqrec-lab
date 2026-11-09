"""Seed-count rule (METHODS 7.5, D018).

y is log validation NDCG@10 per run. With SE = sigma * sqrt(2/n) and df = 2n - 2, n is the
smallest integer from 5 to 10 at which both the strictest F1 directional test and the
TOST equivalence test reach power 0.8 at the SESOI; otherwise 10.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

DELTA = math.log(1.02)  # SESOI (METHODS 7.2)
ALPHA = 0.05
F1_TESTS = 24  # the strictest Holm step in F1 uses alpha / 24
N_RANGE = range(5, 11)
TARGET = 0.8
NEURAL_ARMS = ("sasrec-bce", "sasrec-ce", "sasrec-ss", "gsasrec", "bert4rec")


def _se_df(sigma: float, n: int) -> tuple[float, int]:
    return sigma * math.sqrt(2 / n), 2 * n - 2


def directional_power(sigma: float, n: int, delta: float = DELTA) -> float:
    """P(|T| > t_{1-a/2, df}) with T noncentral t(df, delta/SE) and a = alpha / 24."""
    se, df = _se_df(sigma, n)
    crit = stats.t.ppf(1 - ALPHA / F1_TESTS / 2, df)
    nc = delta / se
    # P(T < -crit | nc) = P(T > crit | -nc); the sf form stays finite where nct.cdf of a far
    # tail returns NaN.
    return float(stats.nct.sf(crit, df, nc) + stats.nct.sf(crit, df, -nc))


def equivalence_power(sigma: float, n: int, delta: float = DELTA) -> float:
    """TOST power when the true difference is 0: max(0, 2 F_t(delta/SE - t_{0.95}) - 1)."""
    se, df = _se_df(sigma, n)
    return max(0.0, float(2 * stats.t.cdf(delta / se - stats.t.ppf(1 - ALPHA, df), df) - 1))


def seed_count(sigma: float) -> int:
    for n in N_RANGE:
        if directional_power(sigma, n) >= TARGET and equivalence_power(sigma, n) >= TARGET:
            return n
    return N_RANGE[-1]


def pooled_sd(groups: list[np.ndarray]) -> float:
    """Pooled within-group standard deviation."""
    dof = sum(len(g) - 1 for g in groups)
    if dof < 1:
        raise ValueError("need at least one group with two or more runs")
    return math.sqrt(sum((len(g) - 1) * np.var(g, ddof=1) for g in groups if len(g) > 1) / dof)


def sigma_from_export(
    path: Path, tag: str, dataset: str, split: str = "loo"
) -> tuple[float, dict[str, list[float]]]:
    """Pooled SD of log validation NDCG@10 within arms, over the trained, unshuffled runs of
    one tag, dataset and split in an export."""
    groups: dict[str, list[float]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = row.get("validation_ndcg@10")
            if (
                row["tag"] == tag
                and row["dataset"] == dataset
                and row["split"] == split
                and row["shuffle"] == "none"
                and row["arm"] in NEURAL_ARMS
                and row["status"] in ("trained", "tested")
                and value
            ):
                groups[row["arm"]].append(math.log(float(value)))
    sigma = pooled_sd([np.asarray(v) for v in groups.values()])
    return sigma, dict(groups)


def report(path: Path, tag: str, dataset: str) -> str:
    sigma, groups = sigma_from_export(path, tag, dataset)
    n = seed_count(sigma)
    lines = [
        f"{arm}: {len(v)} runs, sd {np.std(v, ddof=1):.5f}" for arm, v in sorted(groups.items())
    ]
    lines.append(
        f"pooled sigma {sigma:.5f}; n = {n}; power at n: directional "
        f"{directional_power(sigma, n):.3f}, equivalence {equivalence_power(sigma, n):.3f}"
    )
    return "\n".join(lines)
