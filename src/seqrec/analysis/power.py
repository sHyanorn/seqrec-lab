"""Seed-count rule (METHODS 7.5, D018).

y is log validation NDCG@10 per run. With SE = sigma * sqrt(2/n) and df = 2n - 2, n is the
smallest integer from 5 to 10 at which both the strictest F1 directional test and the
TOST equivalence test reach power 0.8 at the SESOI; otherwise 10.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy import stats

DELTA = math.log(1.02)  # SESOI (METHODS 7.2)
ALPHA = 0.05
F1_TESTS = 24  # the strictest Holm step in F1 uses alpha / 24
N_RANGE = range(5, 11)
TARGET = 0.8
NEURAL_ARMS = ("sasrec-bce", "sasrec-ce", "sasrec-ss", "gsasrec", "bert4rec")
REPLACE_KEY = ("tag", "dataset", "split", "shuffle", "arm")

Exports = Path | Sequence[Path]


FINISHED = ("trained", "tested")


class IncompleteRerun(ValueError):
    """A later export holds a run that did not finish training."""


def _seeds(rows: list[dict]) -> str:
    return ", ".join(sorted((str(r.get("seed", "")) for r in rows), key=lambda x: (len(x), x)))


def read_exports(paths: Exports) -> list[dict]:
    """Rows of one or more exported batch CSVs. The rows of a later file replace the rows of
    earlier files with the same tag, dataset, split, shuffle and arm, so a rerun stands in
    for the runs it repeats (D035). Every run of a later file must have finished training,
    and a replacing group must hold the same seeds as the group it replaces: a failed or
    missing seed would otherwise shrink the group without notice (D040)."""
    merged: dict[tuple, list[dict]] = {}
    for i, path in enumerate([paths] if isinstance(paths, Path) else paths):
        groups = defaultdict(list)
        with Path(path).open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if i and row["status"] not in FINISHED:
                    raise IncompleteRerun(
                        f"{path}: {row.get('run_id', '?')} is {row['status']}; a later export "
                        "replaces earlier runs only when all its runs finished training"
                    )
                groups[tuple(row[k] for k in REPLACE_KEY)].append(row)
        for key, rows in groups.items():
            before, after = (_seeds(merged[key]), _seeds(rows)) if key in merged else ((), ())
            if before != after:
                raise IncompleteRerun(
                    f"{path}: {'/'.join(key)} replaces seeds {before} with {after}"
                )
        merged |= groups
    return [row for rows in merged.values() for row in rows]


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
    paths: Exports, tag: str, dataset: str, split: str = "loo"
) -> tuple[float, dict[str, list[float]]]:
    """Pooled SD of log validation NDCG@10 within arms, over the trained, unshuffled runs of
    one tag, dataset and split in one or more exports (`read_exports`)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for row in read_exports(paths):
        value = row.get("validation_ndcg@10")
        if (
            row["tag"] == tag
            and row["dataset"] == dataset
            and row["split"] == split
            and row["shuffle"] == "none"
            and row["arm"] in NEURAL_ARMS
            and row["status"] in FINISHED
            and value
        ):
            groups[row["arm"]].append(math.log(float(value)))
    sigma = pooled_sd([np.asarray(v) for v in groups.values()])
    return sigma, dict(groups)


def report(paths: Exports, tag: str, dataset: str) -> str:
    sigma, groups = sigma_from_export(paths, tag, dataset)
    n = seed_count(sigma)
    lines = [
        f"{arm}: {len(v)} runs, sd {np.std(v, ddof=1):.5f}" for arm, v in sorted(groups.items())
    ]
    lines.append(
        f"pooled sigma {sigma:.5f}; n = {n}; power at n: directional "
        f"{directional_power(sigma, n):.3f}, equivalence {equivalence_power(sigma, n):.3f}"
    )
    return "\n".join(lines)
