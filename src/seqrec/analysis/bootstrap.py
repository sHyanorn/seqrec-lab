"""The two-level percentile bootstrap of METHODS 7.6, from the per-user ranks that a final
run keeps on the volume after its test read (D051). It reads no test data.

A resample draws, with replacement, each arm's runs from its own seeds, and the test users
from all of them. One draw of users serves every arm, so the arms stay paired by user. A run's
value is y = log of its mean NDCG@10 over the drawn users (METHODS 7.1), and a difference is
the mean y of arm a minus that of arm b over the drawn runs. The interval runs from the 2.5th
to the 97.5th percentile of 10,000 resamples. It is descriptive: the tests of METHODS 7.3
decide. With five to ten seeds the draw over seeds understates the variation between runs
(METHODS 7.6).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from seqrec.analysis import families
from seqrec.analysis.table import format_table
from seqrec.data.stats import write_csv

RESAMPLES = 10_000
CUTOFF = 10  # NDCG@10, the primary metric (METHODS 6)


def ndcg(ranks: np.ndarray) -> np.ndarray:
    """Each user's NDCG@10 from the rank of the target (METHODS 6)."""
    ranks = np.asarray(ranks, dtype=float)
    return np.where(ranks <= CUTOFF, 1.0 / np.log2(ranks + 1.0), 0.0)


def intervals(
    runs: Mapping[str, np.ndarray],
    pairs: Sequence[tuple[str, str]],
    resamples: int = RESAMPLES,
    seed: int | np.random.Generator = 0,
) -> list[tuple[float, float, float]]:
    """For each pair (a, b): the difference in mean y and its percentile interval. `runs`
    holds, per arm, each user's NDCG@10 with one row per run; every row lists the same users
    in the same order."""
    arms = sorted({arm for pair in pairs for arm in pair})
    n = runs[arms[0]].shape[1]
    rng = np.random.default_rng(seed)
    observed = {arm: np.log(runs[arm].mean(axis=1)).mean() for arm in arms}
    draws = np.empty((resamples, len(pairs)))
    for i in range(resamples):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        mean = {}
        for arm in arms:
            y = np.log(runs[arm] @ counts / n)
            mean[arm] = y[rng.integers(0, len(y), len(y))].mean()
        draws[i] = [mean[a] - mean[b] for a, b in pairs]
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return [
        (float(observed[a] - observed[b]), float(lo), float(hi))
        for (a, b), lo, hi in zip(pairs, low, high, strict=True)
    ]


def read_runs(batch_dir: Path) -> dict[tuple[str, str], np.ndarray]:
    """Per dataset and arm, each user's NDCG@10 in the tested final runs of a batch on
    leave-one-out with the order kept, one row per run in seed order. Every run of a dataset
    lists its test users in the order of the split (D051)."""
    found = defaultdict(dict)
    for path in batch_dir.glob("*/config.json"):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        kind = (cfg["phase"], cfg["split"], cfg["shuffle"], cfg["tag"])
        if kind == ("final", "loo", "none", "") and (path.parent / "test.json").exists():
            with np.load(path.parent / "test_ranks.npz") as ranks:
                found[cfg["dataset"], cfg["arm"]][cfg["seed"]] = ndcg(ranks["test"])
    return {key: np.stack([by_seed[s] for s in sorted(by_seed)]) for key, by_seed in found.items()}


def report(batch_dir: Path, out: Path, resamples: int = RESAMPLES) -> str:
    """The interval of every F1 and F2 difference (METHODS 7.4) of each dataset in a batch,
    written to `out` and returned as text."""
    runs = read_runs(batch_dir)
    rows = []
    for dataset in sorted({d for d, _ in runs}):
        compared = [c for f in ("F1", "F2") for c in families.family(f) if c.dataset == dataset]
        arms = {arm: runs[dataset, arm] for c in compared for arm in (c.a, c.b)}
        found = intervals(arms, [(c.a, c.b) for c in compared], resamples)
        for c, (estimate, low, high) in zip(compared, found, strict=True):
            rows.append(
                {
                    "dataset": dataset, "family": c.family, "test": c.kind, "a": c.a, "b": c.b,
                    "n_a": len(arms[c.a]), "n_b": len(arms[c.b]), "users": arms[c.a].shape[1],
                    "resamples": resamples, "estimate": estimate, "low": low, "high": high,
                }
            )  # fmt: skip
    write_csv(out, rows)
    return format_table(
        rows, ["dataset", "family", "a", "b", "n_a", "n_b", "estimate", "low", "high"]
    )
