"""Local data layout and split construction.

    data/raw/<dataset>/<file>         downloaded source file and source.json
    data/processed/<dataset>/log.npz  deduplicated, stably sorted interactions

Nothing under data/ is committed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from seqrec.data.invariants import check_interactions, check_split
from seqrec.data.preprocess import Log, leave_one_out_interactions
from seqrec.data.splits import SplitSet, global_temporal, leave_one_out

K_CORE = 5


def log_path(data_dir: Path, dataset: str) -> Path:
    return data_dir / "processed" / dataset / "log.npz"


@lru_cache(maxsize=4)
def load_log(data_dir: Path, dataset: str) -> Log:
    path = log_path(data_dir, dataset)
    if not path.exists():
        raise FileNotFoundError(f"{dataset} is not prepared; run `seqrec data prepare`")
    return Log.load(path)


def build_splits(log: Log, kind: str) -> tuple[SplitSet, list[str]]:
    """Build a split set and run its invariants; returns the split and the checks passed."""
    if kind == "loo":
        full = leave_one_out_interactions(log, K_CORE)
        checks = check_interactions(full, K_CORE)
        splits = leave_one_out(full)
    elif kind == "gts":
        splits = global_temporal(log, K_CORE)
        checks = check_interactions(splits.full, 0)
    else:
        raise ValueError(f"unknown split kind {kind!r}")
    return splits, checks + check_split(splits)


def load_splits(data_dir: Path, dataset: str, kind: str) -> tuple[SplitSet, list[str]]:
    return build_splits(load_log(data_dir, dataset), kind)
