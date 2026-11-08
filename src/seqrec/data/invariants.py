"""Always-on invariant checks (METHODS 11.1). Each returns the names of the checks it ran."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seqrec.data.preprocess import Interactions
from seqrec.data.splits import EvalSplit, Sequences, SplitSet, TrainSplit


class InvariantError(AssertionError):
    pass


def _require(ok, name: str, passed: list[str]) -> None:
    if not bool(ok):
        raise InvariantError(name)
    passed.append(name)


def _owner(seqs: Sequences) -> np.ndarray:
    return np.repeat(np.arange(len(seqs)), seqs.lengths())


def check_interactions(full: Interactions, k: int) -> list[str]:
    passed: list[str] = []
    owner = np.repeat(np.arange(full.n_users), full.lengths())
    in_vocab = full.items > 0
    _require(full.items[in_vocab].max(initial=0) <= full.n_items, "item ids within 1..N", passed)
    ids = np.unique(full.items[in_vocab])
    _require(len(ids) == full.n_items, "item ids contiguous", passed)
    pairs = owner[in_vocab] * (full.n_items + 1) + full.items[in_vocab]
    _require(len(np.unique(pairs)) == len(pairs), "no repeated (user, item) pairs", passed)
    same_user = owner[1:] == owner[:-1]
    _require(
        (full.ts[1:][same_user] >= full.ts[:-1][same_user]).all(), "timestamps ordered", passed
    )
    if k:
        _require((full.lengths() >= k).all(), f"users meet {k}-core", passed)
        _require((np.bincount(full.items)[1:] >= k).all(), f"items meet {k}-core", passed)
    return passed


def _check_eval_common(full: Interactions, split: EvalSplit, passed: list[str]) -> None:
    name = split.phase
    seqs = split.inputs
    owner = _owner(seqs)
    in_vocab = split.targets > 0
    _require(
        (full.items[split.target_pos] == split.targets).all(), f"{name}: target matches log", passed
    )
    _require((seqs.items > 0).all(), f"{name}: inputs in vocabulary", passed)
    _require((seqs.pos < split.target_pos[owner]).all(), f"{name}: inputs precede target", passed)
    user_start = np.searchsorted(full.indptr, split.target_pos, side="right") - 1
    _require(
        (seqs.pos >= full.indptr[user_start][owner]).all(), f"{name}: inputs from same user", passed
    )
    _require((seqs.lengths() >= 1).all(), f"{name}: non-empty inputs", passed)
    hits = np.zeros(len(split), dtype=bool)
    np.logical_or.at(hits, owner, seqs.items == split.targets[owner])
    _require(not (hits & in_vocab).any(), f"{name}: target not in its history", passed)
    if not split.shuffled:
        same = owner[1:] == owner[:-1]
        _require((np.diff(seqs.pos)[same] > 0).all(), f"{name}: inputs in log order", passed)
        _require(
            (seqs.ts <= split.target_ts[owner]).all(), f"{name}: input times before target", passed
        )


def _first_rows(seqs: Sequences) -> np.ndarray:
    return seqs.indptr[:-1]


def check_split(splits: SplitSet) -> list[str]:
    """Structural checks for a split set; the test split is checked through its handle."""
    full = splits.full
    test = splits.test._split
    passed: list[str] = []
    train = splits.train.seqs
    _require((train.items > 0).all(), "train: items in vocabulary", passed)
    targets = np.concatenate([splits.validation.target_pos, test.target_pos])
    _require(not np.isin(train.pos, targets).any(), "train: no evaluation targets", passed)
    for split in (splits.validation, test):
        _check_eval_common(full, split, passed)

    if splits.kind == "loo":
        lengths = full.lengths()
        val, tst = splits.validation, test
        _require(
            (train.lengths() == lengths - 2).all()
            and (val.inputs.lengths() == lengths - 2).all()
            and (tst.inputs.lengths() == lengths - 1).all(),
            "loo: sequence lengths",
            passed,
        )
        _require((val.target_pos == full.indptr[1:] - 2).all(), "loo: validation target", passed)
        _require((tst.target_pos == full.indptr[1:] - 1).all(), "loo: test target", passed)
        _require(
            np.array_equal(np.sort(train.pos), np.sort(val.inputs.pos)),
            "loo: validation input equals training sequence",
            passed,
        )
        test_rows = np.sort(np.concatenate([train.pos, val.target_pos]))
        _require(
            np.array_equal(np.sort(tst.inputs.pos), test_rows),
            "loo: test input equals training sequence plus validation item",
            passed,
        )
        if not splits.shuffled:
            _require(
                np.array_equal(train.pos, val.inputs.pos)
                and (full.ts[val.target_pos] <= full.ts[tst.target_pos]).all(),
                "loo: order train <= validation <= test",
                passed,
            )
    else:
        cutoff, val_cutoff = splits.meta["cutoff"], splits.meta["validation_cutoff"]
        _require((train.ts <= val_cutoff).all(), "gts: training before validation cutoff", passed)
        v = splits.validation
        _require(
            ((v.target_ts > val_cutoff) & (v.target_ts <= cutoff)).all(),
            "gts: validation targets inside validation period",
            passed,
        )
        _require((test.target_ts > cutoff).all(), "gts: test targets after cutoff", passed)
        last_rows = full.indptr[1:] - 1
        _require(np.isin(test.target_pos, last_rows).all(), "gts: test target is last item", passed)
    return passed


@dataclass(frozen=True)
class ItemCounts:
    """A statistic computed from a training split, tagged with that split's fingerprint."""

    counts: np.ndarray
    source: str


def item_popularity(train: TrainSplit) -> ItemCounts:
    if not isinstance(train, TrainSplit):
        raise TypeError("item popularity accepts only a TrainSplit")
    counts = np.bincount(train.seqs.items, minlength=train.n_items + 1)
    return ItemCounts(counts, train.fingerprint)


def require_from_train(stat: ItemCounts, train: TrainSplit) -> None:
    if stat.source != train.fingerprint:
        raise InvariantError("statistic was not computed from this training split")
