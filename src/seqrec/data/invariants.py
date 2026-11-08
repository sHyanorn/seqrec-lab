"""Always-on invariant checks (METHODS 11.1). Each returns the names of the checks it ran.

Checks recompute what they can from the source data with their own code, so a bug in
the split builders cannot hide behind values the builders report themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seqrec.data.preprocess import Interactions, Log, kcore_mask
from seqrec.data.splits import EvalSplit, Sequences, SplitSet, TrainSplit

GTS_QUANTILE = 0.9


class InvariantError(AssertionError):
    pass


def _require(ok, name: str, passed: list[str]) -> None:
    if not bool(ok):
        raise InvariantError(name)
    passed.append(name)


def owner_of(seqs: Sequences | Interactions) -> np.ndarray:
    """Row-to-sequence index for CSR data."""
    return np.repeat(np.arange(len(seqs.indptr) - 1), np.diff(seqs.indptr))


def check_interactions(full: Interactions, k: int) -> list[str]:
    passed: list[str] = []
    owner = owner_of(full)
    in_vocab = full.items > 0
    _require(full.items[in_vocab].max(initial=0) <= full.n_items, "item ids within 1..N", passed)
    _require(len(np.unique(full.items[in_vocab])) == full.n_items, "item ids contiguous", passed)
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


def _check_eval(full: Interactions, split: EvalSplit, passed: list[str]) -> None:
    """Each input must be exactly the user's in-vocabulary rows before the target."""
    name = split.phase
    seqs = split.inputs
    owner = owner_of(seqs)
    _require(
        (full.items[split.target_pos] == split.targets).all(), f"{name}: target matches log", passed
    )
    _require((seqs.items > 0).all(), f"{name}: inputs in vocabulary", passed)
    _require((seqs.pos < split.target_pos[owner]).all(), f"{name}: inputs precede target", passed)
    user_start = full.indptr[np.searchsorted(full.indptr, split.target_pos, side="right") - 1]
    _require((seqs.pos >= user_start[owner]).all(), f"{name}: inputs from same user", passed)
    order = np.lexsort((seqs.pos, owner))
    same = owner[order][1:] == owner[order][:-1]
    _require((np.diff(seqs.pos[order])[same] > 0).all(), f"{name}: no repeated input rows", passed)
    in_vocab_before = np.concatenate([[0], np.cumsum(full.items > 0)])
    expected = in_vocab_before[split.target_pos] - in_vocab_before[user_start]
    _require(
        np.array_equal(seqs.lengths(), expected),
        f"{name}: input holds every earlier in-vocabulary item",
        passed,
    )
    _require((seqs.lengths() >= 1).all(), f"{name}: non-empty inputs", passed)
    hits = np.zeros(len(split), dtype=bool)
    np.logical_or.at(hits, owner, seqs.items == split.targets[owner])
    _require(not (hits & (split.targets > 0)).any(), f"{name}: target not in its history", passed)
    if not split.shuffled:
        _require(
            (np.diff(seqs.pos)[owner[1:] == owner[:-1]] > 0).all(), f"{name}: log order", passed
        )


def _last_row_at_or_before(full: Interactions, cutoff: int) -> np.ndarray:
    owner = owner_of(full)
    last = np.full(full.n_users, -1, dtype=np.int64)
    rows = np.flatnonzero(full.ts <= cutoff)
    np.maximum.at(last, owner[rows], rows)
    return last


def _check_gts_against_log(splits: SplitSet, log: Log, k: int, passed: list[str]) -> None:
    """Recompute cutoffs and vocabulary from the raw log."""
    full = splits.full
    cutoff = int(np.quantile(log.ts, GTS_QUANTILE, method="lower"))
    early = log.ts <= cutoff
    val_cutoff = int(np.quantile(log.ts[early], GTS_QUANTILE, method="lower"))
    _require(
        splits.meta["cutoff"] == cutoff and splits.meta["validation_cutoff"] == val_cutoff,
        "gts: cutoffs recomputed from log",
        passed,
    )
    vocab = np.unique(log.item[early][kcore_mask(log.user[early], log.item[early], k)])
    _require(
        np.array_equal(full.items > 0, np.isin(log.item, vocab)),
        f"gts: vocabulary is the {k}-core of the training period",
        passed,
    )
    v = splits.validation
    last_val = _last_row_at_or_before(full, cutoff)
    _require(
        np.array_equal(v.target_pos, last_val[owner_of(full)[v.target_pos]]),
        "gts: validation target is the last item up to the cutoff",
        passed,
    )


def check_split(splits: SplitSet, log: Log | None = None, k: int = 5) -> list[str]:
    """Structural checks for a split set; the temporal split also needs the source log."""
    full = splits.full
    test = splits.test.unseal("check")
    passed: list[str] = []
    train = splits.train.seqs
    _require((train.items > 0).all(), "train: items in vocabulary", passed)
    targets = np.concatenate([splits.validation.target_pos, test.target_pos])
    _require(not np.isin(train.pos, targets).any(), "train: no evaluation targets", passed)
    for split in (splits.validation, test):
        _check_eval(full, split, passed)

    if splits.kind == "loo":
        lengths = full.lengths()
        val = splits.validation
        _require(
            (train.lengths() == lengths - 2).all()
            and (val.inputs.lengths() == lengths - 2).all()
            and (test.inputs.lengths() == lengths - 1).all(),
            "loo: sequence lengths",
            passed,
        )
        _require((val.target_pos == full.indptr[1:] - 2).all(), "loo: validation target", passed)
        _require((test.target_pos == full.indptr[1:] - 1).all(), "loo: test target", passed)
        _require(
            np.array_equal(np.sort(train.pos), np.sort(val.inputs.pos)),
            "loo: validation input equals training sequence",
            passed,
        )
        test_rows = np.sort(np.concatenate([train.pos, val.target_pos]))
        _require(
            np.array_equal(np.sort(test.inputs.pos), test_rows),
            "loo: test input equals training sequence plus validation item",
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
        _require(
            np.isin(test.target_pos, full.indptr[1:] - 1).all(), "gts: test target is last", passed
        )
        if log is not None:
            _check_gts_against_log(splits, log, k, passed)
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
