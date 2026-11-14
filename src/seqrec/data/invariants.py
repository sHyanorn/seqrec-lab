"""Always-on invariant checks (METHODS 11.1). Each returns the names of the checks it ran.

Checks recompute what they can from the source data with their own code, so a bug in
the split builders cannot hide behind values the builders report themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seqrec.data.preprocess import Interactions, Log
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
    _require(np.array_equal(seqs.items, full.items[seqs.pos]), f"{name}: items match log", passed)
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


def _targets_from_log(
    log: Log, in_vocab: np.ndarray, after: np.ndarray, up_to: np.ndarray
) -> np.ndarray:
    """Sorted log rows of the targets of one temporal evaluation set (METHODS 3.2): each user
    with an interaction in `after` whose last interaction in `up_to`, the target, has a
    vocabulary item at an earlier position. Positions follow the log's order (METHODS 2.3)."""
    rows = np.arange(len(log))
    n_users = log.user.max(initial=-1) + 1
    last = np.full(n_users, -1, dtype=np.int64)
    np.maximum.at(last, log.user[up_to], rows[up_to])
    has_after = np.bincount(log.user[after], minlength=n_users) > 0
    earlier_vocab = in_vocab & (rows < last[log.user])
    has_input = np.bincount(log.user[earlier_vocab], minlength=n_users) > 0
    return np.sort(last[has_after & has_input])


def _core(user: np.ndarray, item: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Which user and item codes the k-core of the user-item graph keeps (METHODS 2.5), as
    masks over the codes, written apart from the split builders' filter: users and items are
    removed in turn, each by its degree among the survivors, until a round over both removes
    nothing. The k-core is the largest subgraph in which every user and item has at least k
    interactions, so any order of removal ends there."""
    user_alive = np.bincount(user) > 0
    item_alive = np.bincount(item) > 0
    while True:
        edges = user_alive[user] & item_alive[item]
        user_degree = np.bincount(user[edges], minlength=len(user_alive))
        users_out = user_alive & (user_degree < k)
        user_alive &= ~users_out
        edges = user_alive[user] & item_alive[item]
        item_degree = np.bincount(item[edges], minlength=len(item_alive))
        items_out = item_alive & (item_degree < k)
        item_alive &= ~items_out
        if not users_out.any() and not items_out.any():
            return user_alive, item_alive


def _loo_rows_from_log(full: Interactions, log: Log, k: int) -> bool:
    """The leave-one-out sequences are the rows of the whole log that its k-core keeps, in
    the log's order, with users numbered in order and items 1..N in key order (METHODS 2.5,
    2.6); the k-core is recomputed here."""
    user_alive, item_alive = _core(log.user, log.item, k)
    keep = user_alive[log.user] & item_alive[log.item]
    _, user = np.unique(log.user[keep], return_inverse=True)
    item = np.searchsorted(np.flatnonzero(item_alive), log.item[keep]) + 1
    return (
        full.n_users == int(user_alive.sum())
        and full.n_items == int(item_alive.sum())
        and np.array_equal(owner_of(full), user)
        and np.array_equal(full.ts, log.ts[keep])
        and np.array_equal(full.items, item)
    )


def _training_from_log(
    log: Log, in_vocab: np.ndarray, val_cutoff: int
) -> tuple[np.ndarray, np.ndarray]:
    """Log rows and lengths of the temporal training sequences, in the log's order: "each
    user's vocabulary interactions with t ≤ T_val, for users with at least two of them"
    (METHODS 3.2)."""
    fit = in_vocab & (log.ts <= val_cutoff)
    counts = np.bincount(log.user[fit], minlength=log.user.max(initial=-1) + 1)
    return np.flatnonzero(fit & (counts[log.user] >= 2)), counts[counts >= 2]


def _check_gts_against_log(
    splits: SplitSet, test: EvalSplit, log: Log, k: int, passed: list[str]
) -> None:
    """Recompute cutoffs, vocabulary and the evaluation users with their targets from the
    raw log."""
    full = splits.full
    cutoff = int(np.quantile(log.ts, GTS_QUANTILE, method="lower"))
    early = log.ts <= cutoff
    val_cutoff = int(np.quantile(log.ts[early], GTS_QUANTILE, method="lower"))
    _require(
        splits.meta["cutoff"] == cutoff and splits.meta["validation_cutoff"] == val_cutoff,
        "gts: cutoffs recomputed from log",
        passed,
    )
    _, items_alive = _core(log.user[early], log.item[early], k)
    in_vocab = np.isin(log.item, np.flatnonzero(items_alive))
    # Row by row, the split's rows are the log's: the same user and time, and in the
    # vocabulary exactly where the recomputed 5-core puts them.
    _require(
        np.array_equal(owner_of(full), log.user)
        and np.array_equal(full.ts, log.ts)
        and np.array_equal(full.items > 0, in_vocab),
        f"gts: vocabulary is the {k}-core of the training period",
        passed,
    )
    # Each validation target is its user's last row up to the cutoff in the split; and each
    # evaluation set holds every user it should and no other, with that user's target, as
    # recomputed from the source log. The training sequences are compared the same way with
    # those of METHODS 3.2, which METHODS 11.1 does not list. The name predates both.
    v = splits.validation
    train = splits.train.seqs
    last_val = _last_row_at_or_before(full, cutoff)
    validation = _targets_from_log(log, in_vocab, early & (log.ts > val_cutoff), early)
    testing = _targets_from_log(log, in_vocab, ~early, np.ones(len(log), dtype=bool))
    train_rows, train_lengths = _training_from_log(log, in_vocab, val_cutoff)
    _require(
        np.array_equal(v.target_pos, last_val[owner_of(full)[v.target_pos]])
        and np.array_equal(np.sort(v.target_pos), validation)
        and np.array_equal(np.sort(test.target_pos), testing)
        and np.array_equal(train.pos, train_rows)
        and np.array_equal(train.lengths(), train_lengths),
        "gts: validation target is the last item up to the cutoff",
        passed,
    )


def check_split(splits: SplitSet, log: Log | None = None, k: int = 5) -> list[str]:
    """Structural checks for a split set. With the source log, both splits are also compared
    with it; the temporal split's construction is checked only then."""
    full = splits.full
    test = splits.test.unseal("check")
    passed: list[str] = []
    train = splits.train.seqs
    _require(np.array_equal(train.items, full.items[train.pos]), "train: items match log", passed)
    _require((train.items > 0).all(), "train: items in vocabulary", passed)
    targets = np.concatenate([splits.validation.target_pos, test.target_pos])
    _require(not np.isin(train.pos, targets).any(), "train: no evaluation targets", passed)
    for split in (splits.validation, test):
        _check_eval(full, split, passed)

    if splits.kind == "loo":
        lengths = full.lengths()
        val = splits.validation
        # With the log, each user's full sequence is also compared with the log's rows.
        _require(
            (log is None or _loo_rows_from_log(full, log, k))
            and (train.lengths() == lengths - 2).all()
            and (val.inputs.lengths() == lengths - 2).all()
            and (test.inputs.lengths() == lengths - 1).all(),
            "loo: sequence lengths",
            passed,
        )
        _require((val.target_pos == full.indptr[1:] - 2).all(), "loo: validation target", passed)
        _require((test.target_pos == full.indptr[1:] - 1).all(), "loo: test target", passed)
        # Row by row and in order: a shuffled split permutes the training sequence and the
        # inputs built from it by the same keys (METHODS 3.3), so the order must agree too.
        _require(
            np.array_equal(train.indptr, val.inputs.indptr)
            and np.array_equal(train.pos, val.inputs.pos),
            "loo: validation input equals training sequence",
            passed,
        )
        test_owner = owner_of(test.inputs)
        is_val_item = test.inputs.pos == val.target_pos[test_owner]
        _require(
            (np.bincount(test_owner[is_val_item], minlength=len(test)) == 1).all()
            and np.array_equal(test.inputs.pos[~is_val_item], train.pos),
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
            _check_gts_against_log(splits, test, log, k, passed)
    return passed


def check_batch_positions(splits: SplitSet, positions: np.ndarray) -> list[str]:
    """A sampled training batch uses training rows only (METHODS 11.1). The test split may
    be sealed here; check_split already showed that training rows hold no test target."""
    passed: list[str] = []
    _require(np.isin(positions, splits.train.seqs.pos).all(), "batch: training rows only", passed)
    _require(
        not np.isin(positions, splits.validation.target_pos).any(),
        "batch: no validation targets",
        passed,
    )
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
