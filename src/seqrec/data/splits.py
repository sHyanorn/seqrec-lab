"""Typed train, validation and test splits (METHODS section 3).

Every sequence keeps `pos`, the index of each element in the full per-user log, so that
invariants can check targets and inputs against the source order. Test data sit behind
`TestHandle` and are reached only through `seqrec.eval.test_access.read_test`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from seqrec.data.preprocess import Interactions, Log, fingerprint, kcore_mask, remap


@dataclass(frozen=True)
class Sequences:
    indptr: np.ndarray
    items: np.ndarray
    ts: np.ndarray
    pos: np.ndarray  # flat index into the full sequences of the split set

    def __len__(self) -> int:
        return len(self.indptr) - 1

    def lengths(self) -> np.ndarray:
        return np.diff(self.indptr)

    def row(self, i: int) -> np.ndarray:
        return self.items[self.indptr[i] : self.indptr[i + 1]]


def select(full: Interactions, owner: np.ndarray, keep: np.ndarray, n_rows: int) -> Sequences:
    """Rows of `full` where keep is true, grouped into `n_rows` sequences by `owner`.

    `owner` must be non-decreasing over the kept rows, which holds when rows follow
    the user order of `full`.
    """
    idx = np.flatnonzero(keep)
    counts = np.bincount(owner[idx], minlength=n_rows)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return Sequences(indptr, full.items[idx], full.ts[idx], idx.astype(np.int64))


@dataclass(frozen=True)
class TrainSplit:
    n_items: int
    seqs: Sequences
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fingerprint", fingerprint(self.seqs.items, self.seqs.pos))


@dataclass(frozen=True)
class EvalSplit:
    phase: str  # "validation" or "test"
    n_items: int
    inputs: Sequences  # everything before the target; models truncate to their length
    targets: np.ndarray  # item ids; -1 marks an out-of-vocabulary target (always a miss)
    target_ts: np.ndarray
    target_pos: np.ndarray  # flat index of each target in the full sequences
    shuffled: bool = False

    def __len__(self) -> int:
        return len(self.targets)


class TestHandle:
    """Sealed container for the test split."""

    __slots__ = ("_split",)

    def __init__(self, split: EvalSplit):
        self._split = split

    def __repr__(self) -> str:
        return "TestHandle(<sealed>)"


@dataclass(frozen=True)
class SplitSet:
    kind: str  # "loo" or "gts"
    full: Interactions  # every user's full sequence; -1 marks out-of-vocabulary items
    train: TrainSplit
    validation: EvalSplit
    test: TestHandle
    meta: dict
    shuffled: bool = False

    @property
    def n_items(self) -> int:
        return self.full.n_items


def _row_owner(full: Interactions) -> np.ndarray:
    return np.repeat(np.arange(full.n_users), full.lengths())


def _local_pos(full: Interactions) -> np.ndarray:
    """Position of each flat row within its user's sequence."""
    return np.arange(len(full.items)) - np.repeat(full.indptr[:-1], full.lengths())


def _loo_train_len(lengths: np.ndarray) -> np.ndarray:
    return lengths - 2


def _loo_target_offsets(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Offsets of the validation and test targets within each user sequence."""
    return lengths - 2, lengths - 1


def leave_one_out(full: Interactions) -> SplitSet:
    lengths = full.lengths()
    if full.n_users == 0:
        raise ValueError("no users left after filtering")
    if (lengths < 3).any():
        raise ValueError("leave-one-out needs at least three interactions per user")
    owner, local = _row_owner(full), _local_pos(full)
    train_len = _loo_train_len(lengths)
    val_off, test_off = _loo_target_offsets(lengths)
    starts = full.indptr[:-1]

    n = full.n_users
    train = select(full, owner, local < train_len[owner], n)
    validation = EvalSplit(
        "validation",
        full.n_items,
        select(full, owner, local < val_off[owner], n),
        full.items[starts + val_off],
        full.ts[starts + val_off],
        (starts + val_off).astype(np.int64),
    )
    test = EvalSplit(
        "test",
        full.n_items,
        select(full, owner, local < test_off[owner], n),
        full.items[starts + test_off],
        full.ts[starts + test_off],
        (starts + test_off).astype(np.int64),
    )
    meta = {"users": full.n_users, "items": full.n_items, "interactions": len(full.items)}
    return SplitSet(
        "loo", full, TrainSplit(full.n_items, train), validation, TestHandle(test), meta
    )


def time_quantile(ts: np.ndarray, q: float) -> int:
    """An actual timestamp from the data: the lower q-quantile."""
    return int(np.quantile(ts, q, method="lower"))


def _gts_in_period(ts: np.ndarray, cutoff: int) -> np.ndarray:
    """Interactions at or before the cutoff belong to the earlier period ([12] code)."""
    return ts <= cutoff


def _last_target(
    full: Interactions, owner: np.ndarray, candidate: np.ndarray, phase: str
) -> EvalSplit:
    """Each user's last candidate row is the target; earlier in-vocabulary rows are the input."""
    n_users = full.n_users
    last = np.full(n_users, -1, dtype=np.int64)
    rows = np.flatnonzero(candidate)
    last[owner[rows]] = rows  # rows are increasing, so the last write per user wins
    input_rows = (np.arange(len(full.items)) < last[owner]) & (full.items > 0)
    has_input = np.bincount(owner[input_rows], minlength=n_users) > 0
    users = np.flatnonzero((last >= 0) & has_input)
    keep_rows = input_rows & np.isin(owner, users)
    dense = np.full(n_users, -1, dtype=np.int64)
    dense[users] = np.arange(len(users))
    inputs = select(full, dense[owner].clip(min=0), keep_rows, len(users))
    target_pos = last[users]
    return EvalSplit(
        phase, full.n_items, inputs, full.items[target_pos], full.ts[target_pos], target_pos
    )


def global_temporal(log: Log, k: int = 5, q: float = 0.9) -> SplitSet:
    """GTS with last-item targets (METHODS 3.2)."""
    if len(log) == 0:
        raise ValueError("no interactions to split")
    cutoff = time_quantile(log.ts, q)
    in_train_period = _gts_in_period(log.ts, cutoff)
    core = kcore_mask(log.user[in_train_period], log.item[in_train_period], k)
    vocab = np.unique(log.item[in_train_period][core])
    items = remap(log.item, vocab, start=1)  # -1 outside the vocabulary
    full = Interactions(
        np.concatenate([[0], np.cumsum(np.bincount(log.user))]).astype(np.int64),
        items,
        log.ts.astype(np.int64),
        len(vocab),
    )
    owner = _row_owner(full)
    val_cutoff = time_quantile(log.ts[in_train_period], q)
    in_fit_period = _gts_in_period(full.ts, val_cutoff)

    fit_rows = in_fit_period & (full.items > 0)
    fit_counts = np.bincount(owner[fit_rows], minlength=full.n_users)
    train_users = np.flatnonzero(fit_counts >= 2)
    dense = np.full(full.n_users, -1, dtype=np.int64)
    dense[train_users] = np.arange(len(train_users))
    train = select(
        full, dense[owner].clip(min=0), fit_rows & np.isin(owner, train_users), len(train_users)
    )

    validation = _last_target(full, owner, in_train_period & ~in_fit_period, "validation")
    test = _last_target(full, owner, ~in_train_period, "test")
    meta = {
        "cutoff": cutoff,
        "validation_cutoff": val_cutoff,
        "vocabulary_items": len(vocab),
        "train_users": len(train_users),
        "train_interactions": len(train.items),
        "validation_users": len(validation),
        "test_users": len(test),
        "validation_oov_target_share": float((validation.targets < 1).mean())
        if len(validation)
        else 0.0,
        "test_oov_target_share": float((test.targets < 1).mean()) if len(test) else 0.0,
    }
    return SplitSet(
        "gts", full, TrainSplit(full.n_items, train), validation, TestHandle(test), meta
    )


def _shuffle_rows(seqs: Sequences, keys: np.ndarray) -> Sequences:
    """Reorder each sequence by a random key attached to its full-log position."""
    owner = np.repeat(np.arange(len(seqs)), seqs.lengths())
    order = np.lexsort((keys[seqs.pos], owner))
    return Sequences(seqs.indptr, seqs.items[order], seqs.ts[order], seqs.pos[order])


def shuffle_eval_inputs(split: EvalSplit, seed: int, n_full_rows: int) -> EvalSplit:
    keys = np.random.default_rng(seed).random(n_full_rows)
    return replace(split, inputs=_shuffle_rows(split.inputs, keys), shuffled=True)


def shuffle_order(splits: SplitSet, seed: int) -> SplitSet:
    """Train-and-evaluate shuffle (METHODS 3.3): one random key per full-log position, so a
    user's training sequence and validation input are permuted consistently."""
    keys = np.random.default_rng(seed).random(len(splits.full.items))
    train = TrainSplit(splits.n_items, _shuffle_rows(splits.train.seqs, keys))
    validation = replace(
        splits.validation, inputs=_shuffle_rows(splits.validation.inputs, keys), shuffled=True
    )
    test_split = splits.test._split
    test = TestHandle(
        replace(test_split, inputs=_shuffle_rows(test_split.inputs, keys), shuffled=True)
    )
    return replace(splits, train=train, validation=validation, test=test, shuffled=True)
