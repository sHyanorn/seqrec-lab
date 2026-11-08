"""Inject known silent bugs and require a check to fail (METHODS 11.3, docs/MUTATIONS.md)."""

import numpy as np
import pytest
import torch
from conftest import frame, random_frame

from seqrec.data import splits as splits_mod
from seqrec.data.invariants import (
    InvariantError,
    check_split,
    item_popularity,
    require_from_train,
)
from seqrec.data.preprocess import build_log
from seqrec.data.splits import Sequences, TrainSplit
from seqrec.data.store import build_splits
from seqrec.eval import ranking


@pytest.fixture
def log():
    return build_log(random_frame())


def _expect_failure(log, kind="loo"):
    with pytest.raises(InvariantError):
        build_splits(log, kind)


def test_validation_item_in_training(log, monkeypatch):
    monkeypatch.setattr(splits_mod, "_loo_train_len", lambda lengths: lengths - 1)
    _expect_failure(log)


def test_test_item_in_training(log, monkeypatch):
    monkeypatch.setattr(splits_mod, "_loo_train_len", lambda lengths: lengths)
    _expect_failure(log)


def test_targets_off_by_one(log, monkeypatch):
    monkeypatch.setattr(splits_mod, "_loo_target_offsets", lambda n: (n - 3, n - 2))
    _expect_failure(log)


def test_target_inside_its_input(log, monkeypatch):
    original = splits_mod.select

    def leaky(full, owner, keep, n_rows):  # every selection also takes the next item
        return original(full, owner, keep | np.roll(keep, 1), n_rows)

    monkeypatch.setattr(splits_mod, "select", leaky)
    _expect_failure(log)


def test_statistic_from_all_data(log):
    splits, _ = build_splits(log, "loo")
    full = splits.full
    everything = Sequences(full.indptr, full.items, full.ts, np.arange(len(full.items)))
    stat = item_popularity(TrainSplit(full.n_items, everything))
    with pytest.raises(InvariantError):
        require_from_train(stat, splits.train)


def test_temporal_cutoff_off_by_one_at_ties(monkeypatch):
    rows = [(f"u{u}", f"i{(u + j) % 15}", 10 * j) for u in range(40) for j in range(10)]
    log = build_log(frame(rows))  # every timestamp is shared by 40 interactions
    monkeypatch.setattr(splits_mod, "_gts_in_period", lambda ts, cutoff: ts < cutoff)
    splits = splits_mod.global_temporal(log)
    with pytest.raises(InvariantError):
        check_split(splits)


def _validation(log):
    splits, _ = build_splits(log, "loo")
    return splits.validation


def _uniform_scorer(n_items):
    weights = torch.linspace(0, 1, n_items + 1)

    def scorer(inputs):
        emb = torch.ones(len(inputs), 1)
        return emb @ weights[None, :]  # a matmul, so autocast would change the dtype

    return scorer


def test_history_masking_removed(log, monkeypatch):
    split = _validation(log)
    monkeypatch.setattr(ranking, "mask_history", lambda scores, history: None)
    with pytest.raises(InvariantError):
        ranking.evaluate(_uniform_scorer(split.n_items), split, max_len=50)


def test_history_masking_limited_to_truncated_input(log, monkeypatch):
    split = _validation(log)
    monkeypatch.setattr(
        ranking, "history_matrix", lambda s, rows: ranking.left_padded(s.inputs, rows, 2)
    )
    with pytest.raises(InvariantError):
        ranking.evaluate(_uniform_scorer(split.n_items), split, max_len=2)


def test_history_masking_applied_to_target(log, monkeypatch):
    split = _validation(log)
    original = ranking.history_matrix

    def with_target(s, rows):
        return np.concatenate([original(s, rows), s.targets[rows][:, None]], axis=1)

    monkeypatch.setattr(ranking, "history_matrix", with_target)
    with pytest.raises(InvariantError):
        ranking.evaluate(_uniform_scorer(split.n_items), split, max_len=50)


def test_scores_in_bf16(log, monkeypatch):
    split = _validation(log)
    monkeypatch.setattr(
        ranking, "eval_autocast", lambda device: torch.autocast("cpu", dtype=torch.bfloat16)
    )
    with pytest.raises(InvariantError):
        ranking.evaluate(_uniform_scorer(split.n_items), split, max_len=50)
