"""Inject known silent bugs; the listed check must fail (METHODS 11.3, docs/MUTATIONS.md)."""

import numpy as np
import pytest
import torch
from conftest import frame, random_frame

from seqrec.data import splits as splits_mod
from seqrec.data.invariants import InvariantError, item_popularity, require_from_train
from seqrec.data.preprocess import build_log
from seqrec.data.splits import Sequences, TrainSplit
from seqrec.data.store import build_splits
from seqrec.eval import ranking


@pytest.fixture
def log():
    return build_log(random_frame())


def _caught(log, match, kind="loo"):
    with pytest.raises(InvariantError, match=match):
        build_splits(log, kind)


def test_validation_item_in_training(log, monkeypatch):
    monkeypatch.setattr(splits_mod, "_loo_train_len", lambda lengths: lengths - 1)
    _caught(log, "train: no evaluation targets")


def test_test_item_in_training(log, monkeypatch):
    monkeypatch.setattr(splits_mod, "_loo_train_len", lambda lengths: lengths)
    _caught(log, "train: no evaluation targets")


def test_targets_off_by_one(log, monkeypatch):
    monkeypatch.setattr(splits_mod, "_loo_target_offsets", lambda n: (n - 3, n - 2))
    _caught(log, "train: no evaluation targets")


def test_target_inside_its_input(log, monkeypatch):
    original = splits_mod.select

    def leaky(full, owner, keep, n_rows):  # every selection also takes the next item
        return original(full, owner, keep | np.roll(keep, 1), n_rows)

    monkeypatch.setattr(splits_mod, "select", leaky)
    _caught(log, "train: no evaluation targets")


def test_evaluation_input_drops_an_item(log, monkeypatch):
    original = splits_mod.select

    def lossy(full, owner, keep, n_rows):  # evaluation inputs lose their first item
        first = np.zeros_like(keep)
        idx = np.flatnonzero(keep)
        if len(idx):
            starts = idx[np.r_[True, owner[idx][1:] != owner[idx][:-1]]]
            first[starts] = True
        return original(full, owner, keep & ~first, n_rows)

    monkeypatch.setattr(splits_mod, "select", lossy)
    _caught(log, "earlier in-vocabulary item|validation input equals training")


def test_statistic_from_all_data(log):
    splits, _ = build_splits(log, "loo")
    full = splits.full
    everything = Sequences(full.indptr, full.items, full.ts, np.arange(len(full.items)))
    stat = item_popularity(TrainSplit(full.n_items, everything))
    with pytest.raises(InvariantError, match="not computed from this training split"):
        require_from_train(stat, splits.train)


def _tied_rows():
    rows = [(f"u{u}", f"i{(u + j) % 15}", 10 * j) for u in range(40) for j in range(10)]
    return rows + [("late", "i1", 75), ("late", "new", 90)]


def _tied_log():
    return build_log(frame(_tied_rows()))  # every timestamp is shared by many interactions


def test_temporal_cutoff_off_by_one_at_ties(monkeypatch):
    monkeypatch.setattr(splits_mod, "_gts_in_period", lambda ts, cutoff: ts < cutoff)
    _caught(_tied_log(), "gts: validation targets inside validation period", "gts")


def test_temporal_vocabulary_from_whole_log(monkeypatch):
    def whole_log_vocab(log, in_train_period, k):  # post-cutoff items enter the vocabulary
        return np.unique(log.item)

    monkeypatch.setattr(splits_mod, "_gts_vocabulary", whole_log_vocab)
    _caught(_tied_log(), "vocabulary is the 5-core of the training period", "gts")


def test_temporal_validation_target_first_not_last(monkeypatch):
    original = splits_mod._last_target

    def first_target(full, owner, candidate, phase):
        if phase != "validation":
            return original(full, owner, candidate, phase)
        firsts = np.zeros_like(candidate)
        rows = np.flatnonzero(candidate)
        if len(rows):
            firsts[rows[np.r_[True, owner[rows][1:] != owner[rows][:-1]]]] = True
        return original(full, owner, firsts, phase)

    monkeypatch.setattr(splits_mod, "_last_target", first_target)
    dense = [(f"u{u}", f"i{(u + j) % 25}", 40 * j + u) for u in range(40) for j in range(20)]
    _caught(build_log(frame(dense)), "last item up to the cutoff", "gts")


def _validation(log):
    splits, _ = build_splits(log, "loo")
    return splits.validation


def _uniform_scorer(n_items):
    weights = torch.linspace(0, 1, n_items + 1)

    def scorer(inputs):
        return torch.ones(len(inputs), 1) @ weights[None, :]  # a matmul: autocast changes dtype

    return scorer


def test_history_masking_removed(log, monkeypatch):
    split = _validation(log)
    monkeypatch.setattr(ranking, "mask_history", lambda scores, rows, items: None)
    with pytest.raises(InvariantError, match="candidate count"):
        ranking.evaluate(_uniform_scorer(split.n_items), split, max_len=50)


def test_history_masking_limited_to_truncated_input(log, monkeypatch):
    split = _validation(log)

    def truncated(ds, a, b):
        rows, items = ranking.model_inputs(ds, a, b, 2).nonzero(as_tuple=True)
        return rows, ranking.model_inputs(ds, a, b, 2)[rows, items]

    monkeypatch.setattr(ranking, "history_entries", truncated)
    with pytest.raises(InvariantError, match="candidate count"):
        ranking.evaluate(_uniform_scorer(split.n_items), split, max_len=2)


def test_history_masking_applied_to_target(log, monkeypatch):
    split = _validation(log)
    original = ranking.history_entries

    def with_target(ds, a, b):
        rows, items = original(ds, a, b)
        extra = torch.arange(b - a)
        return torch.cat([rows, extra]), torch.cat([items, ds.targets[a:b]])

    monkeypatch.setattr(ranking, "history_entries", with_target)
    with pytest.raises(InvariantError, match="target score must stay finite"):
        ranking.evaluate(_uniform_scorer(split.n_items), split, max_len=50)


def test_scores_in_bf16(log, monkeypatch):
    split = _validation(log)
    monkeypatch.setattr(
        ranking, "eval_autocast", lambda device: torch.autocast("cpu", dtype=torch.bfloat16)
    )
    with pytest.raises(InvariantError, match="fp32"):
        ranking.evaluate(_uniform_scorer(split.n_items), split, max_len=50)
