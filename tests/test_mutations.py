"""Inject known silent bugs; the listed check must fail (METHODS 11.3, docs/MUTATIONS.md)."""

from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import frame, random_frame
from test_losses import (
    check_full_ce_matches_naive_softmax_over_items_only,
    check_sampled_softmax_drops_negatives_equal_to_the_positive,
)
from test_sasrec import assert_causal, net

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
from seqrec.models import batches, losses, sasrec
from seqrec.runner.config import RunConfig
from seqrec.runner.run import ARMS


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


@pytest.mark.parametrize("part", ["train", "validation", "test"])
def test_items_detached_from_their_log_positions(log, part):
    """Two items of one sequence change places while their positions stay: every check made
    through the positions still passes."""
    splits, _ = build_splits(log, "loo")

    def swapped(seqs):
        items = seqs.items.copy()
        items[[0, 1]] = items[[1, 0]]
        return replace(seqs, items=items)

    if part == "train":
        mutated = replace(splits, train=replace(splits.train, seqs=swapped(splits.train.seqs)))
    elif part == "validation":
        inputs = swapped(splits.validation.inputs)
        mutated = replace(splits, validation=replace(splits.validation, inputs=inputs))
    else:
        test = splits.test.unseal("check")
        mutated = replace(
            splits, test=splits_mod.TestHandle(replace(test, inputs=swapped(test.inputs)))
        )
    with pytest.raises(InvariantError, match=f"{part}: items match log"):
        check_split(mutated)


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


# Added in M2, with the models.


def _sasrec(arm, train):
    torch.manual_seed(0)
    cfg = RunConfig("b", "ml-1m", "loo", arm, 1, "tune", params={"dim": 16}, max_len=8)
    return ARMS[arm](cfg, train, torch.device("cpu"))


def test_sasrec_labels_not_shifted(log, monkeypatch):
    splits, _ = build_splits(log, "loo")
    original = batches.sasrec_batch

    def unshifted(data, rows, length):
        b = original(data, rows, length)
        return batches.Batch(b.rows, b.inputs, b.inputs, b.window)

    monkeypatch.setattr(batches, "sasrec_batch", unshifted)
    with pytest.raises(InvariantError, match="shifted by one"):
        _sasrec("sasrec-ce", splits.train).train_epoch(splits.train, 0)


def test_bce_negatives_exclude_the_full_history(log, monkeypatch):
    splits, _ = build_splits(log, "loo")
    full = splits.full
    everything = TrainSplit(
        full.n_items, Sequences(full.indptr, full.items, full.ts, np.arange(len(full.items)))
    )  # includes the validation and test items
    original = batches.HistoryExcludingSampler
    monkeypatch.setattr(
        batches, "HistoryExcludingSampler", lambda train, device: original(everything, device)
    )
    model = _sasrec("sasrec-bce", splits.train)
    with pytest.raises(InvariantError, match="not computed from this training split"):
        model.train_epoch(splits.train, 0)


def test_sasrec_causal_mask_removed(monkeypatch):
    def bidirectional(real):
        n = real.shape[1]
        return (real[:, None, :] | torch.eye(n, dtype=torch.bool))[:, None].expand(-1, 1, n, n)

    monkeypatch.setattr(sasrec, "attention_mask", bidirectional)
    with pytest.raises(AssertionError):
        assert_causal(net(), torch.randint(1, 31, (4, 8)), 3)


def test_softmax_includes_the_padding_id(monkeypatch):
    def with_padding(h, w, y, bias=None, n_items=None, chunk=0):
        logits = h @ w[: n_items + 1].T + (0 if bias is None else bias[: n_items + 1])
        return F.cross_entropy(logits, y)

    monkeypatch.setattr(losses, "full_ce", with_padding)
    with pytest.raises(AssertionError):
        check_full_ce_matches_naive_softmax_over_items_only()


def test_sampled_softmax_keeps_collisions(monkeypatch):
    monkeypatch.setattr(losses, "mask_collisions", lambda neg_logits, neg, y: neg_logits)
    with pytest.raises(AssertionError):
        check_sampled_softmax_drops_negatives_equal_to_the_positive()
