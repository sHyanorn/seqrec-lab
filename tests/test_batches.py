import numpy as np
import pytest
import torch
from conftest import random_frame

from seqrec.data.invariants import InvariantError, check_batch_positions
from seqrec.data.preprocess import build_log
from seqrec.data.splits import Sequences, TrainSplit
from seqrec.data.store import build_splits
from seqrec.models import batches

CPU = torch.device("cpu")


def train_split(rows: list[list[int]], n_items: int) -> TrainSplit:
    lengths = [len(r) for r in rows]
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    items = np.concatenate([np.asarray(r, dtype=np.int64) for r in rows])
    pos = np.arange(len(items), dtype=np.int64) * 3  # positions in some larger log
    return TrainSplit(n_items, Sequences(indptr, items, np.zeros_like(items), pos))


def test_sasrec_batch_shifts_labels_and_pads_left():
    data = batches.TrainData(train_split([[5, 6, 7, 8], [3, 9]], 10), CPU)
    batch = batches.sasrec_batch(data, torch.tensor([0, 1]), length=4)
    assert batch.inputs.tolist() == [[0, 5, 6, 7], [0, 0, 0, 3]]
    assert batch.labels.tolist() == [[0, 6, 7, 8], [0, 0, 0, 9]]
    batches.check_sasrec_batch(batch, data)
    short = batches.sasrec_batch(data, torch.tensor([0]), length=2)
    assert short.inputs.tolist() == [[6, 7]] and short.labels.tolist() == [[7, 8]]


def test_sasrec_check_rejects_unshifted_and_borrowed_labels():
    data = batches.TrainData(train_split([[5, 6, 7, 8], [3, 9]], 10), CPU)
    good = batches.sasrec_batch(data, torch.tensor([0, 1]), length=4)
    unshifted = batches.Batch(good.rows, good.inputs, good.inputs, good.window)
    with pytest.raises(InvariantError):
        batches.check_sasrec_batch(unshifted, data)
    borrowed = batches.Batch(good.rows, good.inputs, good.labels, good.window.flip(0))
    with pytest.raises(InvariantError, match="another training row"):
        batches.check_sasrec_batch(borrowed, data)


def test_rows_with_one_item_give_no_training_pair():
    data = batches.TrainData(train_split([[4], [1, 2], [3]], 5), CPU)
    assert data.rows.tolist() == [1]


def test_bert4rec_masks_at_rate_with_at_least_one_per_row():
    rng = np.random.default_rng(0)
    items = np.arange(1, 400)
    rows = [list(rng.choice(items, rng.integers(2, 60), replace=False)) for _ in range(500)]
    data = batches.TrainData(train_split(rows, 400), CPU)
    gen = torch.Generator().manual_seed(0)
    batch = batches.bert4rec_batch(data, data.rows, 50, 0.2, gen)
    batches.check_bert4rec_batch(batch, data)
    masked, real = batch.labels > 0, batch.window >= 0
    assert masked.any(1).all()
    rate = masked.sum().item() / real.sum().item()
    assert abs(rate - 0.2) < 0.02  # plus a small excess from the at-least-one rule
    assert (batch.inputs[masked] == 401).all() and (batch.inputs[real & ~masked] <= 400).all()


def test_bert4rec_check_rejects_unmasked_rows():
    data = batches.TrainData(train_split([[5, 6, 7], [3, 9]], 10), CPU)
    batch = batches.bert4rec_batch(data, data.rows, 4, 0.5, torch.Generator().manual_seed(1))
    leaky = batches.Batch(batch.rows, batch.labels.where(batch.labels > 0, batch.inputs),
                          batch.labels, batch.window)  # fmt: skip
    with pytest.raises(InvariantError):
        batches.check_bert4rec_batch(leaky, data)


def test_history_sampler_is_uniform_over_unseen_items():
    rows = [[2, 5, 9], [1, 2, 3, 4, 5, 6, 7, 8], [10]]
    train = train_split(rows, 10)
    sampler = batches.HistoryExcludingSampler(train, CPU)
    gen = torch.Generator().manual_seed(0)
    users = torch.tensor([0, 1, 2])
    neg = sampler.sample(users, (20000,), gen)
    for r, seen in enumerate(rows):
        unseen = sorted(set(range(1, 11)) - set(seen))
        values, counts = np.unique(neg[r].numpy(), return_counts=True)
        assert values.tolist() == unseen
        expected = 20000 / len(unseen)
        assert np.abs(counts - expected).max() < 5 * np.sqrt(expected)


def test_history_sampler_rejects_other_splits():
    train = train_split([[2, 5], [1, 3]], 6)
    other = train_split([[2, 5, 6], [1, 3]], 6)
    sampler = batches.HistoryExcludingSampler(train, CPU)
    sampler.require(train)
    with pytest.raises(InvariantError, match="not computed from this training split"):
        sampler.require(other)
    with pytest.raises(TypeError):
        batches.HistoryExcludingSampler(train.seqs, CPU)


def test_batch_positions_pass_the_split_check():
    splits, _ = build_splits(build_log(random_frame()), "loo")
    data = batches.TrainData(splits.train, CPU)
    batch = batches.sasrec_batch(data, data.rows[:32], 8)
    positions = batches.batch_positions(data, batch)
    assert check_batch_positions(splits, positions)
    with pytest.raises(InvariantError, match="training rows only"):
        check_batch_positions(splits, np.append(positions, splits.validation.target_pos[0]))
