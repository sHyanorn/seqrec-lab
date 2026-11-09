"""Training batches and negative samplers, built from the training split only.

METHODS 4.2, 4.3, 5.1 and 11.1. Sequences stay in CSR form on the device; a batch is a
set of training rows, each cut to its most recent items and left-padded with 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from seqrec.data.invariants import InvariantError
from seqrec.data.splits import TrainSplit


def device_array(a: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int64)).to(device)


def window_index(indptr: torch.Tensor, rows: torch.Tensor, length: int) -> torch.Tensor:
    """Flat indices of the last `length` entries of each row, -1 where the row is shorter."""
    starts, ends = indptr[rows], indptr[rows + 1]
    idx = ends[:, None] - length + torch.arange(length, device=indptr.device)
    return torch.where(idx >= starts[:, None], idx, -1)


def gather(values: torch.Tensor, idx: torch.Tensor, fill: int = 0) -> torch.Tensor:
    return torch.where(idx >= 0, values[idx.clamp(min=0)], fill)


class TrainData:
    """A training split on the device. Rows with fewer than two items give no training pair."""

    def __init__(self, train: TrainSplit, device: torch.device):
        if not isinstance(train, TrainSplit):
            raise TypeError("training batches accept only a TrainSplit")
        seqs = train.seqs
        self.source = train.fingerprint
        self.n_items = train.n_items
        self.device = device
        self.indptr = device_array(seqs.indptr, device)
        self.items = device_array(seqs.items, device)
        self.pos = device_array(seqs.pos, device)
        self.owner = device_array(np.repeat(np.arange(len(seqs)), seqs.lengths()), device)
        self.rows = device_array(np.flatnonzero(seqs.lengths() >= 2), device)

    def require(self, train: TrainSplit) -> None:
        if train.fingerprint != self.source:
            raise InvariantError("training data were not built from this training split")


@dataclass(frozen=True)
class Batch:
    rows: torch.Tensor  # (B,) training rows
    inputs: torch.Tensor  # (B, L) item ids, 0 = padding, N+1 = mask token (BERT4Rec)
    labels: torch.Tensor  # (B, L) target item ids, 0 where no loss applies
    window: torch.Tensor  # (B, W) flat index of every training item used, -1 for padding


def sasrec_batch(data: TrainData, rows: torch.Tensor, length: int) -> Batch:
    """The most recent L+1 training items: inputs are the first L, labels the next item."""
    idx = window_index(data.indptr, rows, length + 1)
    window = gather(data.items, idx)
    inputs = window[:, :-1]
    has_label = inputs > 0
    labels = torch.where(has_label, window[:, 1:], 0)
    return Batch(rows, inputs, labels, idx)


def check_sasrec_batch(batch: Batch, data: TrainData) -> None:
    """Labels are the inputs shifted by one position and end with each row's last training
    item; every element comes from its own row (METHODS 11.1)."""
    inputs, labels = batch.inputs, batch.labels
    has_label = labels > 0
    if not torch.equal(has_label, inputs > 0):
        raise InvariantError("labels must exist exactly at non-padded input positions")
    inner = has_label[:, :-1]
    if not torch.equal(labels[:, :-1][inner], inputs[:, 1:][inner]):
        raise InvariantError("labels must be the inputs shifted by one position")
    last = data.items[data.indptr[batch.rows + 1] - 1]
    if not torch.equal(labels[:, -1], last):
        raise InvariantError("the last label must be the row's last training item")
    _check_rows(batch, data)
    if not torch.equal(gather(data.items, batch.window[:, 1:])[has_label], labels[has_label]):
        raise InvariantError("labels must be the training items that follow the inputs")


def _check_rows(batch: Batch, data: TrainData) -> None:
    used = batch.window >= 0
    if not (data.owner[batch.window.clamp(min=0)] == batch.rows[:, None])[used].all():
        raise InvariantError("a batch row holds items from another training row")


def bert4rec_batch(
    data: TrainData, rows: torch.Tensor, length: int, mask_prob: float, generator: torch.Generator
) -> Batch:
    """The most recent L training items. Each position is masked with probability
    `mask_prob`, at least one per row, and a masked position always shows the mask token."""
    idx = window_index(data.indptr, rows, length)
    items = gather(data.items, idx)
    real = items > 0
    draw = torch.rand(items.shape, generator=generator, device=items.device)
    masked = (draw < mask_prob) & real
    # Rows without a masked position mask their real position with the smallest draw.
    fallback = torch.where(real, draw, 2.0).argmin(1)
    none = ~masked.any(1)
    masked[none, fallback[none]] = True
    mask_id = data.n_items + 1
    return Batch(rows, torch.where(masked, mask_id, items), torch.where(masked, items, 0), idx)


def check_bert4rec_batch(batch: Batch, data: TrainData) -> None:
    masked = batch.labels > 0
    if not masked.any(1).all():
        raise InvariantError("every row needs at least one masked position")
    if not (batch.inputs[masked] == data.n_items + 1).all():
        raise InvariantError("masked positions must show the mask token")
    if (batch.inputs[~masked] == data.n_items + 1).any():
        raise InvariantError("the mask token may appear only at masked positions")
    _check_rows(batch, data)
    items = gather(data.items, batch.window)
    if not torch.equal(torch.where(masked, batch.labels, batch.inputs), items):
        raise InvariantError("inputs and labels must be the row's training items")


def batch_positions(data: TrainData, batch: Batch) -> np.ndarray:
    """Full-log positions of every item a batch shows or predicts, for the runner's check."""
    return np.unique(data.pos[batch.window[batch.window >= 0]].cpu().numpy())


class HistoryExcludingSampler:
    """Uniform negatives over the items absent from the user's training sequence (A1).

    Sampling is exact, without rejection: with the user's training items sorted as
    m_0 < m_1 < ..., c_i = m_i - 1 - i counts the non-members below m_i, so the u-th
    non-member (0-based) is u + 1 + #{i : c_i <= u}.
    """

    def __init__(self, train: TrainSplit, device: torch.device):
        if not isinstance(train, TrainSplit):
            raise TypeError("the negative sampler accepts only a TrainSplit")
        self.source = train.fingerprint
        n = train.n_items
        seqs = train.seqs
        owner = np.repeat(np.arange(len(seqs)), seqs.lengths())
        order = np.lexsort((seqs.items, owner))
        members = seqs.items[order]
        rank = np.arange(len(members)) - np.repeat(seqs.indptr[:-1], seqs.lengths())
        self.stride = n + 1
        self.keys = device_array(owner * self.stride + (members - 1 - rank), device)
        self.starts = device_array(seqs.indptr[:-1], device)
        free = n - seqs.lengths()  # training items are distinct
        if (free < 1).any():
            raise ValueError("a user's training sequence covers every item")
        self.free = device_array(free, device)
        self.n_items = n

    def require(self, train: TrainSplit) -> None:
        if train.fingerprint != self.source:
            raise InvariantError("statistic was not computed from this training split")

    def sample(self, rows: torch.Tensor, shape: tuple[int, ...], generator: torch.Generator):
        """Negatives for each row, shaped (len(rows), *shape)."""
        free = self.free[rows].view(-1, *([1] * len(shape)))
        draw = torch.rand((len(rows), *shape), generator=generator, device=rows.device)
        u = (draw * free).long().clamp(max=free - 1)
        query = rows.view_as(free) * self.stride + u
        below = torch.searchsorted(self.keys, query.flatten(), right=True).view_as(u)
        return u + 1 + below - self.starts[rows].view_as(free)


def uniform_negatives(n_items: int, shape: tuple[int, ...], generator: torch.Generator, device):
    """Uniform item ids 1..N with replacement (A3, A4)."""
    return torch.randint(1, n_items + 1, shape, generator=generator, device=device)
