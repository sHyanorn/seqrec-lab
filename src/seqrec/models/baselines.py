"""Floor baselines B1 (MostPop) and B2 (first-order transition), METHODS 4.4.

Both statistics come from the training split only and carry its fingerprint. They need no
training, so the first epoch reports convergence and the runner stops after it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from seqrec.data.invariants import item_popularity, require_from_train
from seqrec.data.splits import TrainSplit
from seqrec.models.batches import device_array
from seqrec.runner.config import RunConfig
from seqrec.runner.run import register_arm


@dataclass(frozen=True)
class TransitionCounts:
    """Counts of consecutive training pairs (a, b) in CSR form over a."""

    indptr: np.ndarray
    nxt: np.ndarray
    counts: np.ndarray
    source: str


def transition_counts(train: TrainSplit) -> TransitionCounts:
    if not isinstance(train, TrainSplit):
        raise TypeError("transition counts accept only a TrainSplit")
    seqs, n = train.seqs, train.n_items
    owner = np.repeat(np.arange(len(seqs)), seqs.lengths())
    same = owner[1:] == owner[:-1]
    pairs, counts = np.unique(
        seqs.items[:-1][same] * (n + 1) + seqs.items[1:][same], return_counts=True
    )
    prev, nxt = np.divmod(pairs, n + 1)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(prev, minlength=n + 1))])
    return TransitionCounts(indptr.astype(np.int64), nxt, counts, train.fingerprint)


def dense_rank_rows(key: torch.Tensor) -> torch.Tensor:
    """Per-row dense ranks of int64 keys as fp32: equal keys keep equal scores, and ranks
    stay exact in fp32 where large keys would not."""
    sorted_key, order = key.sort(dim=1)
    new = torch.ones_like(sorted_key, dtype=torch.bool)
    new[:, 1:] = sorted_key[:, 1:] != sorted_key[:, :-1]
    return torch.empty_like(order).scatter_(1, order, new.cumsum(1)).float()


class _Baseline:
    hparams: dict = {}

    def __init__(self, cfg: RunConfig, train: TrainSplit, device: torch.device):
        if cfg.params:
            raise ValueError("baselines take no hyperparameters")
        self.device = device
        self.popularity = item_popularity(train)
        self.pop = device_array(self.popularity.counts, device)

    def train_epoch(self, train: TrainSplit, epoch: int) -> dict:
        require_from_train(self.popularity, train)
        return {"converged": True}

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


class MostPop(_Baseline):
    def score(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pop.float().expand(len(inputs), -1)


class Transition(_Baseline):
    """Score of j: count of (last input item, j) in training sequences; ties broken by
    training popularity."""

    def __init__(self, cfg: RunConfig, train: TrainSplit, device: torch.device):
        super().__init__(cfg, train, device)
        self.table = transition_counts(train)
        self.indptr = device_array(self.table.indptr, device)
        self.nxt = device_array(self.table.nxt, device)
        self.counts = device_array(self.table.counts, device)

    def train_epoch(self, train: TrainSplit, epoch: int) -> dict:
        require_from_train(self.table, train)
        return super().train_epoch(train, epoch)

    def score(self, inputs: torch.Tensor) -> torch.Tensor:
        last = inputs[:, -1]  # inputs are left-padded, so the last column is the latest item
        starts, ends = self.indptr[last], self.indptr[last + 1]
        lengths = ends - starts
        rows = torch.repeat_interleave(torch.arange(len(last), device=last.device), lengths)
        offsets = torch.arange(int(lengths.sum()), device=last.device)
        offsets = offsets - torch.repeat_interleave(lengths.cumsum(0) - lengths, lengths)
        flat = starts[rows] + offsets
        count = torch.zeros((len(last), len(self.pop)), dtype=torch.int64, device=last.device)
        count[rows, self.nxt[flat]] = self.counts[flat]
        key = count * (int(self.pop.max()) + 1) + self.pop
        return dense_rank_rows(key)


register_arm("mostpop")(MostPop)
register_arm("transition")(Transition)
