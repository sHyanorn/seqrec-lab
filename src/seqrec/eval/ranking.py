"""Full-ranking evaluation (METHODS section 6).

Scores are fp32 with autocast disabled. Every item the user saw before the target is
excluded, ties count against the model, and each batch checks its own invariants.
Inputs are gathered from the flat CSR arrays, so no per-user Python loop runs.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from seqrec.data.invariants import InvariantError
from seqrec.data.splits import EvalSplit

Scorer = Callable[[torch.Tensor], torch.Tensor]  # (B, L) left-padded ids -> (B, >= N+1) scores
KS = (10, 20)


class NonFiniteScores(InvariantError):
    """The model produced NaN or infinite scores, usually after training diverged."""


@dataclass(frozen=True)
class EvalResult:
    metrics: dict[str, float]
    tie_rate: float
    users: int
    ranks: np.ndarray  # kept in memory only; never written to results/


@dataclass(frozen=True)
class DeviceSplit:
    """An evaluation split's arrays on the scoring device."""

    indptr: torch.Tensor
    items: torch.Tensor
    targets: torch.Tensor

    @classmethod
    def of(cls, split: EvalSplit, device: torch.device) -> DeviceSplit:
        def put(a):
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int64)).to(device)

        return cls(put(split.inputs.indptr), put(split.inputs.items), put(split.targets))


def model_inputs(ds: DeviceSplit, a: int, b: int, length: int) -> torch.Tensor:
    """The last `length` items of rows a..b-1, left-padded with 0."""
    starts, ends = ds.indptr[a:b], ds.indptr[a + 1 : b + 1]
    idx = ends[:, None] - length + torch.arange(length, device=ends.device)
    return torch.where(idx >= starts[:, None], ds.items[idx.clamp(min=0)], 0)


def history_entries(ds: DeviceSplit, a: int, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(row, item) pairs for every item rows a..b-1 saw before their target."""
    lengths = ds.indptr[a + 1 : b + 1] - ds.indptr[a:b]
    rows = torch.repeat_interleave(torch.arange(b - a, device=lengths.device), lengths)
    return rows, ds.items[ds.indptr[a] : ds.indptr[b]]


def mask_history(scores: torch.Tensor, rows: torch.Tensor, items: torch.Tensor) -> None:
    scores.index_put_((rows, items), torch.tensor(float("-inf"), device=scores.device))


@contextmanager
def eval_autocast(device: torch.device):
    """Strict fp32 scores: autocast off and no TF32 in fp32 matmuls."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.autocast(device.type, enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(previous)


def rank_targets(
    scores: torch.Tensor,
    targets: torch.Tensor,
    history: tuple[torch.Tensor, torch.Tensor],
    history_sizes: torch.Tensor,
    n_items: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pessimistic 1-based ranks of the targets and whether each target ties with a candidate.

    `history` is (row, item) pairs; `history_sizes` is the number of distinct items each row
    saw before its target. Out-of-vocabulary targets (id < 1) get rank N+1, a guaranteed miss.
    """
    if scores.dtype != torch.float32:
        raise InvariantError("evaluation scores must be fp32")
    if scores.dim() != 2 or scores.shape[1] < n_items + 1:
        raise InvariantError("scores must cover item ids 0..N")
    s = scores[:, : n_items + 1].clone()
    if not torch.isfinite(s[:, 1:]).all():
        raise NonFiniteScores("scores must be finite")
    s[:, 0] = float("-inf")
    mask_history(s, *history)

    oov = targets < 1
    target_scores = s.gather(1, targets.clamp(min=1)[:, None]).squeeze(1)
    if not (torch.isfinite(target_scores) | oov).all():
        raise InvariantError("target score must stay finite after masking")
    candidates = torch.isfinite(s[:, 1:]).sum(1)
    if not torch.equal(candidates, n_items - history_sizes):
        raise InvariantError("candidate count must equal N minus distinct history items")

    at_least = (s >= target_scores[:, None]).sum(1) - 1  # minus the target itself
    ranks = torch.where(oov, torch.full_like(at_least, n_items + 1), at_least + 1)
    ties = ((s == target_scores[:, None]).sum(1) - 1 > 0) & ~oov
    return ranks, ties


def metrics_from_ranks(ranks: np.ndarray, ks=KS) -> dict[str, float]:
    out = {}
    for k in ks:
        hit = ranks <= k
        out[f"ndcg@{k}"] = float(np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0).mean())
        out[f"hr@{k}"] = float(hit.mean())
    return out


@torch.no_grad()
def evaluate(
    scorer: Scorer,
    split: EvalSplit,
    max_len: int,
    device: torch.device | str = "cpu",
    batch_size: int = 1024,
    ks=KS,
) -> EvalResult:
    device = torch.device(device)
    ds = DeviceSplit.of(split, device)
    # Inputs hold distinct items (METHODS 2.4, checked by the split invariants),
    # so each input length is the number of distinct history items.
    sizes = ds.indptr[1:] - ds.indptr[:-1]
    ranks, ties = [], []
    for a in range(0, len(split), batch_size):
        b = min(a + batch_size, len(split))
        with eval_autocast(device):
            scores = scorer(model_inputs(ds, a, b, max_len))
        r, t = rank_targets(
            scores, ds.targets[a:b], history_entries(ds, a, b), sizes[a:b], split.n_items
        )
        ranks.append(r.cpu().numpy())
        ties.append(t.cpu().numpy())
    all_ranks = np.concatenate(ranks) if ranks else np.zeros(0, dtype=np.int64)
    all_ties = np.concatenate(ties) if ties else np.zeros(0, dtype=bool)
    return EvalResult(
        metrics_from_ranks(all_ranks, ks),
        float(all_ties.mean()) if len(all_ties) else 0.0,
        len(all_ranks),
        all_ranks,
    )
