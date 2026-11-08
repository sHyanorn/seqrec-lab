"""Full-ranking evaluation (METHODS section 6).

Scores are fp32 with autocast disabled. Every item the user saw before the target is
excluded, ties count against the model, and each batch checks its own invariants.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from seqrec.data.invariants import InvariantError
from seqrec.data.splits import EvalSplit, Sequences

Scorer = Callable[[torch.Tensor], torch.Tensor]  # (B, L) left-padded ids -> (B, >= N+1) scores
KS = (10, 20)


@dataclass(frozen=True)
class EvalResult:
    metrics: dict[str, float]
    tie_rate: float
    users: int
    ranks: np.ndarray  # kept in memory only; never written to results/


def left_padded(seqs: Sequences, rows: np.ndarray, length: int) -> np.ndarray:
    """The last `length` items of each row, left-padded with 0."""
    out = np.zeros((len(rows), length), dtype=np.int64)
    for j, r in enumerate(rows):
        items = seqs.row(r)[-length:]
        if len(items):
            out[j, -len(items) :] = items
    return out


def _check_scores(scores: torch.Tensor, n_items: int) -> None:
    if scores.dtype != torch.float32:
        raise InvariantError("evaluation scores must be fp32")
    if scores.dim() != 2 or scores.shape[1] < n_items + 1:
        raise InvariantError("scores must cover item ids 0..N")


def mask_history(scores: torch.Tensor, history: torch.Tensor) -> None:
    """Set every history item to -inf; history is (B, H) with 0 as padding."""
    scores.scatter_(1, history, float("-inf"))


def rank_targets(
    scores: torch.Tensor,
    targets: torch.Tensor,
    history: torch.Tensor,
    n_items: int,
    history_sizes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pessimistic 1-based ranks of the targets and whether each target ties with a candidate.

    Out-of-vocabulary targets (id < 1) get rank N+1, a guaranteed miss. `history_sizes`
    is the number of distinct items each user saw before the target; when given, the
    candidate count is checked against it rather than against `history` itself.
    """
    _check_scores(scores, n_items)
    s = scores[:, : n_items + 1].clone()
    s[:, 0] = float("-inf")
    mask_history(s, history)

    oov = targets < 1
    target_scores = s.gather(1, targets.clamp(min=1)[:, None]).squeeze(1)
    if not (torch.isfinite(target_scores) | oov).all():
        raise InvariantError("target score must stay finite after masking")
    if history_sizes is None:
        sorted_hist = history.sort(dim=1).values
        history_sizes = (sorted_hist[:, :1] > 0).sum(1) + (
            (sorted_hist[:, 1:] != sorted_hist[:, :-1]) & (sorted_hist[:, 1:] > 0)
        ).sum(1)
    candidates = torch.isfinite(s[:, 1:]).sum(1)
    if not torch.equal(candidates, n_items - history_sizes.to(candidates.device)):
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


def history_matrix(split: EvalSplit, rows: np.ndarray) -> np.ndarray:
    """Every item each user saw before the target, not only the model's truncated input."""
    return left_padded(split.inputs, rows, max(int(split.inputs.lengths()[rows].max()), 1))


def eval_autocast(device: torch.device):
    return torch.autocast(device.type, enabled=False)


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
    ranks, ties = [], []
    for start in range(0, len(split), batch_size):
        rows = np.arange(start, min(start + batch_size, len(split)))
        model_in = torch.from_numpy(left_padded(split.inputs, rows, max_len)).to(device)
        history = history_matrix(split, rows)
        with eval_autocast(device):
            scores = scorer(model_in)
        r, t = rank_targets(
            scores,
            torch.from_numpy(split.targets[rows]).to(scores.device),
            torch.from_numpy(history).to(scores.device),
            split.n_items,
            # inputs hold distinct items (METHODS 2.4), so their length is the history size
            torch.from_numpy(split.inputs.lengths()[rows]),
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
