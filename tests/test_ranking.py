"""Known-answer tests and independent cross-checks for the evaluator (METHODS 11.2, 11.4)."""

import math

import numpy as np
import pytest
import torch
from conftest import random_frame

from seqrec.data.invariants import InvariantError
from seqrec.data.preprocess import build_log
from seqrec.data.store import build_splits
from seqrec.eval.ranking import evaluate, metrics_from_ranks, rank_targets


def naive_rank(scores, target, history):
    """Reference: sort every candidate by score, ties placed before the target."""
    candidates = [j for j in range(1, len(scores)) if j not in set(history)]
    better_or_equal = [j for j in candidates if j != target and scores[j] >= scores[target]]
    return len(better_or_equal) + 1


def test_hand_computed_ranks_and_ties():
    scores = torch.tensor([[9.0, 0.5, 0.9, 0.7, 0.7, 0.1]])  # column 0 is padding
    history = torch.tensor([[2, 0]])  # item 2 already seen
    ranks, ties = rank_targets(scores, torch.tensor([3]), history, n_items=5)
    assert ranks.item() == 2 and ties.item()  # item 4 ties with the target and counts ahead
    m = metrics_from_ranks(ranks.numpy())
    assert m["ndcg@10"] == pytest.approx(1 / math.log2(3))
    assert m["hr@10"] == 1.0


def test_all_equal_scores_rank_last():
    n = 50
    ranks, ties = rank_targets(torch.zeros(1, n + 1), torch.tensor([7]), torch.tensor([[1, 2]]), n)
    assert ranks.item() == n - 2 and ties.item()


def test_out_of_vocabulary_target_is_a_miss():
    ranks, ties = rank_targets(torch.rand(1, 11), torch.tensor([-1]), torch.tensor([[0]]), 10)
    assert ranks.item() == 11 and not ties.item()


def test_matches_naive_reference_on_random_inputs():
    rng = np.random.default_rng(0)
    n, users = 300, 2000
    scores = torch.from_numpy(rng.normal(size=(users, n + 1)).astype(np.float32))
    scores[:, 1::7] = 0.25  # deliberate ties
    history = torch.from_numpy(rng.integers(0, n + 1, size=(users, 12)))
    targets = torch.from_numpy(rng.integers(1, n + 1, size=users))
    clash = (history == targets[:, None]).any(1)
    targets[clash] = 0  # avoid targets inside the history, as the splits guarantee
    targets[clash] = torch.where(history[clash, 0] == 1, 2, 1)
    keep = ~(history == targets[:, None]).any(1)
    ranks, _ = rank_targets(scores[keep], targets[keep], history[keep], n)
    expected = [
        naive_rank(s.tolist(), int(t), h.tolist())
        for s, t, h in zip(scores[keep], targets[keep], history[keep], strict=True)
    ]
    assert ranks.tolist() == expected


def test_random_scorer_matches_analytic_expectation():
    """Uniform random rank r in 1..C: E[HR@K] = K/C, E[NDCG@K] = (1/C) sum_{r<=K} 1/log2(r+1)."""
    rng = np.random.default_rng(1)
    c, k, users = 200, 10, 40_000
    ranks = rng.integers(1, c + 1, size=users)
    m = metrics_from_ranks(ranks, ks=(k,))
    expected_hr = k / c
    expected_ndcg = sum(1 / math.log2(r + 1) for r in range(1, k + 1)) / c
    se_hr = math.sqrt(expected_hr * (1 - expected_hr) / users)
    assert abs(m[f"hr@{k}"] - expected_hr) < 4 * se_hr
    assert abs(m[f"ndcg@{k}"] - expected_ndcg) < 4 * se_hr

    scores = torch.from_numpy(rng.random((users // 10, c + 1)).astype(np.float32))
    targets = torch.from_numpy(rng.integers(1, c + 1, size=users // 10))
    ranks_t, _ = rank_targets(scores, targets, torch.zeros(users // 10, 1, dtype=torch.long), c)
    hr = (ranks_t <= k).float().mean().item()
    assert abs(hr - expected_hr) < 4 * math.sqrt(expected_hr * (1 - expected_hr) / (users // 10))


def test_rejects_low_precision_scores():
    with pytest.raises(InvariantError):
        rank_targets(
            torch.zeros(1, 6, dtype=torch.bfloat16),
            torch.tensor([1]),
            torch.zeros(1, 1, dtype=torch.long),
            5,
        )


def test_evaluate_masks_full_history_beyond_model_length():
    splits, _ = build_splits(build_log(random_frame()), "loo")
    split = splits.validation
    n = split.n_items

    def scorer(inputs):
        return torch.arange(n + 1, dtype=torch.float32).repeat(len(inputs), 1)

    result = evaluate(scorer, split, max_len=2)  # inputs truncated to 2, history is not
    assert result.users == len(split)
    for i in range(0, len(split), 7):
        history = set(split.inputs.row(i).tolist())
        target = int(split.targets[i])
        better = [j for j in range(target + 1, n + 1) if j not in history]
        assert result.ranks[i] == len(better) + 1


def test_cross_check_with_ranx():
    ranx = pytest.importorskip("ranx")
    rng = np.random.default_rng(2)
    n, users = 100, 500
    scores = (
        rng.permutation(np.arange(1, (n + 1) * users + 1)).reshape(users, n + 1).astype(np.float32)
    )
    targets = rng.integers(1, n + 1, size=users)
    ranks, _ = rank_targets(
        torch.from_numpy(scores),
        torch.from_numpy(targets),
        torch.zeros(users, 1, dtype=torch.long),
        n,
    )
    ours = metrics_from_ranks(ranks.numpy(), ks=(10,))
    qrels = ranx.Qrels({f"u{u}": {f"i{targets[u]}": 1} for u in range(users)})
    run = ranx.Run(
        {f"u{u}": {f"i{j}": float(scores[u, j]) for j in range(1, n + 1)} for u in range(users)}
    )
    theirs = ranx.evaluate(qrels, run, ["ndcg@10", "hit_rate@10"])
    assert ours["ndcg@10"] == pytest.approx(theirs["ndcg@10"], abs=1e-6)
    assert ours["hr@10"] == pytest.approx(theirs["hit_rate@10"], abs=1e-6)
