"""Known-answer tests and independent cross-checks for the evaluator (METHODS 11.2, 11.4)."""

import math

import numpy as np
import pytest
import torch
from conftest import random_frame

from seqrec.data.invariants import InvariantError
from seqrec.data.preprocess import build_log
from seqrec.data.store import build_splits
from seqrec.eval.ranking import NonFiniteScores, evaluate, metrics_from_ranks, rank_targets


def flat(history):
    """Dense (B, H) history with 0 padding -> (rows, items) pairs and distinct counts."""
    rows, cols = torch.nonzero(history > 0, as_tuple=True)
    sizes = torch.tensor([len(set(h[h > 0].tolist())) for h in history])
    return (rows, history[rows, cols]), sizes


def rank(scores, targets, history, n):
    pairs, sizes = flat(history)
    return rank_targets(scores, targets, pairs, sizes, n)


def naive_rank(scores, target, history):
    """Reference: every candidate scoring at least the target's score ranks ahead of it."""
    seen = set(history)
    ahead = [j for j in range(1, len(scores)) if j not in seen and j != target]
    return sum(scores[j] >= scores[target] for j in ahead) + 1


def test_hand_computed_ranks_and_ties():
    scores = torch.tensor([[9.0, 0.5, 0.9, 0.7, 0.7, 0.1]])  # column 0 is padding
    ranks, ties = rank(scores, torch.tensor([3]), torch.tensor([[2, 0]]), 5)
    assert ranks.item() == 2 and ties.item()  # item 4 ties with the target and counts ahead
    m = metrics_from_ranks(ranks.numpy())
    assert m["ndcg@10"] == pytest.approx(1 / math.log2(3))
    assert m["hr@10"] == 1.0


def test_all_equal_scores_rank_last():
    n = 50
    ranks, ties = rank(torch.zeros(1, n + 1), torch.tensor([7]), torch.tensor([[1, 2]]), n)
    assert ranks.item() == n - 2 and ties.item()


def test_out_of_vocabulary_target_is_a_miss():
    ranks, ties = rank(torch.rand(1, 11), torch.tensor([-1]), torch.tensor([[0]]), 10)
    assert ranks.item() == 11 and not ties.item()


def test_non_finite_scores_are_reported_as_such():
    scores = torch.rand(1, 11)
    scores[0, 4] = float("nan")
    with pytest.raises(NonFiniteScores):
        rank(scores, torch.tensor([1]), torch.tensor([[0]]), 10)


def test_matches_naive_reference_on_random_inputs():
    rng = np.random.default_rng(0)
    n, users = 300, 2000
    scores = rng.normal(size=(users, n + 1)).astype(np.float32)
    scores[:, 1::7] = 0.25  # deliberate ties
    history = np.stack([rng.choice(np.arange(1, n + 1), 12, replace=False) for _ in range(users)])
    targets = np.array([rng.choice(np.setdiff1d(np.arange(1, n + 1), h)) for h in history])
    ranks, _ = rank(
        torch.from_numpy(scores), torch.from_numpy(targets), torch.from_numpy(history), n
    )
    expected = [naive_rank(s, t, h) for s, t, h in zip(scores, targets, history, strict=True)]
    assert ranks.tolist() == expected


def test_random_scorer_matches_analytic_expectation():
    """Uniform random scores over C candidates: E[HR@K] = K/C, E[NDCG@K] = (1/C) sum 1/log2(r+1)."""
    rng = np.random.default_rng(1)
    c, k, users = 200, 10, 20_000
    scores = torch.from_numpy(rng.random((users, c + 1)).astype(np.float32))
    targets = torch.from_numpy(rng.integers(1, c + 1, size=users))
    ranks, _ = rank(scores, targets, torch.zeros(users, 1, dtype=torch.long), c)
    m = metrics_from_ranks(ranks.numpy(), ks=(k,))
    gains = np.array([1 / math.log2(r + 1) for r in range(1, k + 1)])
    expected_hr, expected_ndcg = k / c, gains.sum() / c
    se_hr = math.sqrt(expected_hr * (1 - expected_hr) / users)
    se_ndcg = math.sqrt(((gains**2).sum() / c - expected_ndcg**2) / users)
    assert abs(m[f"hr@{k}"] - expected_hr) < 4 * se_hr
    assert abs(m[f"ndcg@{k}"] - expected_ndcg) < 4 * se_ndcg


def test_rejects_low_precision_scores():
    with pytest.raises(InvariantError, match="fp32"):
        rank(
            torch.zeros(1, 6, dtype=torch.bfloat16),
            torch.tensor([1]),
            torch.zeros(1, 1, dtype=torch.long),
            5,
        )


@pytest.fixture(scope="module")
def validation():
    splits, _ = build_splits(build_log(random_frame()), "loo")
    return splits.validation


def _ordered_scorer(n):
    def scorer(inputs):
        return torch.arange(n + 1, dtype=torch.float32).repeat(len(inputs), 1)

    return scorer


def test_evaluate_masks_full_history_beyond_model_length(validation):
    n = validation.n_items
    result = evaluate(_ordered_scorer(n), validation, max_len=2)  # input truncated, history not
    assert result.users == len(validation)
    for i in range(0, len(validation), 7):
        history = set(validation.inputs.row(i).tolist())
        target = int(validation.targets[i])
        assert result.ranks[i] == len([j for j in range(target + 1, n + 1) if j not in history]) + 1


def test_model_inputs_are_the_last_items_left_padded(validation):
    seen = []

    def scorer(inputs):
        seen.append(inputs.clone())
        return torch.zeros(len(inputs), validation.n_items + 1)

    evaluate(scorer, validation, max_len=4, batch_size=5)
    inputs = torch.cat(seen)
    for i in range(len(validation)):
        row = validation.inputs.row(i)[-4:]
        expected = np.concatenate([np.zeros(4 - len(row), dtype=np.int64), row])
        assert inputs[i].tolist() == expected.tolist()


def test_batch_size_does_not_change_ranks(validation):
    scorer = _ordered_scorer(validation.n_items)
    small = evaluate(scorer, validation, max_len=3, batch_size=7)
    full = evaluate(scorer, validation, max_len=3, batch_size=10_000)
    assert np.array_equal(small.ranks, full.ranks)


def test_cross_check_with_ranx():
    ranx = pytest.importorskip("ranx")
    rng = np.random.default_rng(2)
    n, users = 100, 3000
    scores = rng.permutation(np.arange(1, (n + 1) * users + 1)).reshape(users, n + 1)
    scores = scores.astype(np.float32)  # distinct values: ranx breaks ties differently
    targets = rng.integers(1, n + 1, size=users)
    ranks, _ = rank(
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
