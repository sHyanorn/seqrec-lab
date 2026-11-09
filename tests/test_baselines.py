import numpy as np
import pytest
import torch
from conftest import random_frame
from test_batches import train_split

from seqrec.data.invariants import InvariantError
from seqrec.data.preprocess import build_log
from seqrec.data.store import load_splits, log_path
from seqrec.models import baselines
from seqrec.runner import run as run_mod
from seqrec.runner.config import RunConfig

CPU = torch.device("cpu")


def cfg(arm):
    return RunConfig("b", "ml-1m", "loo", arm, 1, "final", max_len=4)


def test_mostpop_scores_are_training_counts():
    train = train_split([[1, 2, 3], [2, 3], [3, 4]], 5)
    scores = baselines.MostPop(cfg("mostpop"), train, CPU).score(torch.tensor([[0, 0, 1, 2]]))
    assert scores.tolist() == [[0, 1, 2, 3, 1, 0]]


def test_transition_counts_pairs_and_breaks_ties_by_popularity():
    train = train_split([[1, 2, 3], [1, 2], [4, 3], [5, 1, 4], [6, 5]], 6)
    model = baselines.Transition(cfg("transition"), train, CPU)
    inputs = torch.tensor([[0, 0, 3, 1], [0, 0, 0, 6], [0, 0, 0, 2]])
    s = model.score(inputs)
    assert s.dtype == torch.float32
    # after 1: item 2 (count 2), item 4 (count 1), then zero-count items by popularity
    # (1: 3, 3: 2, 5: 2, 6: 1); equal count and popularity stay tied
    assert s[0].argsort(descending=True).tolist()[:2] == [2, 4]
    assert s[0, 1] > s[0, 3] == s[0, 5] > s[0, 6]
    assert s[1].argmax().item() == 5 and s[2].argmax().item() == 3


def test_dense_rank_keeps_ties_and_order():
    key = torch.tensor([[5, 1, 5, 9, 1], [10**12, 3, 10**12 + 1, 0, 3]])
    ranks = baselines.dense_rank_rows(key)
    assert ranks.tolist() == [[2.0, 1.0, 2.0, 3.0, 1.0], [3.0, 2.0, 4.0, 1.0, 2.0]]


def test_statistics_must_come_from_the_runs_training_split():
    train = train_split([[1, 2, 3], [2, 3]], 4)
    other = train_split([[1, 2, 3], [2, 4]], 4)
    for arm in (baselines.MostPop, baselines.Transition):
        model = arm(cfg("x"), train, CPU)
        model.train_epoch(train, 0)
        with pytest.raises(InvariantError, match="not computed from this training split"):
            model.train_epoch(other, 0)


def test_baselines_stop_after_one_epoch_in_the_runner(tmp_path):
    data = tmp_path / "data"
    build_log(random_frame()).save(log_path(data, "ml-1m"))
    load_splits.cache_clear()
    for arm in ("mostpop", "transition"):
        c = RunConfig("b", "ml-1m", "loo", arm, 1, "final")
        summary = run_mod.execute(c, tmp_path / "runs", data, CPU, log=lambda _: None)
        assert summary["stop_reason"] == "converged" and summary["epochs"] == 1
        assert not summary["truncated"] and np.isfinite(summary["validation"]["ndcg@10"])
    load_splits.cache_clear()
