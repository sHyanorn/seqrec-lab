"""Known-answer tests on planted data, run end to end through the runner (METHODS 11.2, D027).

Each arm trains on CPU with a small configuration chosen for speed; the thresholds are
the pre-registered ones.
"""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from planted import planted_frame

from seqrec import models  # noqa: F401  registers the arms
from seqrec.data.preprocess import build_log
from seqrec.data.store import load_splits, log_path
from seqrec.eval.ranking import evaluate
from seqrec.runner import run as run_mod
from seqrec.runner.config import RunConfig

CPU = torch.device("cpu")
QUIET = {"log": lambda _: None}
LENGTH = 50
SASREC = {"dim": 32, "lr": 3e-3}
PLANTED = {  # arm: (params, max epochs, patience)
    "sasrec-ce": (SASREC, 60, 10),
    "sasrec-bce": (SASREC, 60, 10),
    "sasrec-ss": (SASREC | {"negatives": 16}, 60, 10),
    # With K/N large, gBCE first settles on a shared score offset before it learns the rule;
    # the reference gSASRec code shows the same on this data. K = 4 keeps the test fast.
    "gsasrec": (SASREC | {"negatives": 4, "t": 0.5}, 60, 10),
    "bert4rec": (
        {"dim": 32, "lr": 5e-3, "heads": 2, "blocks": 1, "dropout": 0.0, "mask_prob": 0.25},
        100,
        30,
    ),
}


def workspace(tmp_path_factory, name: str, p: float):
    root = tmp_path_factory.mktemp(name)
    build_log(planted_frame(users=1000, p=p)).save(log_path(root / "data", "ml-1m"))
    return root / "runs", root / "data"


def train_and_test(ws, arm, params, epochs, patience, shuffle="none", seed=1):
    runs, data = ws
    cfg = RunConfig(
        "b", "ml-1m", "loo", arm, seed, "final", shuffle=shuffle, params=params,
        max_epochs=epochs, patience=patience, max_len=LENGTH,
    )  # fmt: skip
    summary = run_mod.execute(cfg, runs, data, CPU, **QUIET)
    tested = run_mod.test_run(cfg, runs, data, CPU, **QUIET)
    return cfg, summary, tested["test"]


@pytest.fixture(scope="module")
def rule(tmp_path_factory):
    """Every transition follows the planted permutation (the literal rule of 11.2)."""
    return workspace(tmp_path_factory, "rule", 1.0)


@pytest.fixture(scope="module")
def noisy(tmp_path_factory):
    """Training transitions follow the rule with probability 0.2; targets always do (D027)."""
    return workspace(tmp_path_factory, "noisy", 0.2)


@pytest.fixture(scope="module")
def trained(rule):
    out = {arm: train_and_test(rule, arm, *spec) for arm, spec in PLANTED.items()}
    yield out
    load_splits.cache_clear()


@pytest.mark.parametrize("arm", PLANTED)
def test_planted_rule_is_learned(arm, trained, rule):
    cfg, summary, test = trained[arm]
    assert test["ndcg@10"] >= 0.95
    lines = (rule[0] / "b" / cfg.run_id / "metrics.jsonl").read_text().splitlines()
    best = max(json.loads(x)["val_ndcg@10"] for x in lines)
    assert summary["validation"]["ndcg@10"] == pytest.approx(best)  # best checkpoint restored
    run_dir = rule[0] / "b" / cfg.run_id
    assert set(torch.load(run_dir / "best.pt", weights_only=False)) == {"net"}  # weights only
    assert not (run_dir / "last.pt").exists()


def load_model(ws, cfg):
    splits, _ = run_mod._prepare_splits(cfg, ws[1])
    model = run_mod.ARMS[cfg.arm](cfg, splits.train, CPU)
    model.load_state_dict(torch.load(ws[0] / "b" / cfg.run_id / "best.pt", weights_only=False))
    return model, splits


def random_targets(split, n_items, rng):
    """For each user, a target drawn uniformly from the items outside their history."""
    targets = np.empty(len(split), dtype=np.int64)
    for u in range(len(split)):
        history = split.inputs.items[split.inputs.indptr[u] : split.inputs.indptr[u + 1]]
        targets[u] = rng.choice(np.setdiff1d(np.arange(1, n_items + 1), history))
    return targets


def test_random_targets_score_at_chance(trained, rule):
    model, splits = load_model(rule, trained["sasrec-ce"][0])
    test = splits.test.unseal("read")
    shuffled = replace(test, targets=random_targets(test, splits.n_items, np.random.default_rng(0)))
    got = evaluate(model.score, shuffled, LENGTH).metrics["ndcg@10"]
    gains = 1 / np.log2(np.arange(2, 12))
    c = splits.n_items - test.inputs.lengths()  # candidates per user
    mean = (gains.sum() / c).mean()
    var = ((gains**2).sum() / c - (gains.sum() / c) ** 2).mean()
    assert abs(got - mean) <= 4 * np.sqrt(var / len(c))


def test_shuffled_order_falls_to_popularity(noisy):
    _, _, ordered = train_and_test(noisy, "sasrec-ce", SASREC, 60, 10)
    _, _, shuffled = train_and_test(noisy, "sasrec-ce", SASREC, 60, 10, shuffle="train_eval")
    _, _, popular = train_and_test(noisy, "mostpop", {}, 1, 1)
    load_splits.cache_clear()
    assert ordered["ndcg@10"] >= 0.95
    assert shuffled["ndcg@10"] <= 0.10, (shuffled, popular)
    assert popular["ndcg@10"] <= 0.10  # so the threshold sits at the popularity level


def test_bert4rec_scored_without_its_mask_token_fails(trained, rule, monkeypatch):
    """Mutation: inference reads the last real item's output instead of an appended mask."""
    model, splits = load_model(rule, trained["bert4rec"][0])
    monkeypatch.setattr(model.net, "with_mask_token", lambda inputs: inputs)
    test = splits.test.unseal("read")
    assert evaluate(model.score, test, LENGTH).metrics["ndcg@10"] < 0.95
