"""Committed plans parse, and every run's hyperparameters are accepted by its arm."""

from pathlib import Path

import pytest
import torch
from conftest import random_frame

from seqrec import models  # noqa: F401  registers the arms
from seqrec.data.preprocess import build_log
from seqrec.data.store import build_splits
from seqrec.runner.config import load_plan
from seqrec.runner.run import ARMS
from seqrec.runner.tune import load_tuning_plan

PLANS = Path(__file__).resolve().parents[1] / "configs" / "plans"


@pytest.fixture(scope="module")
def train():
    return build_splits(build_log(random_frame()), "loo")[0].train


@pytest.mark.parametrize("path", sorted(PLANS.glob("m2-*.toml")), ids=lambda p: p.name)
def test_plan_runs_build_their_models(path, train):
    if "tune" in path.name:
        _, _, studies = load_tuning_plan(path)
        assert studies
        return
    plan = load_plan(path)
    assert plan.runs
    seen = set()
    for cfg in plan.runs:
        key = (cfg.arm, str(sorted(cfg.params.items())))
        if key not in seen:
            seen.add(key)
            ARMS[cfg.arm](cfg, train, torch.device("cpu"))
    if plan.batch == "m2-anchor":
        assert {r.phase for r in plan.runs} == {"anchor"} and len(plan.runs) == 4
    else:
        assert {r.phase for r in plan.runs} == {"tune"}


def test_anchor_runs_train_without_the_start_rule():
    """The anchors follow their sources, which have no start rule (D035, D040)."""
    assert {r.start_limit for r in load_plan(PLANS / "m2-anchor.toml").runs} == {0}
