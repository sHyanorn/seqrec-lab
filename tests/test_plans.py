"""Committed plans parse, and every run's hyperparameters are accepted by its arm."""

import csv
from pathlib import Path

import pytest
import torch
from conftest import random_frame

from seqrec import models  # noqa: F401  registers the arms
from seqrec.data.preprocess import build_log
from seqrec.data.store import build_splits
from seqrec.runner.config import default_limits, load_plan, read_plan
from seqrec.runner.final_plan import final_plan
from seqrec.runner.run import ARMS
from seqrec.runner.tune import SPACES, load_tuning_plan

PLANS = Path(__file__).resolve().parents[1] / "configs" / "plans"
RESULTS = Path(__file__).resolve().parents[1] / "results"


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


def test_the_gsasrec_rerun_repeats_the_pilot_runs():
    """D035: the rerun differs from the pilot's gSASRec seed-variance runs only by its batch."""
    from dataclasses import replace

    rerun = load_plan(PLANS / "m2-gsasrec.toml").runs
    pilot = load_plan(PLANS / "m2-pilot.toml").runs
    assert [replace(r, batch="m2-pilot") for r in rerun] == [
        r for r in pilot if r.arm == "gsasrec" and r.tag == "sigma"
    ]


@pytest.mark.parametrize("dataset", ["ml-1m", "video-games"])
def test_a_tuning_plan_follows_the_registered_procedure(dataset):
    """METHODS 5.2 to 5.4: one study per neural arm, 20 trials, the arm's own limits, and a
    warm-up before the first trial."""
    batch, warmup_s, studies = load_tuning_plan(PLANS / f"m3-tune-{dataset}.toml")
    assert batch == f"m3-tune-{dataset}" and warmup_s == 180
    assert sorted(s.arm for s in studies) == sorted(SPACES)
    assert {(s.dataset, s.trials, s.sampler_seed) for s in studies} == {(dataset, 20, 0)}
    for s in studies:
        cfg = s.config(0, {})
        assert (cfg.max_epochs, cfg.patience) == default_limits(s.arm)
        assert (cfg.start_limit, cfg.guard_hours, cfg.max_len) == (300, 8.0, None)


@pytest.mark.parametrize("dataset", ["ml-1m", "video-games"])
def test_a_final_plan_is_what_its_tuning_studies_generate(dataset, train):
    """METHODS 5.5: the plan is the output of `seqrec final-plan` on the committed studies,
    for its number of leave-one-out seeds, and every arm accepts its parameters."""
    path = PLANS / f"m3-final-{dataset}.toml"
    plan = load_plan(path)
    seeds = max(r.seed for r in plan.runs)
    with (RESULTS / "runs" / "tuning_best.csv").open(encoding="utf-8", newline="") as f:
        best = list(csv.DictReader(f))
    expected = final_plan(best, dataset, f"m3-tune-{dataset}", seeds, f"m3-final-{dataset}")
    assert path.read_text(encoding="utf-8") == expected
    assert {r.phase for r in plan.runs} == {"final"}
    for cfg in {(r.arm, str(sorted(r.params.items()))): r for r in plan.runs}.values():
        ARMS[cfg.arm](cfg, train, torch.device("cpu"))


def test_every_tuning_plan_has_the_registered_trial_budget():
    """METHODS 5.4: 20 trials per study, 10 on MovieLens-20M, as budget.TRIALS holds them.
    The M2 batches were checks of the harness and the pilot (budget.PILOT_BATCH_PREFIX)."""
    from seqrec.analysis.budget import PILOT_BATCH_PREFIX, TRIALS

    assert TRIALS == {"ml-1m": 20, "video-games": 20, "steam": 20, "ml-20m": 10}
    checked = []
    for path in sorted(PLANS.glob("*.toml")):
        if "studies" not in read_plan(path):
            continue
        batch, _, studies = load_tuning_plan(path)
        if batch.startswith(PILOT_BATCH_PREFIX):
            continue
        assert {(s.dataset, s.trials) for s in studies} <= set(TRIALS.items()), path.name
        checked.append(path.name)
    assert {"m3-tune-ml-1m.toml", "m3-tune-video-games.toml"} <= set(checked)
