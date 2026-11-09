import json

import optuna
import pytest
import torch
from conftest import random_frame

from seqrec import models  # noqa: F401  registers the arms
from seqrec.cli import main as cli_main
from seqrec.data.preprocess import build_log
from seqrec.data.store import load_splits, log_path
from seqrec.runner import run as run_mod
from seqrec.runner import tune
from seqrec.runner.export import export

CPU = torch.device("cpu")
QUIET = {"log": lambda _: None}


@pytest.fixture
def workspace(tmp_path):
    data = tmp_path / "data"
    build_log(random_frame()).save(log_path(data, "ml-1m"))
    plan = tmp_path / "tune.toml"
    plan.write_text(
        'batch = "t"\n[defaults]\ndataset = "ml-1m"\ntrials = 6\nmax_epochs = 3\npatience = 3\n'
        'max_len = 8\n[[studies]]\narm = "sasrec-ce"\n',
        encoding="utf-8",
    )
    load_splits.cache_clear()
    yield tmp_path, plan
    load_splits.cache_clear()


def run_plan(root, plan, runs="runs"):
    _, _, studies = tune.load_tuning_plan(plan)
    return tune.run_study(studies[0], root / runs, root / "data", CPU, **QUIET)[0]


def trial_params(opt):
    return [t.params for t in sorted(opt.trials, key=lambda t: t.number)]


def test_trials_are_tagged_tuning_runs_with_sealed_test_data(workspace):
    root, plan = workspace
    args = [str(plan), "--runs-dir", str(root / "runs"), "--data-dir", str(root / "data")]
    assert cli_main(["tune", *args, "--device", "cpu"]) == 0
    dirs = sorted((root / "runs" / "t").glob("sasrec-ce-*"))
    assert [d.name for d in dirs] == [f"sasrec-ce-ml-1m-loo-s0-t{i:02d}" for i in range(6)]
    for d in dirs:
        cfg = json.loads((d / "config.json").read_text())
        assert cfg["phase"] == "tune" and cfg["seed"] == 0
        assert set(cfg["params"]) == set(tune.SPACES["sasrec-ce"])
        assert not (d / "test_access.jsonl").exists()
        assert not (d / "last.pt").exists() and not (d / "best.pt").exists()  # disk space
        manifest = json.loads((d / "manifest.json").read_text())
        assert manifest["hparams"] == {"ce_chunk": 0} | cfg["params"]
        summary = json.loads((d / "summary.json").read_text())
        assert summary["epoch_train_s_median"] > 0 and summary["epoch_eval_s_median"] > 0
    out = root / "results"
    assert export(root / "runs", out) == []
    best = (out / "tuning_best.csv").read_text().splitlines()
    assert len(best) == 2 and ",6," in best[1]


def test_interrupted_study_resumes_with_the_same_proposals(workspace, monkeypatch):
    root, plan = workspace
    reference = trial_params(run_plan(root, plan, "uninterrupted"))
    original = tune.execute

    def crash_once(cfg, *args, report, **kw):
        def wrapped(epoch, value):
            if cfg.tag == "t02" and epoch == 1 and not crash_once.done:
                crash_once.done = True
                raise KeyboardInterrupt
            return report(epoch, value)

        return original(cfg, *args, report=wrapped, **kw)

    crash_once.done = False
    monkeypatch.setattr(tune, "execute", crash_once)
    with pytest.raises(KeyboardInterrupt):
        run_plan(root, plan)
    opt = run_plan(root, plan)
    assert len(opt.trials) == 6
    assert trial_params(opt) == reference
    assert sorted(opt.trials[2].intermediate_values) == [0, 1, 2]  # epoch 1 re-reported
    resumed = root / "runs" / "t" / "sasrec-ce-ml-1m-loo-s0-t02"
    assert json.loads((resumed / "config.json").read_text())["params"] == reference[2]
    epochs = [json.loads(x)["epoch"] for x in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert epochs == [0, 1, 2]


def test_trials_cannot_reach_test_data(workspace, monkeypatch):
    root, plan = workspace
    original = run_mod.evaluate

    def evaluate_test_instead(scorer, split, max_len, device):  # a bug that tunes on test
        splits, _ = run_mod._prepare_splits(
            run_mod.RunConfig("t", "ml-1m", "loo", "sasrec-ce", 0, "tune"), root / "data"
        )
        return original(scorer, splits.test.unseal("read"), max_len, device)

    monkeypatch.setattr(run_mod, "evaluate", evaluate_test_instead)
    _, _, studies = tune.load_tuning_plan(plan)
    opt, failed = tune.run_study(studies[0], root / "runs", root / "data", CPU, **QUIET)
    assert {t.state for t in opt.trials} == {optuna.trial.TrialState.FAIL} and len(failed) == 6
    error = (root / "runs" / "t" / "sasrec-ce-ml-1m-loo-s0-t00" / "error.txt").read_text()
    assert "tuning runs cannot access test data" in error


def test_pruned_trials_count_toward_the_budget(workspace, monkeypatch):
    root, plan = workspace
    monkeypatch.setattr(optuna.trial.Trial, "should_prune", lambda self: True)
    opt = run_plan(root, plan)
    assert len(opt.trials) == 6
    assert {t.state for t in opt.trials} == {optuna.trial.TrialState.PRUNED}
    summary = json.loads(
        (root / "runs" / "t" / "sasrec-ce-ml-1m-loo-s0-t05" / "summary.json").read_text()
    )
    assert summary["stop_reason"] == "pruned" and summary["epochs"] == 1


def test_a_trial_stored_without_all_parameters_fails_and_uses_budget(workspace):
    root, plan = workspace
    _, _, studies = tune.load_tuning_plan(plan)
    study = studies[0]
    opt = optuna.create_study(
        study_name=study.name, storage=tune._storage(root / "runs", "t"), direction="maximize"
    )
    opt.ask({"lr": tune.SPACES["sasrec-ce"]["lr"]})  # interrupted while storing parameters
    opt, failed = tune.run_study(study, root / "runs", root / "data", CPU, **QUIET)
    assert opt.trials[0].state == optuna.trial.TrialState.FAIL and len(opt.trials) == 6
    assert failed == ["sasrec-ce-ml-1m-loo-s0-t00"]
    assert not (root / "runs" / "t" / failed[0]).exists()  # it never trained


def test_pruner_waits_for_five_complete_trials_and_the_warm_up():
    opt = optuna.create_study(direction="maximize", pruner=tune.pruner(cap=300))

    def complete_trial():
        trial = opt.ask()
        for step in range(40):
            trial.report(1.0, step)
        opt.tell(trial, 1.0)

    def worse_trial_prunes_at(step: int) -> bool:
        trial = opt.ask()
        trial.report(0.0, step)
        pruned = trial.should_prune()
        opt.tell(trial, state=optuna.trial.TrialState.FAIL)
        return pruned

    for _ in range(4):
        complete_trial()
    assert not worse_trial_prunes_at(35)  # only 4 complete trials
    complete_trial()
    assert not worse_trial_prunes_at(29)  # inside the warm-up of 10% of the cap
    assert worse_trial_prunes_at(30)
