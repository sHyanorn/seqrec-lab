import json

import optuna
import pytest
import torch
from conftest import random_frame

from seqrec import models  # noqa: F401  registers the arms
from seqrec.cli import main as cli_main
from seqrec.data.preprocess import build_log
from seqrec.data.store import load_splits, log_path
from seqrec.runner import config as config_mod
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
    monkeypatch.setattr(run_mod, "START_FACTOR", -1.0)  # every trial starts at epoch 0 (D044)
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


def test_a_trial_that_never_starts_completes_without_pruning_steps(workspace, monkeypatch):
    """D044: before a trial beats 1.5 times MostPop (D035) it reports nothing, so it cannot
    be pruned and has no value in the median at any step."""
    root, plan = workspace
    monkeypatch.setattr(run_mod, "START_FACTOR", 1e9)
    opt = run_plan(root, plan)
    assert {t.state for t in opt.trials} == {optuna.trial.TrialState.COMPLETE}
    assert all(t.intermediate_values == {} for t in opt.trials)
    summary = json.loads(
        (root / "runs" / "t" / "sasrec-ce-ml-1m-loo-s0-t05" / "summary.json").read_text()
    )
    assert summary["start_epoch"] is None and summary["epochs"] == 3


def test_a_trial_stopped_at_its_start_limit_is_a_complete_trial(workspace, monkeypatch):
    """D044: a trial that ends with `no_start` counts toward the trial budget and the 5 trials
    that enable pruning, with the value of its best checkpoint and no pruning step."""
    root, plan = workspace
    monkeypatch.setattr(run_mod, "START_FACTOR", 1e9)
    monkeypatch.setattr(config_mod, "default_start_limit", lambda arm: 2)
    opt = run_plan(root, plan)
    assert {t.state for t in opt.trials} == {optuna.trial.TrialState.COMPLETE}
    assert all(t.intermediate_values == {} and t.value > 0 for t in opt.trials)
    for number in range(len(opt.trials)):
        run_dir = root / "runs" / "t" / f"sasrec-ce-ml-1m-loo-s0-t{number:02d}"
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["stop_reason"] == "no_start" and summary["epochs"] == 2


def test_trials_that_never_started_enable_pruning_but_stay_out_of_the_median():
    """Optuna's median is taken over the complete trials that reported the step; the count
    of 5 complete trials includes those that reported nothing (D044)."""

    def study(reporting: int, silent: int) -> optuna.Study:
        opt = optuna.create_study(direction="maximize", pruner=tune.pruner(cap=300))
        for i in range(reporting + silent):
            trial = opt.ask()
            if i < reporting:
                for step in range(40):
                    trial.report(1.0, step)
            opt.tell(trial, 1.0 if i < reporting else 0.0)
        return opt

    def middling_trial_is_pruned(opt) -> bool:
        trial = opt.ask()
        trial.report(0.5, 30)
        return trial.should_prune()

    assert not middling_trial_is_pruned(study(reporting=3, silent=1))  # 4 complete trials
    # with the silent trials at their final 0.0 the median would be 0.0 and nothing pruned
    assert middling_trial_is_pruned(study(reporting=3, silent=2))
    assert not middling_trial_is_pruned(study(reporting=0, silent=5))  # no value at the step
    # fewer than 3 complete trials with a value at the step are no median to prune against
    assert not middling_trial_is_pruned(study(reporting=1, silent=4))
    assert not middling_trial_is_pruned(study(reporting=2, silent=3))


def _crash_once(monkeypatch, tag):
    original = tune.execute
    state = {"done": False}

    def crash(cfg, *args, **kw):
        if cfg.tag == tag and not state["done"]:
            state["done"] = True
            raise RuntimeError("simulated out-of-memory")
        return original(cfg, *args, **kw)

    monkeypatch.setattr(tune, "execute", crash)


def test_a_failed_trial_is_retried_once_without_using_budget(workspace, monkeypatch):
    """D031: the retry happens before the trial is told to Optuna."""
    root, plan = workspace
    reference = run_plan(root, plan, "uninterrupted")
    _crash_once(monkeypatch, "t01")
    opt = run_plan(root, plan)
    assert [t.state for t in opt.trials] == [t.state for t in reference.trials]  # no FAIL
    assert trial_params(opt) == trial_params(reference)
    failures = root / "runs" / "t" / "sasrec-ce-ml-1m-loo-s0-t01" / "failures.jsonl"
    assert len(failures.read_text().splitlines()) == 1


def test_a_lost_device_leaves_the_trial_to_resume_on_rerun(workspace, monkeypatch):
    from seqrec.runner import attempts
    from seqrec.runner.batch import tune_batch

    root, plan = workspace
    _crash_once(monkeypatch, "t02")
    monkeypatch.setattr(attempts, "device_usable", lambda device: False)
    failed = tune_batch(plan, root / "runs", root / "data", CPU, **QUIET)
    assert failed == ["sasrec-ce-ml-1m-loo-s0-t02"]
    _, _, studies = tune.load_tuning_plan(plan)
    storage = tune._storage(root / "runs", "t")
    opt = optuna.load_study(study_name=studies[0].name, storage=storage)
    assert [t.state.name for t in opt.trials][-1] == "RUNNING"
    monkeypatch.setattr(attempts, "device_usable", lambda device: True)
    opt = run_plan(root, plan)
    assert len(opt.trials) == 6
    assert optuna.trial.TrialState.FAIL not in {t.state for t in opt.trials}
    t02 = root / "runs" / "t" / "sasrec-ce-ml-1m-loo-s0-t02" / "failures.jsonl"
    assert len(t02.read_text().splitlines()) == 1


def test_a_config_conflict_in_one_study_is_reported_and_others_run(workspace):
    from seqrec.runner.batch import tune_batch

    root, plan = workspace
    plan.write_text(plan.read_text() + '[[studies]]\narm = "sasrec-bce"\n')
    _, _, studies = tune.load_tuning_plan(plan)
    clash = studies[0].config(0, {})
    run_mod._claim(clash, run_mod.run_dir_of(clash, root / "runs"))
    failed = tune_batch(plan, root / "runs", root / "data", CPU, **QUIET)
    assert failed == ["sasrec-ce-ml-1m"]
    assert not (run_mod.run_dir_of(clash, root / "runs") / "failures.jsonl").exists()
    assert len(list((root / "runs" / "t").glob("sasrec-bce-*"))) == 6


def test_export_with_a_tuning_plan_refuses_until_every_study_reached_its_budget(workspace):
    root, plan = workspace
    out = root / "results"
    export_plan = ["export", "--runs-dir", str(root / "runs"), "--out-dir", str(out),
                   "--plan", str(plan)]  # fmt: skip
    assert cli_main(export_plan) == 1  # no study exists yet
    assert not (root / "runs").exists()  # the check creates no study database
    shorter = root / "shorter.toml"
    shorter.write_text(plan.read_text().replace("trials = 6", "trials = 3"), encoding="utf-8")
    run_plan(root, shorter)
    assert cli_main(export_plan) == 1  # 3 of 6 trials
    assert not out.exists()
    opt = run_plan(root, plan)
    assert cli_main(export_plan) == 0
    opt.ask()  # a trial left running, as after an interruption
    assert cli_main(export_plan) == 1
