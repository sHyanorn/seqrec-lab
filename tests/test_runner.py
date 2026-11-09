import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from conftest import random_frame

from seqrec.cli import main as cli_main
from seqrec.data.invariants import item_popularity
from seqrec.data.preprocess import build_log
from seqrec.data.store import load_splits, log_path
from seqrec.eval import test_access
from seqrec.runner import run as run_mod
from seqrec.runner.config import RunConfig
from seqrec.runner.export import export

CPU = torch.device("cpu")
QUIET = {"log": lambda _: None}


class PopularityModel:
    """Test double: popularity plus fixed noise. With `peak` set, only that epoch scores well,
    so restoring any other checkpoint changes the reported validation score."""

    peak: int | None = None
    crash: str | None = None  # "train" raises inside an epoch

    def __init__(self, cfg, train, device):
        counts = item_popularity(train).counts.astype(np.float32)
        noise = np.random.default_rng(cfg.seed).random(len(counts)).astype(np.float32)
        self.good = torch.from_numpy(counts + 1e-3 * noise)
        self.bad = torch.from_numpy(noise)  # random scores
        self.epochs = 0

    def train_epoch(self, train, epoch):
        if PopularityModel.crash == f"train@{epoch}":
            PopularityModel.crash = None
            raise RuntimeError("simulated crash")
        self.epochs = epoch + 1
        return {"loss": 0.0}

    def _scores(self):
        if PopularityModel.peak is None or self.epochs == PopularityModel.peak + 1:
            return self.good
        return self.bad

    def score(self, inputs):
        return self._scores().repeat(len(inputs), 1)

    def state_dict(self):
        return {"epochs": self.epochs}

    def load_state_dict(self, state):
        self.epochs = state["epochs"]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setitem(run_mod.ARMS, "pop", PopularityModel)
    monkeypatch.setattr(PopularityModel, "peak", None)
    monkeypatch.setattr(PopularityModel, "crash", None)
    data_dir = tmp_path / "data"
    build_log(random_frame()).save(log_path(data_dir, "ml-1m"))
    load_splits.cache_clear()
    yield tmp_path / "runs", data_dir
    load_splits.cache_clear()


def _cfg(phase="final", **kw):
    base = dict(batch="b", dataset="ml-1m", split="loo", arm="pop", seed=1, phase=phase)
    return RunConfig(**(base | {"max_epochs": 50, "patience": 3} | kw))


def test_training_reads_no_test_data_and_testing_reads_once(workspace):
    runs, data = workspace
    summary = run_mod.execute(_cfg(), runs, data, CPU, **QUIET)
    assert summary["stop_reason"] == "patience" and summary["epochs"] == 4
    run_dir = runs / "b" / _cfg().run_id
    assert not (run_dir / "test_access.jsonl").exists()
    out = run_mod.test_run(_cfg(), runs, data, CPU, **QUIET)
    assert 0 <= out["test"]["ndcg@10"] <= 1 and "ndcg@10" in out["test_inputs_shuffled"]
    assert test_access.count_reads(run_dir / "test_access.jsonl") == {
        "final": 1, "anchor": 0, "completed": 1,
    }  # fmt: skip
    assert run_mod.test_run(_cfg(), runs, data, CPU, **QUIET) == out  # no second read


def test_best_checkpoint_is_the_one_scored(workspace):
    runs, data = workspace
    PopularityModel.peak = 1
    summary = run_mod.execute(_cfg(), runs, data, CPU, **QUIET)
    assert summary["best_epoch"] == 1
    lines = (runs / "b" / _cfg().run_id / "metrics.jsonl").read_text().splitlines()
    best = max(json.loads(x)["val_ndcg@10"] for x in lines)
    assert summary["validation"]["ndcg@10"] == pytest.approx(best)


def test_tuning_run_cannot_reach_test_data(workspace, monkeypatch):
    runs, data = workspace
    original = run_mod.evaluate

    def evaluate_test_instead(scorer, split, max_len, device):  # a bug that tunes on test
        splits, _ = run_mod._prepare_splits(_cfg("tune"), data)
        return original(scorer, splits.test.unseal("read"), max_len, device)

    monkeypatch.setattr(run_mod, "evaluate", evaluate_test_instead)
    with pytest.raises(PermissionError, match="tuning runs cannot access test data"):
        run_mod.execute(_cfg("tune"), runs, data, CPU, **QUIET)
    with pytest.raises(PermissionError):
        run_mod.test_run(_cfg("tune"), runs, data, CPU, **QUIET)


def test_interrupted_run_resumes_without_duplicate_epochs(workspace):
    runs, data = workspace
    PopularityModel.crash = "train@2"
    with pytest.raises(RuntimeError):
        run_mod.execute(_cfg(), runs, data, CPU, **QUIET)
    messages = []
    summary = run_mod.execute(_cfg(), runs, data, CPU, log=messages.append)
    assert any("resuming at epoch 2" in m for m in messages)
    lines = (runs / "b" / _cfg().run_id / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(x)["epoch"] for x in lines] == [0, 1, 2, 3]
    assert summary["epochs"] == 4


def test_changed_configuration_is_refused(workspace):
    runs, data = workspace
    run_mod.execute(_cfg(), runs, data, CPU, **QUIET)
    with pytest.raises(run_mod.RunConflict):
        run_mod.execute(replace(_cfg(), params={"lr": 0.5}), runs, data, CPU, **QUIET)


def test_anchor_reads_are_capped(workspace):
    runs, data = workspace
    for tag in "abc":
        cfg = _cfg("anchor", tag=tag)
        run_mod.execute(cfg, runs, data, CPU, **QUIET)
        run_mod.test_run(cfg, runs, data, CPU, **QUIET)
    fourth = _cfg("anchor", tag="d")
    run_mod.execute(fourth, runs, data, CPU, **QUIET)
    with pytest.raises(PermissionError, match="anchor reads"):
        run_mod.test_run(fourth, runs, data, CPU, **QUIET)


def test_test_reader_rejects_other_phases(tmp_path):
    with pytest.raises(PermissionError):
        test_access.read_test(
            None, phase="tune", log_path=tmp_path / "t.jsonl", run_id="r", git_commit="c",
            checkpoint_hash="h",
        )  # fmt: skip


def test_batches_through_cli(workspace):
    runs, data = workspace
    plan = runs.parent / "plan.toml"
    plan.write_text(
        '﻿batch = "b"\norder = "shuffled"\n[defaults]\ndataset = "ml-1m"\nsplit = "loo"\n'
        'phase = "final"\nmax_epochs = 10\npatience = 2\n'
        '[[runs]]\narm = "pop"\nseeds = [1, 2]\n'
        '[[runs]]\narm = "pop"\nsplit = "gts"\nseeds = [1]\n'
        '[[runs]]\narm = "pop"\nphase = "tune"\nseeds = [3]\n'
        '[[runs]]\narm = "pop"\nshuffle = "train_eval"\nseeds = [1]\n',
        encoding="utf-8",
    )
    args = [str(plan), "--runs-dir", str(runs), "--data-dir", str(data), "--device", "cpu"]
    with pytest.raises(RuntimeError, match="not trained yet"):
        cli_main(["test", *args])
    assert cli_main(["run", *args]) == 0
    assert cli_main(["test", *args]) == 0
    out = runs.parent / "results"
    assert cli_main(["export", "--runs-dir", str(runs), "--out-dir", str(out)]) == 0
    assert len((out / "b.csv").read_text().splitlines()) == 1 + 5
    assert (out / "overview.csv").exists()

    tune_dir = next((runs / "b").glob("*s3*"))
    (tune_dir / "test_access.jsonl").write_text(json.dumps({"phase": "final"}) + "\n")
    assert any("tuning run read test data" in p for p in export(runs, out))


def test_crash_after_a_checkpoint_keeps_every_epoch_logged(workspace, monkeypatch):
    runs, data = workspace
    original = run_mod._save_atomic
    calls = {"last": 0}

    def crash_on_third_checkpoint(obj, path):
        original(obj, path)
        if path.name == "last.pt":
            calls["last"] += 1
            if calls["last"] == 3:
                raise RuntimeError("simulated crash right after the checkpoint")

    monkeypatch.setattr(run_mod, "_save_atomic", crash_on_third_checkpoint)
    with pytest.raises(RuntimeError):
        run_mod.execute(_cfg(), runs, data, CPU, **QUIET)
    summary = run_mod.execute(_cfg(), runs, data, CPU, **QUIET)
    lines = (runs / "b" / _cfg().run_id / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(x)["epoch"] for x in lines] == list(range(summary["epochs"]))
    assert not (runs / "b" / _cfg().run_id / "last.pt").exists()


def _trial_row(tag, ndcg, stop="patience", status="trained", truncated=False):
    return {
        "batch": "t", "dataset": "ml-1m", "split": "loo", "arm": "sasrec-ce", "phase": "tune",
        "tag": tag, "run_id": f"r-{tag}", "status": status, "stop_reason": stop,
        "truncated": truncated, "validation_ndcg@10": ndcg, "params": "{}", "gpu_hours": 0.1,
    }  # fmt: skip


def test_best_trial_skips_pruned_runs_and_breaks_ties_by_trial_number():
    from seqrec.runner.export import tuning_best

    rows = [
        _trial_row("t99", 0.20),
        _trial_row("t100", 0.20),
        _trial_row("t07", 0.25, stop="pruned"),
        _trial_row("t08", None, stop=None, status="failed", truncated=None),
        _trial_row("t09", 0.10, stop="max_epochs", truncated=True),
        _trial_row("t10", 0.05, stop="diverged"),
        _trial_row("t11", None, stop="diverged", status="diverged"),
    ]
    best = tuning_best(rows)
    assert len(best) == 1 and best[0]["best_run_id"] == "r-t99"
    assert best[0]["trials"] == 7 and best[0]["pruned"] == 1 and best[0]["diverged"] == 2
    assert best[0]["failed_or_unfinished"] == 1


def test_truncated_share_counts_only_runs_that_stopped_by_patience_or_a_limit():
    """D030: pruned, diverged and failed runs stop before any limit applies."""
    from seqrec.runner.export import _overview, tuning_best

    rows = [_trial_row(f"t{i:02d}", 0.2) for i in range(12)]
    rows += [_trial_row(f"t{i}", 0.1, stop="max_epochs", truncated=True) for i in (12, 13, 14)]
    rows.append(_trial_row("t15", 0.1, stop="guard", truncated=True))
    rows += [_trial_row(f"t{i}", 0.1, stop="pruned") for i in (16, 17, 18)]
    rows.append(_trial_row("t19", 0.1, stop="diverged"))  # 4 of 16 stopped runs, not 4 of 17
    best = tuning_best(rows)[0]
    assert best["stopped"] == 16 and best["truncated"] == 4 and best["truncated_share"] == 0.25
    overview = _overview(rows)[0]
    assert overview["stopped"] == 16 and overview["truncated_share"] == 0.25


def _plan(root, seeds=(1,), phase="tune"):
    plan = root / "plan.toml"
    plan.write_text(
        f'batch = "b"\n[defaults]\ndataset = "ml-1m"\nsplit = "loo"\nphase = "{phase}"\n'
        f'max_epochs = 6\npatience = 2\n[[runs]]\narm = "pop"\nseeds = {list(seeds)}\n',
        encoding="utf-8",
    )
    return plan


def _exported(runs, root):
    import csv

    export(runs, root / "results")
    with (root / "results" / "b.csv").open(encoding="utf-8") as f:
        return {r["run_id"]: r for r in csv.DictReader(f)}


def _failures(run_dir):
    path = run_dir / "failures.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_a_failed_run_is_retried_once_and_exported_as_trained(workspace):
    """D031: one retry, recorded; the retried run still counts as trained (e.g. for sigma)."""
    from seqrec.runner.batch import run_batch

    runs, data = workspace
    PopularityModel.crash = "train@1"
    assert run_batch(_plan(runs.parent), runs, data, CPU, **QUIET) == []
    run_dir = runs / "b" / _cfg("tune").run_id
    assert [f["step"] for f in _failures(run_dir)] == ["train"]
    assert "simulated crash" in _failures(run_dir)[0]["error"]
    row = _exported(runs, runs.parent)[_cfg("tune").run_id]
    assert row["status"] == "trained" and row["failures"] == "1"


def test_a_run_that_fails_twice_is_not_retried_again(workspace, monkeypatch):
    from seqrec.runner.batch import run_batch

    runs, data = workspace

    def always_fail(self, train, epoch):
        raise RuntimeError("deterministic failure")

    monkeypatch.setattr(PopularityModel, "train_epoch", always_fail)
    run_id = _cfg("tune").run_id
    plan = _plan(runs.parent)
    assert run_batch(plan, runs, data, CPU, **QUIET) == [run_id]
    assert run_batch(plan, runs, data, CPU, **QUIET) == [run_id]  # rerun: skipped
    assert len(_failures(runs / "b" / run_id)) == 2
    row = _exported(runs, runs.parent)[run_id]
    assert row["status"] == "failed" and row["failures"] == "2"


def test_a_lost_device_stops_the_batch_and_the_rerun_retries(workspace, monkeypatch):
    from seqrec.runner import attempts
    from seqrec.runner.batch import run_batch

    runs, data = workspace
    PopularityModel.crash = "train@1"
    monkeypatch.setattr(attempts, "device_usable", lambda device: False)
    plan = _plan(runs.parent, seeds=(1, 2))
    first, second = (_cfg("tune", seed=s).run_id for s in (1, 2))
    assert run_batch(plan, runs, data, CPU, **QUIET) == [first]
    assert not (runs / "b" / first / "summary.json").exists()
    assert not (runs / "b" / second).exists()  # the batch stopped
    monkeypatch.setattr(attempts, "device_usable", lambda device: True)
    assert run_batch(plan, runs, data, CPU, **QUIET) == []
    rows = _exported(runs, runs.parent)
    assert rows[first]["status"] == "trained" and rows[first]["failures"] == "1"
    assert rows[second]["failures"] == "0"


def test_a_final_run_whose_test_read_failed_is_not_read_again(workspace, monkeypatch):
    """METHODS 11.6: a final run reads test data once, even if that read did not finish."""
    runs, data = workspace
    original = run_mod.evaluate
    final, anchor = _cfg("final"), _cfg("anchor", tag="a")
    for cfg in (final, anchor):
        run_mod.execute(cfg, runs, data, CPU, **QUIET)

        def fail_on_test(scorer, split, max_len, device):
            raise RuntimeError("simulated failure while scoring")

        monkeypatch.setattr(run_mod, "evaluate", fail_on_test)
        with pytest.raises(RuntimeError, match="simulated"):
            run_mod.test_run(cfg, runs, data, CPU, **QUIET)
        monkeypatch.setattr(run_mod, "evaluate", original)
    with pytest.raises(PermissionError, match="did not complete"):
        run_mod.test_run(final, runs, data, CPU, **QUIET)
    log = runs / "b" / final.run_id / "test_access.jsonl"
    assert test_access.count_reads(log) == {"final": 1, "anchor": 0, "completed": 0}
    run_mod.test_run(anchor, runs, data, CPU, **QUIET)  # anchor: counted, allowed
    problems = export(runs, runs.parent / "results")
    assert sum("a test read did not complete" in p for p in problems) == 2


def test_a_pruned_stop_survives_a_failure_before_the_summary(workspace, monkeypatch):
    """A retry after the loop has ended must not train further epochs."""
    runs, data = workspace
    original = run_mod._save_atomic
    calls = {"best": 0}

    def fail_best_once(obj, path):
        if path.name == "best.pt" and not calls["best"]:
            calls["best"] += 1
            raise OSError("simulated failure while saving best.pt")
        original(obj, path)

    monkeypatch.setattr(run_mod, "_save_atomic", fail_best_once)
    cfg = _cfg("tune")

    def prune_after_epoch_1(epoch, value):
        return epoch >= 1

    with pytest.raises(OSError):
        run_mod.execute(cfg, runs, data, CPU, report=prune_after_epoch_1, **QUIET)
    summary = run_mod.execute(cfg, runs, data, CPU, report=prune_after_epoch_1, **QUIET)
    assert summary["stop_reason"] == "pruned" and summary["epochs"] == 2


def test_cleanup_after_the_summary_is_best_effort(workspace, monkeypatch):
    runs, data = workspace
    original = run_mod.Path.unlink

    def locked(self, missing_ok=False):
        if self.name == "best.pt":
            raise PermissionError("file in use")
        return original(self, missing_ok=missing_ok)

    monkeypatch.setattr(run_mod.Path, "unlink", locked)
    summary = run_mod.execute(_cfg("tune"), runs, data, CPU, **QUIET)
    assert summary["status"] == "trained" and summary["checkpoint_s"] >= 0


def test_a_transient_lock_on_replace_is_retried(tmp_path, monkeypatch):
    original, calls = run_mod.os.replace, {"n": 0}

    def locked_once(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("file in use")
        original(src, dst)

    monkeypatch.setattr(run_mod.os, "replace", locked_once)
    monkeypatch.setattr(run_mod.time, "sleep", lambda s: None)
    run_mod._write_atomic(tmp_path / "x.json", "{}")
    assert (tmp_path / "x.json").read_text() == "{}" and calls["n"] == 2


def test_a_config_conflict_is_not_a_failed_attempt(workspace):
    from seqrec.runner.batch import run_batch

    runs, data = workspace
    run_mod._claim(_cfg("tune"), runs / "b" / _cfg("tune").run_id)
    plan = _plan(runs.parent)
    plan.write_text(plan.read_text().replace('arm = "pop"', 'arm = "pop"\npatience = 3'))
    with pytest.raises(run_mod.RunConflict, match="differs from 1 existing run"):
        run_batch(plan, runs, data, CPU, **QUIET)
    assert _failures(runs / "b" / _cfg("tune").run_id) == []


def test_an_edited_plan_is_refused_before_any_test_read(workspace):
    """METHODS 5.5: no test read while a run of the plan can still change."""
    from seqrec.runner.batch import run_batch, test_batch

    runs, data = workspace
    plan = _plan(runs.parent, seeds=(1, 2), phase="final")
    assert run_batch(plan, runs, data, CPU, **QUIET) == []
    plan.write_text(plan.read_text().replace("seeds = [1, 2]", "seeds = [1, 2]\npatience = 3"))
    with pytest.raises(run_mod.RunConflict):
        test_batch(plan, runs, data, CPU, **QUIET)
    assert not any(runs.rglob("test_access.jsonl"))


def test_a_prune_decision_is_saved_with_the_epoch_it_follows(workspace, monkeypatch):
    runs, data = workspace
    original, calls = run_mod._save_atomic, {"last": 0}

    def fail_second_checkpoint(obj, path):
        if path.name == "last.pt":
            calls["last"] += 1
            if calls["last"] == 2:
                raise OSError("simulated failure while saving last.pt")
        original(obj, path)

    monkeypatch.setattr(run_mod, "_save_atomic", fail_second_checkpoint)
    cfg = _cfg("tune")

    def prune_after_epoch_1(epoch, value):
        return epoch >= 1

    with pytest.raises(OSError):
        run_mod.execute(cfg, runs, data, CPU, report=prune_after_epoch_1, **QUIET)
    summary = run_mod.execute(cfg, runs, data, CPU, report=prune_after_epoch_1, **QUIET)
    assert summary["stop_reason"] == "pruned" and summary["epochs"] == 2


def test_retries_end_even_when_a_finished_run_keeps_failing(tmp_path):
    from seqrec.runner.attempts import TrainingFailed, train_with_retry

    (tmp_path / "summary.json").write_text("")  # unreadable, e.g. after a power loss

    def train():
        raise ValueError("cannot read summary.json")

    with pytest.raises(TrainingFailed):
        train_with_retry(train, tmp_path, CPU, lambda _: None)
    assert len(_failures(tmp_path)) == 2


def test_a_file_error_after_the_test_read_started_is_recorded(workspace, monkeypatch):
    from seqrec.runner.batch import run_batch, test_batch

    runs, data = workspace
    plan = _plan(runs.parent, phase="final")
    assert run_batch(plan, runs, data, CPU, **QUIET) == []

    def locked(log_path, run_id):
        raise PermissionError("file in use")

    monkeypatch.setattr(run_mod, "mark_completed", locked)
    assert test_batch(plan, runs, data, CPU, **QUIET) == [_cfg().run_id]
    assert [f["step"] for f in _failures(runs / "b" / _cfg().run_id)] == ["test"]


def test_a_new_failure_after_a_torn_record_is_kept(tmp_path):
    from seqrec.runner.attempts import failures, record_failure

    (tmp_path / "failures.jsonl").write_text('{"step": "train"}\n{"step": "tr')
    try:
        raise RuntimeError("x")
    except RuntimeError:
        record_failure(tmp_path, "train")
    assert [f["step"] for f in failures(tmp_path)] == ["train", "train"]


def test_a_missing_dataset_stops_the_batch_before_any_attempt(workspace):
    from seqrec.runner.batch import run_batch

    runs, data = workspace
    plan = _plan(runs.parent)
    plan.write_text(plan.read_text().replace('"ml-1m"', '"steam"'))
    with pytest.raises(FileNotFoundError, match="steam"):
        run_batch(plan, runs, data, CPU, **QUIET)
    assert not runs.exists() or not any(runs.rglob("failures.jsonl"))


def test_a_lost_device_after_the_final_failure_still_stops_the_batch(workspace, monkeypatch):
    from seqrec.runner import attempts
    from seqrec.runner.batch import run_batch

    runs, data = workspace

    def always_fail(self, train, epoch):
        raise RuntimeError("deterministic failure")

    checks = iter([True, False])
    monkeypatch.setattr(PopularityModel, "train_epoch", always_fail)
    monkeypatch.setattr(attempts, "device_usable", lambda device: next(checks))
    first, second = (_cfg("tune", seed=s).run_id for s in (1, 2))
    assert run_batch(_plan(runs.parent, seeds=(1, 2)), runs, data, CPU, **QUIET) == [first]
    assert len(_failures(runs / "b" / first)) == 2
    assert not (runs / "b" / second).exists()


def test_testing_skips_runs_that_failed_twice_or_diverged(workspace, monkeypatch):
    from seqrec.runner.batch import run_batch, test_batch

    runs, data = workspace
    plan = _plan(runs.parent, seeds=(1, 2, 3), phase="final")
    failed, diverged = (runs / "b" / _cfg(seed=s).run_id for s in (1, 3))
    run_mod._claim(_cfg(seed=1, max_epochs=6, patience=2), failed)  # as in the plan
    (failed / "failures.jsonl").write_text('{"step": "train"}\n{"step": "train"}\n')
    original = PopularityModel.score

    def nan_for_seed_3(self, inputs):
        scores = original(self, inputs)
        return scores * float("nan") if self.seed == 3 else scores

    monkeypatch.setattr(PopularityModel, "seed", 0, raising=False)
    original_init = PopularityModel.__init__

    def init(self, cfg, train, device):
        original_init(self, cfg, train, device)
        self.seed = cfg.seed

    monkeypatch.setattr(PopularityModel, "__init__", init)
    monkeypatch.setattr(PopularityModel, "score", nan_for_seed_3)
    assert run_batch(plan, runs, data, CPU, **QUIET) == [_cfg(seed=1).run_id]
    assert test_batch(plan, runs, data, CPU, **QUIET) == []
    rows = _exported(runs, runs.parent)
    assert rows[_cfg(seed=2).run_id]["status"] == "tested"
    assert rows[_cfg(seed=3).run_id]["status"] == "diverged"
    assert rows[_cfg(seed=3).run_id]["failures"] == "0"
    assert rows[_cfg(seed=1).run_id]["status"] == "failed"


def test_a_refused_test_read_is_not_recorded_as_a_failure(workspace):
    from seqrec.runner.batch import test_batch

    runs, data = workspace
    for tag in "abc":
        cfg = _cfg("anchor", tag=tag)
        run_mod.execute(cfg, runs, data, CPU, **QUIET)
        run_mod.test_run(cfg, runs, data, CPU, **QUIET)
    plan = runs.parent / "anchor.toml"
    plan.write_text(
        'batch = "b"\n[defaults]\ndataset = "ml-1m"\nsplit = "loo"\nphase = "anchor"\n'
        'max_epochs = 50\npatience = 3\n[[runs]]\narm = "pop"\nseeds = [1]\ntag = "d"\n',
        encoding="utf-8",
    )
    run_mod.execute(_cfg("anchor", tag="d"), runs, data, CPU, **QUIET)
    assert test_batch(plan, runs, data, CPU, **QUIET) == [_cfg("anchor", tag="d").run_id]
    assert _failures(runs / "b" / _cfg("anchor", tag="d").run_id) == []
    assert _exported(runs, runs.parent)[_cfg("anchor", tag="d").run_id]["status"] == "trained"


def test_export_groups_trials_apart_from_other_tags_and_lists_limits(workspace):
    runs, data = workspace
    run_mod.execute(_cfg("tune", tag="sigma"), runs, data, CPU, **QUIET)
    run_mod.execute(_cfg("tune", tag="time-small", max_epochs=1, patience=1), runs, data, CPU,
                    **QUIET)  # fmt: skip
    export(runs, runs.parent / "results")
    import csv

    with (runs.parent / "results" / "overview.csv").open() as f:
        groups = {r["tag"]: r for r in csv.DictReader(f)}
    assert groups["sigma"]["truncated_share"] == "0.0"
    assert groups["time-small"]["truncated_share"] == "1.0"
    row = _exported(runs, runs.parent)[_cfg("tune", tag="sigma").run_id]
    assert row["max_epochs"] == "50" and row["patience"] == "3"


def test_a_torn_last_failure_record_is_ignored(tmp_path):
    from seqrec.runner.attempts import failures

    (tmp_path / "failures.jsonl").write_text('{"step": "train"}\n{"step": "tr')
    assert failures(tmp_path) == [{"step": "train"}]


def test_the_guard_counts_training_and_validation_time_only(workspace, monkeypatch):
    """D036: checkpoint writes and set-up do not count toward the guard."""
    import time as real_time
    from types import SimpleNamespace

    runs, data = workspace
    clock = [0.0]
    monkeypatch.setattr(
        run_mod, "time", SimpleNamespace(perf_counter=lambda: clock[0], sleep=real_time.sleep)
    )
    save = run_mod._save_atomic

    def slow_save(obj, path):  # each checkpoint write takes an hour of the fake clock
        clock[0] += 3600.0
        save(obj, path)

    def epoch_of_one_second(self, train, epoch):
        clock[0] += 1.0
        self.epochs = epoch + 1
        return {"loss": 0.0}

    monkeypatch.setattr(run_mod, "_save_atomic", slow_save)
    monkeypatch.setattr(PopularityModel, "train_epoch", epoch_of_one_second)
    summary = run_mod.execute(_cfg(guard_hours=2.0), runs, data, CPU, **QUIET)
    assert summary["stop_reason"] == "patience" and summary["epochs"] == 4
    assert summary["wall_s"] == pytest.approx(4.0) and summary["checkpoint_s"] >= 4 * 3600


def _timed_row(seed, train_s, tag="sigma", params="{}", eval_s=0.1):
    return {
        "batch": "b", "dataset": "ml-1m", "split": "loo", "shuffle": "none", "arm": "sasrec-ce",
        "tag": tag, "params": params, "seed": seed, "run_id": f"r{seed}-{tag}-{params}",
        "epoch_train_s_median": train_s, "epoch_eval_s_median": eval_s,
    }  # fmt: skip


def test_slow_epochs_flags_runs_far_slower_than_the_same_configuration():
    """D038: a run whose epoch time exceeds 1.5 times the median of its other seeds."""
    from seqrec.runner.export import flag_slow_epochs

    rows = [_timed_row(1, 1.73), _timed_row(2, 30.56), _timed_row(3, 1.91)]
    rows += [_timed_row(1, 1.0, tag="t01"), _timed_row(1, 9.0, tag="t02")]  # no other seeds
    rows += [_timed_row(1, 1.0, params='{"lr": 0.1}'), _timed_row(2, 1.4, params='{"lr": 0.1}')]
    rows.append(_timed_row(4, None, eval_s=None))  # never finished an epoch
    flag_slow_epochs(rows)
    assert [r["slow_epochs"] for r in rows] == [False, True, False, None, None, False, False, None]


def test_patience_waits_until_a_run_beats_popularity(workspace):
    """D035: before validation NDCG@10 exceeds 1.5 x MostPop's, patience does not count."""
    import seqrec.models  # noqa: F401  registers mostpop

    runs, data = workspace
    PopularityModel.peak = 6  # random scores (below 1.5 x MostPop here) except at epoch 6
    summary = run_mod.execute(_cfg(), runs, data, CPU, **QUIET)
    assert summary["stop_reason"] == "patience" and summary["epochs"] == 10
    assert summary["start_epoch"] == 6 and summary["best_epoch"] == 6
    floor = run_mod.execute(_cfg(arm="mostpop", tag="floor"), runs, data, CPU, **QUIET)
    assert summary["start_reference"] == pytest.approx(floor["validation"]["ndcg@10"])
    manifest = json.loads((runs / "b" / _cfg().run_id / "manifest.json").read_text())
    assert manifest["start_reference"] == summary["start_reference"]
    assert floor["start_reference"] is None and floor["stop_reason"] == "converged"


def test_a_run_that_never_beats_popularity_stops_at_its_start_limit(workspace):
    runs, data = workspace
    PopularityModel.peak = 1000
    summary = run_mod.execute(_cfg(start_limit=8), runs, data, CPU, **QUIET)
    assert summary["stop_reason"] == "no_start" and summary["truncated"]
    assert summary["epochs"] == 8 and summary["start_epoch"] is None
    row = _exported(runs, runs.parent)[_cfg().run_id]
    assert row["start_limit"] == "8" and row["start_epoch"] == ""


def test_start_limits_default_by_arm_and_can_be_set():
    assert _cfg(arm="gsasrec").start_limit == 200 and _cfg(arm="bert4rec").start_limit == 200
    assert _cfg(arm="mostpop").start_limit == 0 and _cfg(arm="transition").start_limit == 0
    assert _cfg(arm="gsasrec", start_limit=400).start_limit == 400


def test_runs_that_did_not_start_count_as_truncated():
    """D035: a no_start stop is a limit, like the epoch cap, for the rule of METHODS 5.2."""
    from seqrec.runner.export import tuning_best

    rows = [_trial_row(f"t{i:02d}", 0.2) for i in range(4)]
    rows.append(_trial_row("t04", 0.02, stop="no_start", truncated=True))
    best = tuning_best(rows)[0]
    assert best["stopped"] == 5 and best["truncated"] == 1


def _legacy_config(cfg, runs):
    """A run directory claimed before D035, whose config.json has no start_limit."""
    run_dir = runs / cfg.batch / cfg.run_id
    run_dir.mkdir(parents=True)
    stored = json.loads(cfg.to_json())
    del stored["start_limit"]
    (run_dir / "config.json").write_text(json.dumps(stored, indent=1, sort_keys=True))
    return run_dir


def test_run_directories_from_before_the_start_rule_are_not_conflicts(workspace):
    runs, data = workspace
    _legacy_config(_cfg(), runs)
    assert run_mod.config_conflicts([_cfg()], runs) == []
    assert run_mod.execute(_cfg(), runs, data, CPU, **QUIET)["stop_reason"] == "patience"
    assert run_mod.test_run(_cfg(), runs, data, CPU, **QUIET)["test"]["users"] > 0


def test_a_run_from_before_the_start_rule_accepts_only_its_own_limits(workspace):
    """A pre-D035 directory trained without the rule: a plan with the default limit, written
    before D035, or with 0 matches it; any other limit is a conflict that names the field."""
    runs, data = workspace
    _legacy_config(_cfg(), runs)
    assert run_mod.config_conflicts([_cfg(), _cfg(start_limit=0)], runs) == []
    assert run_mod.config_conflicts([_cfg(start_limit=400)], runs) == [_cfg().run_id]
    with pytest.raises(run_mod.RunConflict, match="start_limit absent, now 400"):
        run_mod.execute(_cfg(start_limit=400), runs, data, CPU, **QUIET)


def test_a_changed_default_start_limit_stops_an_interrupted_run(workspace, monkeypatch):
    """A run keeps the start limit it began with (D035, D040)."""
    from seqrec.runner import config

    runs, data = workspace
    PopularityModel.crash = "train@2"
    with pytest.raises(RuntimeError):
        run_mod.execute(_cfg(), runs, data, CPU, **QUIET)
    monkeypatch.setattr(config, "START_LIMIT", 300)
    with pytest.raises(run_mod.RunConflict, match="start_limit 200, now 300"):
        run_mod.execute(_cfg(), runs, data, CPU, **QUIET)


def test_a_negative_start_limit_is_rejected():
    with pytest.raises(ValueError, match="start_limit"):
        _cfg(start_limit=-1)


def _crash_then_resume(runs, data, cfg, crash_epoch, edit_state=None):
    PopularityModel.crash = f"train@{crash_epoch}"
    with pytest.raises(RuntimeError):
        run_mod.execute(cfg, runs, data, CPU, **QUIET)
    if edit_state is not None:
        last = runs / cfg.batch / cfg.run_id / "last.pt"
        saved = torch.load(last, weights_only=False)
        edit_state(saved["state"])
        torch.save(saved, last)
    return run_mod.execute(cfg, runs, data, CPU, **QUIET)


def test_an_interrupted_run_keeps_its_start_rule_state(workspace):
    """D035: interrupted before it started, a run resumes as if uninterrupted."""
    runs, data = workspace
    PopularityModel.peak = 6
    summary = _crash_then_resume(runs, data, _cfg(), crash_epoch=3)
    assert summary["stop_reason"] == "patience" and summary["epochs"] == 10
    assert summary["start_epoch"] == 6 and summary["best_epoch"] == 6


def test_a_run_from_before_the_start_rule_finishes_without_it(workspace):
    """Resumed after D035, a run that never beat 1.5 x MostPop stops by patience as it would
    have, instead of training on to the cap (D040)."""
    runs, data = workspace
    PopularityModel.peak = 1000  # constant random scores: best at epoch 0, never started
    _legacy_config(_cfg(), runs)
    summary = _crash_then_resume(runs, data, _cfg(), crash_epoch=2)
    assert summary["stop_reason"] == "patience" and summary["epochs"] == 4
    assert summary["start_limit"] == 0 and summary["start_reference"] is None


def test_a_checkpoint_without_its_start_record_is_refused(workspace):
    runs, data = workspace

    def drop_start_record(state):
        del state["started"]

    with pytest.raises(run_mod.RunConflict, match="start rule"):
        _crash_then_resume(runs, data, _cfg(), crash_epoch=2, edit_state=drop_start_record)


def test_export_lists_peak_device_memory(workspace):
    runs, data = workspace
    run_mod.execute(_cfg("tune"), runs, data, CPU, **QUIET)
    path = runs / "b" / _cfg("tune").run_id / "summary.json"
    summary = json.loads(path.read_text())
    path.write_text(json.dumps(summary | {"nvml": {"mem_used_mib": {"max": 15500}}}))
    assert _exported(runs, runs.parent)[_cfg().run_id]["nvml_mem_used_mib_max"] == "15500"


@pytest.mark.parametrize("field", ["train_fingerprint", "full_fingerprint"])
def test_a_run_is_not_tested_on_data_other_than_it_trained_on(workspace, field):
    """METHODS 11.6: the test read uses the data recorded in the manifest, or none."""
    runs, data = workspace
    run_mod.execute(_cfg(), runs, data, CPU, **QUIET)
    path = runs / "b" / _cfg().run_id / "manifest.json"
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps(manifest | {field: "0" * 16}))
    with pytest.raises(run_mod.TestRefused, match="data differ"):
        run_mod.test_run(_cfg(), runs, data, CPU, **QUIET)
    assert not (path.parent / "test_access.jsonl").exists()


def test_a_rebuilt_dataset_is_refused_before_the_test_read(workspace):
    runs, data = workspace
    run_mod.execute(_cfg(), runs, data, CPU, **QUIET)
    build_log(random_frame(seed=1)).save(log_path(data, "ml-1m"))
    load_splits.cache_clear()
    with pytest.raises(run_mod.TestRefused, match="data differ"):
        run_mod.test_run(_cfg(), runs, data, CPU, **QUIET)


def _fake_clock(monkeypatch):
    import time as real_time
    from types import SimpleNamespace

    clock = [0.0]
    monkeypatch.setattr(
        run_mod, "time", SimpleNamespace(perf_counter=lambda: clock[0], sleep=real_time.sleep)
    )
    return clock


def test_the_guard_leaves_out_set_up_and_adds_up_sessions(workspace, monkeypatch):
    """D036: each session's set-up (data, invariants, model) takes an hour of the fake clock
    and is not counted; epoch time from before an interruption is."""
    runs, data = workspace
    clock = _fake_clock(monkeypatch)
    init = PopularityModel.__init__

    def slow_init(self, cfg, train, device):
        clock[0] += 3600.0
        init(self, cfg, train, device)

    def epoch_of_1000_seconds(self, train, epoch):
        if epoch == 2 and not getattr(PopularityModel, "crashed", False):
            PopularityModel.crashed = True
            raise RuntimeError("simulated crash")
        clock[0] += 1000.0
        self.epochs = epoch + 1
        return {"loss": 0.0}

    monkeypatch.setattr(PopularityModel, "__init__", slow_init)
    monkeypatch.setattr(PopularityModel, "train_epoch", epoch_of_1000_seconds)
    monkeypatch.setattr(PopularityModel, "crashed", False, raising=False)
    cfg = _cfg(guard_hours=1.0, patience=50)
    with pytest.raises(RuntimeError):
        run_mod.execute(cfg, runs, data, CPU, **QUIET)
    summary = run_mod.execute(cfg, runs, data, CPU, **QUIET)
    assert summary["stop_reason"] == "guard" and summary["truncated"]
    assert summary["epochs"] == 4 and summary["wall_s"] == pytest.approx(4000.0)


def test_export_flags_the_slow_seed(workspace, monkeypatch):
    """D038: of three seeds with epochs of 1, 1 and 10 s, only the last is flagged."""
    runs, data = workspace
    clock = _fake_clock(monkeypatch)
    seconds = {1: 1.0, 2: 1.0, 3: 10.0}
    init = PopularityModel.__init__

    def init_with_seed(self, cfg, train, device):
        init(self, cfg, train, device)
        self.seed = cfg.seed

    def timed_epoch(self, train, epoch):
        clock[0] += seconds[self.seed]
        self.epochs = epoch + 1
        return {"loss": 0.0}

    monkeypatch.setattr(PopularityModel, "__init__", init_with_seed)
    monkeypatch.setattr(PopularityModel, "train_epoch", timed_epoch)
    for seed in seconds:
        run_mod.execute(_cfg("tune", seed=seed, tag="sigma"), runs, data, CPU, **QUIET)
    rows = _exported(runs, runs.parent).values()
    assert {int(r["seed"]): r["slow_epochs"] for r in rows} == {1: "False", 2: "False", 3: "True"}
