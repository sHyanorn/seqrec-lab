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
from seqrec.eval.ranking import metrics_from_ranks
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
    monkeypatch.chdir(tmp_path)  # the CLI's defaults under results/ are the test's own
    _anchor_export(tmp_path / "results" / "laptop", 0)
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


def test_export_lists_the_epochs_that_set_a_new_best_for_final_runs(workspace):
    """Time to target (METHODS 8) is computed from the export: a target is first reached at an
    epoch that sets a new best, so those epochs with their cumulative training time are all
    it needs. The best epoch is epoch 1 here; epoch 0 scores at random and sets the first."""
    runs, data = workspace
    PopularityModel.peak = 1
    final, trial = _cfg(), _cfg("tune", tag="t00")
    for cfg in (final, trial):
        run_mod.execute(cfg, runs, data, CPU, **QUIET)
    export(runs, runs.parent / "out")
    rows = {r["run_id"]: r for r in _read_csv(runs.parent / "out" / "b.csv")}
    assert rows[trial.run_id]["validation_records"] == ""  # final runs only
    lines = (runs / "b" / final.run_id / "metrics.jsonl").read_text().splitlines()
    epochs = [json.loads(x) for x in lines]
    assert len(epochs) > 2 and epochs[1]["val_ndcg@10"] > epochs[0]["val_ndcg@10"]
    records = json.loads(rows[final.run_id]["validation_records"])
    assert records == [[e["train_s"], e["val_ndcg@10"]] for e in epochs[:2]]
    assert records[-1][1] == float(rows[final.run_id]["validation_ndcg@10"])
    for target in (0.0, epochs[0]["val_ndcg@10"], epochs[1]["val_ndcg@10"], 1.0):
        first = next((e["train_s"] for e in epochs if e["val_ndcg@10"] >= target), None)
        assert next((t for t, v in records if v >= target), None) == first


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


def test_the_host_signature_names_gpu_driver_cpu_cpu_count_and_data_centre(monkeypatch):
    monkeypatch.setenv("RUNPOD_DC_ID", "EU-RO-1")
    gpu, driver, cpu, count, data_centre = run_mod.host_signature(CPU).split(" | ")
    assert (gpu, driver, data_centre) == ("cpu", "none", "EU-RO-1")
    assert cpu and int(count) >= 1


def test_the_cpu_count_of_a_signature_is_the_containers_quota(tmp_path, monkeypatch):
    """A container sees every thread of its host; two hosts that allocate the same number of
    CPUs must not differ in it, and two allocations on one host must (D046)."""
    monkeypatch.setattr(run_mod.os, "sched_getaffinity", lambda pid: set(range(32)), raising=False)
    monkeypatch.setattr(run_mod, "CPU_QUOTA", tmp_path / "cpu.max")
    monkeypatch.setattr(run_mod, "CPU_QUOTA_V1", ())

    def count():
        return run_mod.host_signature(CPU).split(" | ")[3]

    (tmp_path / "cpu.max").write_text("1600000 100000\n")
    assert count() == "16"
    (tmp_path / "cpu.max").write_text("max 100000\n")  # no quota
    assert count() == "32"
    (tmp_path / "cpu.max").unlink()  # no cgroup v2
    assert count() == "32"


def test_a_cgroup_v1_quota_counts_like_a_v2_quota(tmp_path, monkeypatch):
    """A host without cgroup v2 limits its containers through cgroup v1. Reading only v2, two
    Pods on one kind of host got the counts 14 and 32 (D046)."""
    monkeypatch.setattr(run_mod.os, "sched_getaffinity", lambda pid: set(range(32)), raising=False)
    monkeypatch.setattr(run_mod, "CPU_QUOTA", tmp_path / "cpu.max")
    v1 = tmp_path / "cpu,cpuacct"
    monkeypatch.setattr(run_mod, "CPU_QUOTA_V1", (tmp_path / "cpu", v1))
    v1.mkdir()
    (v1 / "cpu.cfs_period_us").write_text("100000\n")

    (v1 / "cpu.cfs_quota_us").write_text("1360000\n")
    count, source = run_mod.cpu_allocation()
    assert count == 14 and source.startswith("cgroup v1") and "1360000 100000" in source
    (v1 / "cpu.cfs_quota_us").write_text("-1\n")  # no quota
    count, source = run_mod.cpu_allocation()
    assert count == 32 and source.startswith("affinity")
    (tmp_path / "cpu.max").write_text("1600000 100000\n")  # cgroup v2 is read first
    count, source = run_mod.cpu_allocation()
    assert count == 16 and source.startswith("cgroup v2") and "1600000 100000" in source


def test_an_exported_run_names_every_host_and_pod_it_trained_on(workspace, monkeypatch):
    """Run times are compared only between runs made on one Pod (D046), so a run resumed on
    another machine must not pass for a run of either."""
    runs, data = workspace
    monkeypatch.setattr(run_mod, "host_signature", lambda device: "first")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod1")
    PopularityModel.crash = "train@2"
    with pytest.raises(RuntimeError):
        run_mod.execute(_cfg(), runs, data, CPU, **QUIET)
    monkeypatch.setattr(run_mod, "host_signature", lambda device: "second")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod2")
    run_mod.execute(_cfg(), runs, data, CPU, **QUIET)
    run_mod.execute(_cfg(seed=2), runs, data, CPU, **QUIET)
    rows = _exported(runs, runs.parent).values()
    assert {r["seed"]: r["host"] for r in rows} == {"1": "first + second", "2": "second"}
    assert {r["seed"]: r["pod"] for r in rows} == {"1": "pod1 + pod2", "2": "pod2"}


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


def test_the_reader_itself_refuses_a_fourth_anchor_read(workspace):
    """METHODS 11.6: the limit binds every caller of the reader, wherever the reads were made."""
    runs, data = workspace
    for tag in "ab":
        cfg = _cfg("anchor", tag=tag)
        run_mod.execute(cfg, runs, data, CPU, **QUIET)
        run_mod.test_run(cfg, runs, data, CPU, **QUIET)
    third = _cfg("anchor", tag="c")
    run_mod.execute(third, runs, data, CPU, **QUIET)
    log = runs / "b" / third.run_id / "test_access.jsonl"

    def read(earlier_anchor_reads):
        return test_access.read_test(
            load_splits(data, "ml-1m", "loo")[0].test, phase="anchor", log_path=log,
            run_id=third.run_id, git_commit="c", checkpoint_hash="h",
            earlier_anchor_reads=earlier_anchor_reads,
        )  # fmt: skip

    with pytest.raises(test_access.TestRefused, match="anchor reads"):
        read(1)  # two here and one on another machine
    assert not log.exists()  # refused before any record or data
    assert len(read(0).targets) > 0
    with pytest.raises(test_access.TestRefused, match="anchor reads"):
        read(0)
    assert test_access.count_reads(log)["anchor"] == 1


def test_a_final_read_keeps_every_users_rank_in_the_run_directory(workspace):
    """METHODS 7.6 resamples users, and a final run reads test data once (11.6): the ranks of
    that read stay on the volume. They reproduce the reported metrics and enter no export."""
    runs, data = workspace
    final, anchor = _cfg(), _cfg("anchor", tag="a")
    for cfg in (final, anchor):
        run_mod.execute(cfg, runs, data, CPU, **QUIET)
    out = run_mod.test_run(final, runs, data, CPU, **QUIET)
    run_mod.test_run(anchor, runs, data, CPU, **QUIET)
    with np.load(runs / "b" / final.run_id / "test_ranks.npz") as ranks:
        assert sorted(ranks) == ["test", "test_inputs_shuffled"]
        assert len(ranks["test"]) == len(ranks["test_inputs_shuffled"]) == out["test"]["users"]
        for part in ("test", "test_inputs_shuffled"):
            recomputed = metrics_from_ranks(ranks[part])
            assert all(recomputed[k] == pytest.approx(out[part][k]) for k in recomputed)
    assert not (runs / "b" / anchor.run_id / "test_ranks.npz").exists()  # final runs only
    exported = runs.parent / "out"
    assert export(runs, exported) == []
    assert sorted(p.name for p in exported.iterdir()) == ["b.csv", "overview.csv"]
    assert not any("rank" in column for column in _read_csv(exported / "b.csv")[0])


def test_a_read_whose_ranks_cannot_be_written_is_an_incomplete_read(workspace, monkeypatch):
    """The read is used up and says so: no scores, no completion record, a failed job."""
    runs, data = workspace
    cfg = _cfg()
    run_mod.execute(cfg, runs, data, CPU, **QUIET)

    def full_disk(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(run_mod.np, "savez", full_disk)
    with pytest.raises(OSError, match="no space"):
        run_mod.test_run(cfg, runs, data, CPU, **QUIET)
    monkeypatch.undo()
    run_dir = runs / "b" / cfg.run_id
    assert not (run_dir / "test.json").exists() and not (run_dir / "test_ranks.npz").exists()
    assert test_access.count_reads(run_dir / "test_access.jsonl") == {
        "final": 1, "anchor": 0, "completed": 0,
    }  # fmt: skip
    assert export(runs, runs.parent / "out") == [f"{cfg.run_id}: a test read did not complete"]
    with pytest.raises(PermissionError, match="did not complete"):
        run_mod.test_run(cfg, runs, data, CPU, **QUIET)


def test_the_read_record_is_on_disk_before_any_data_are_returned(tmp_path, monkeypatch):
    """A crash after the scores are written must not leave a read without its record."""
    synced = []
    monkeypatch.setattr(test_access.os, "fsync", lambda fd: synced.append(fd))

    class Handle:
        def unseal(self, purpose):
            assert synced, "data returned before the record was synced"
            return "data"

    log = tmp_path / "b" / "r" / "test_access.jsonl"
    data = test_access.read_test(
        Handle(), phase="final", log_path=log, run_id="r", git_commit="c", checkpoint_hash="h",
        earlier_anchor_reads=0,
    )  # fmt: skip
    assert data == "data" and test_access.count_reads(log)["final"] == 1


def test_export_reports_test_scores_without_a_record_of_their_read(workspace):
    """Such a run would count as no read toward the limits of METHODS 11.5 and 11.6."""
    runs, data = workspace
    cfg = _cfg("anchor", tag="a")
    run_mod.execute(cfg, runs, data, CPU, **QUIET)
    run_mod.test_run(cfg, runs, data, CPU, **QUIET)
    assert export(runs, runs.parent / "out") == []
    (runs / "b" / cfg.run_id / "test_access.jsonl").unlink()
    assert export(runs, runs.parent / "out2") == [
        f"{cfg.run_id}: test scores without a record of the read"
    ]


def test_test_reader_rejects_other_phases(tmp_path):
    with pytest.raises(PermissionError):
        test_access.read_test(
            None, phase="tune", log_path=tmp_path / "t.jsonl", run_id="r", git_commit="c",
            checkpoint_hash="h", earlier_anchor_reads=0,
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


def _timed_row(seed, train_s, tag="sigma", params="{}", eval_s=0.1, pod="a"):
    return {
        "host": "one signature", "pod": pod,
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


def test_slow_epochs_compares_only_runs_made_on_one_pod():
    """A batch that went on on a slower Pod with the same signature: its later seeds are not
    slow runs, and a run that trained on both is compared with neither (D046)."""
    from seqrec.runner.export import flag_slow_epochs

    rows = [_timed_row(1, 1.0), _timed_row(2, 1.1)]
    rows += [_timed_row(3, 2.0, pod="b"), _timed_row(4, 2.1, pod="b")]
    rows.append(_timed_row(5, 9.0, pod="a + b"))
    flag_slow_epochs(rows)
    assert [r["slow_epochs"] for r in rows] == [False, False, False, False, None]


def _with_samples(run_dir, reasons):
    """Give a run the NVML record a GPU run has: nvml.jsonl and its summary."""
    import pynvml

    from seqrec.telemetry import summarize_samples

    samples = [{"t_s": float(i), "sm_mhz": 2000, "reasons": r} for i, r in enumerate(reasons)]
    lines = "".join(json.dumps(s) + "\n" for s in samples)
    (run_dir / "nvml.jsonl").write_text(lines, encoding="utf-8")
    path = run_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(summary | {"nvml": summarize_samples(pynvml, samples)}))


@pytest.mark.parametrize(
    ("samples", "flagged"),
    [(None, False), (3000 * [0], False), (2999 * [0] + [8], True), (999 * [0] + [8], True)],
    ids=["no NVML", "3000 quiet", "1 event in 3000", "1 event in 1000"],
)
def test_one_hardware_event_flags_a_run(workspace, samples, flagged):
    """METHODS 8: a run is flagged when NVML reports a hardware slowdown event, however many
    samples hold none. One event sample in 3,000 has a share that rounds to zero; in 1,000 it
    does not. A run without NVML (a CPU run) has no event to report."""
    import pynvml

    assert pynvml.nvmlClocksEventReasonHwSlowdown == 8
    runs, data = workspace
    run_mod.execute(_cfg(), runs, data, CPU, **QUIET)
    if samples is not None:
        _with_samples(runs / "b" / _cfg().run_id, samples)
    assert export(runs, runs.parent / "out") == []
    [row] = _read_csv(runs.parent / "out" / "b.csv")
    assert row["hardware_slowdown"] == str(flagged)


@pytest.mark.parametrize("damage", ["missing", "unreadable"])
def test_a_run_whose_samples_cannot_be_read_is_reported(workspace, damage):
    """A run whose summary holds NVML samples is not reported free of hardware events when
    nvml.jsonl is gone or damaged: the export lists it as a problem and leaves the flag empty."""
    runs, data = workspace
    run_mod.execute(_cfg(), runs, data, CPU, **QUIET)
    run_dir = runs / "b" / _cfg().run_id
    _with_samples(run_dir, 10 * [0])
    if damage == "missing":
        (run_dir / "nvml.jsonl").unlink()
    else:
        with (run_dir / "nvml.jsonl").open("a", encoding="utf-8") as f:
            f.write('{"t_s": 10.0, "reasons"')
    problems = export(runs, runs.parent / "out")
    assert problems == [f"{_cfg().run_id}: NVML samples missing or unreadable"]
    [row] = _read_csv(runs.parent / "out" / "b.csv")
    assert row["hardware_slowdown"] == ""


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


def _reported(runs, data, cfg):
    steps = []

    def report(step, value):
        steps.append((step, value))
        return False

    run_mod.execute(cfg, runs, data, CPU, report=report, **QUIET)
    return steps


def _validation(runs, cfg):
    lines = (runs / cfg.batch / cfg.run_id / "metrics.jsonl").read_text().splitlines()
    return [json.loads(x)["val_ndcg@10"] for x in lines]


def test_pruning_steps_count_from_the_start_epoch(workspace):
    """D044: nothing is reported before a run starts, and its start epoch is step 0."""
    runs, data = workspace
    PopularityModel.peak = 6
    reported = _reported(runs, data, _cfg())
    assert reported == list(enumerate(_validation(runs, _cfg())[6:])) and len(reported) == 4
    PopularityModel.peak = 1000  # never starts
    assert _reported(runs, data, _cfg(start_limit=8, tag="stuck")) == []
    PopularityModel.peak = 6  # without the start rule a run counts from epoch 0
    off = _cfg(start_limit=0, tag="off")
    assert _reported(runs, data, off) == list(enumerate(_validation(runs, off)))


def test_a_resumed_run_reports_its_earlier_steps_again(workspace):
    """D044: after an interruption the pruner gets the steps since the start, not epochs."""
    runs, data = workspace
    PopularityModel.peak = 6
    PopularityModel.crash = "train@8"
    with pytest.raises(RuntimeError):
        run_mod.execute(_cfg(), runs, data, CPU, report=lambda step, value: False, **QUIET)
    reported = _reported(runs, data, _cfg())
    assert reported == list(enumerate(_validation(runs, _cfg())[6:])) and len(reported) == 4


def test_a_run_that_never_beats_popularity_stops_at_its_start_limit(workspace):
    runs, data = workspace
    PopularityModel.peak = 1000
    summary = run_mod.execute(_cfg(start_limit=8), runs, data, CPU, **QUIET)
    assert summary["stop_reason"] == "no_start" and summary["truncated"]
    assert summary["epochs"] == 8 and summary["start_epoch"] is None
    row = _exported(runs, runs.parent)[_cfg().run_id]
    assert row["start_limit"] == "8" and row["start_epoch"] == ""


def test_start_limits_default_by_arm_and_can_be_set():
    assert _cfg(arm="gsasrec").start_limit == 300 and _cfg(arm="bert4rec").start_limit == 300
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
    monkeypatch.setattr(config, "START_LIMIT", 400)
    with pytest.raises(run_mod.RunConflict, match="start_limit 300, now 400"):
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
    """D036: each step of a session's set-up (data and invariants, model, popularity
    reference) takes an hour of the fake clock and is not counted; epoch time from before an
    interruption is."""
    runs, data = workspace
    clock = _fake_clock(monkeypatch)
    init = PopularityModel.__init__

    def an_hour_first(fn):
        def slow(*args, **kw):
            clock[0] += 3600.0
            return fn(*args, **kw)

        return slow

    monkeypatch.setattr(run_mod, "load_splits", an_hour_first(run_mod.load_splits))
    monkeypatch.setattr(
        run_mod, "popularity_reference", an_hour_first(run_mod.popularity_reference)
    )
    slow_init = an_hour_first(init)

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


def test_a_resumed_run_refuses_changed_evaluation_data(workspace):
    """Validation and test targets live in the full data; a session that sees other targets
    than the earlier ones would select the checkpoint on two validation sets (D040)."""
    runs, data = workspace

    def other_full_data():
        path = runs / "b" / _cfg().run_id / "manifest.json"
        manifest = json.loads(path.read_text())
        path.write_text(json.dumps(manifest | {"full_fingerprint": "0" * 16}))

    PopularityModel.crash = "train@2"
    with pytest.raises(RuntimeError):
        run_mod.execute(_cfg(), runs, data, CPU, **QUIET)
    other_full_data()
    with pytest.raises(run_mod.RunConflict, match="full data changed"):
        run_mod.execute(_cfg(), runs, data, CPU, **QUIET)


def test_epochs_record_the_memory_pytorch_reserved(monkeypatch):
    """D041: the allocator's reserved peak, next to the allocated one; a gap between the two
    is memory held by cached blocks."""
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 3 * 2**30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 5 * 2**30)
    assert run_mod.epoch_memory(torch.device("cuda")) == {
        "train_peak_memory_gib": 3.0,
        "train_peak_reserved_gib": 5.0,
    }
    assert run_mod.epoch_memory(CPU) == {}


def test_runs_summarize_and_export_their_memory_peaks(workspace, monkeypatch):
    runs, data = workspace
    peaks = iter(range(1, 1000))

    def fake(device):
        p = next(peaks)
        return {"train_peak_memory_gib": p / 2, "train_peak_reserved_gib": float(p)}

    monkeypatch.setattr(run_mod, "epoch_memory", fake)
    summary = run_mod.execute(_cfg("tune"), runs, data, CPU, **QUIET)
    epochs = summary["epochs"]
    assert summary["train_peak_reserved_gib_max"] == epochs
    assert summary["train_peak_memory_gib_max"] == epochs / 2
    row = _exported(runs, runs.parent)[_cfg().run_id]
    assert float(row["torch_reserved_gib_max"]) == epochs
    assert float(row["torch_allocated_gib_max"]) == epochs / 2


def test_runs_without_memory_records_summarize_to_none(workspace):
    runs, data = workspace
    summary = run_mod.execute(_cfg("tune"), runs, data, CPU, **QUIET)
    assert summary["train_peak_reserved_gib_max"] is None
    assert _exported(runs, runs.parent)[_cfg().run_id]["torch_reserved_gib_max"] == ""


def test_each_run_releases_cached_device_memory_before_it_builds_its_model(workspace, monkeypatch):
    """D041: blocks cached by an earlier run in the same process are freed first, so they
    cannot push the new run's allocations out of device memory."""
    runs, data = workspace
    calls = []
    monkeypatch.setattr(run_mod, "release_device_memory", lambda device: calls.append("free"))
    model = run_mod.ARMS["pop"]
    monkeypatch.setitem(run_mod.ARMS, "pop", lambda *a: calls.append("model") or model(*a))
    run_mod.execute(_cfg(), runs, data, CPU, **QUIET)
    assert calls[:2] == ["free", "model"]


def test_releasing_device_memory_empties_the_cuda_cache_only(monkeypatch):
    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
    run_mod.release_device_memory(CPU)
    assert emptied == []
    run_mod.release_device_memory(torch.device("cuda"))
    assert emptied == [True]


# --- summaries from batch files; anchor reads made on another machine ----------------------


def _batch_file(directory, batch, rows):
    from seqrec.data.stats import write_csv

    reads = {"reads_final": 0, "reads_anchor": 0, "reads_completed": 0}
    write_csv(directory / f"{batch}.csv", [reads | r | {"batch": batch} for r in rows])


def _read_csv(path):
    import csv

    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_summaries_cover_batch_files_exported_from_another_volume(workspace):
    """A machine holds only its own run directories; the batch files hold every machine's."""
    runs, data = workspace
    out = runs.parent / "results"
    run_mod.execute(_cfg("tune", tag="sigma"), runs, data, CPU, **QUIET)
    assert export(runs, out) == []
    other = runs.parent / "other-volume"
    run_mod.execute(_cfg("tune", batch="c", tag="sigma"), other, data, CPU, **QUIET)
    assert export(other, out) == []
    assert [r["batch"] for r in _read_csv(out / "overview.csv")] == ["b", "c"]
    assert len(_read_csv(out / "b.csv")) == 1  # left as the first export wrote it


def test_summaries_from_files_equal_summaries_from_the_rows_written(tmp_path):
    """Values come back from a CSV as text: "False" must not count as truncated, a missing
    stop reason must count as unfinished, and scores must compare as numbers."""
    from seqrec.runner.export import _overview, summarize, tuning_best

    rows = [
        _trial_row("t00", 0.9, stop="pruned"),
        _trial_row("t01", 0.25),
        _trial_row("t02", 0.1234567891234, stop="max_epochs", truncated=True),
        _trial_row("t03", None, stop=None, status="failed", truncated=None),
        _trial_row("t10", 0.3),
    ]
    _batch_file(tmp_path, "t", rows)
    assert summarize(tmp_path) == []
    overview, best = _read_csv(tmp_path / "overview.csv"), _read_csv(tmp_path / "tuning_best.csv")
    assert overview == [{k: "" if v is None else str(v) for k, v in _overview(rows)[0].items()}]
    assert best == [{k: "" if v is None else str(v) for k, v in tuning_best(rows)[0].items()}]
    assert overview[0]["truncated"] == "1" and overview[0]["stopped"] == "3"
    assert best[0]["best_run_id"] == "r-t10" and best[0]["failed_or_unfinished"] == "1"


def test_a_file_that_is_not_its_batch_is_refused(tmp_path):
    from seqrec.runner.export import summarize

    _batch_file(tmp_path, "t", [_trial_row("t00", 0.2)])
    (tmp_path / "renamed.csv").write_text((tmp_path / "t.csv").read_text())
    with pytest.raises(ValueError, match=r"renamed\.csv"):
        summarize(tmp_path)
    (tmp_path / "renamed.csv").write_text("dataset,users\nml-1m,6040\n")
    with pytest.raises(ValueError, match=r"renamed\.csv"):
        summarize(tmp_path)


def test_summaries_alone_through_cli(tmp_path):
    _batch_file(tmp_path, "t", [_trial_row("t00", 0.2)])
    (tmp_path / "overview.csv").write_text("batch\nstale\n")
    args = ["export", "--summaries-only", "--out-dir", str(tmp_path), "--earlier-exports"]
    assert cli_main(args) == 0
    assert [r["batch"] for r in _read_csv(tmp_path / "overview.csv")] == ["t"]
    assert (tmp_path / "tuning_best.csv").exists()


def _anchor_export(directory, n, arm="pop"):
    row = _trial_row("r1", 0.2) | {"arm": arm, "phase": "anchor", "status": "tested"}
    _batch_file(directory, "m", [row | {"reads_anchor": n, "reads_completed": n}])
    return directory


def test_anchor_reads_exported_from_another_machine_count_toward_the_limit(workspace):
    """METHODS 11.5 allows 3 reads per arm, wherever they were made."""
    from seqrec.runner.batch import test_batch

    runs, data = workspace
    earlier = _anchor_export(runs.parent / "laptop", 1)
    plan = runs.parent / "anchor.toml"
    plan.write_text(
        'batch = "b"\n[defaults]\ndataset = "ml-1m"\nsplit = "loo"\nphase = "anchor"\n'
        "max_epochs = 50\npatience = 3\n"
        '[[runs]]\narm = "pop"\nseeds = [1]\ntag = "a"\n'
        '[[runs]]\narm = "pop"\nseeds = [1]\ntag = "b"\n'
        '[[runs]]\narm = "pop"\nseeds = [1]\ntag = "c"\n',
        encoding="utf-8",
    )
    for tag in "abc":
        run_mod.execute(_cfg("anchor", tag=tag), runs, data, CPU, **QUIET)
    refused = test_batch(plan, runs, data, CPU, earlier=[earlier], **QUIET)
    assert refused == [_cfg("anchor", tag="c").run_id]  # the laptop's read and two here
    assert not (runs / "b" / _cfg("anchor", tag="c").run_id / "test_access.jsonl").exists()
    assert export(runs, runs.parent / "results", earlier=[earlier]) == []
    other_arm = _anchor_export(runs.parent / "other", 3, arm="sasrec-ce")
    assert test_batch(plan, runs, data, CPU, earlier=[other_arm], **QUIET) == []


def test_export_reports_anchor_reads_over_the_limit_across_machines(workspace):
    runs, data = workspace
    for tag in "abc":
        cfg = _cfg("anchor", tag=tag)
        run_mod.execute(cfg, runs, data, CPU, **QUIET)
        run_mod.test_run(cfg, runs, data, CPU, **QUIET)
    out = runs.parent / "results"
    assert export(runs, out) == []
    earlier = _anchor_export(runs.parent / "laptop", 1)
    assert export(runs, out, earlier=[earlier]) == ["pop on ml-1m: 4 anchor reads (limit 3)"]


def test_a_missing_directory_of_earlier_exports_stops_before_any_read(workspace):
    """Counting a directory that is not there as zero reads would lift the limit unnoticed."""
    from seqrec.runner.batch import test_batch

    runs, data = workspace
    run_mod.execute(_cfg("anchor", tag="a"), runs, data, CPU, **QUIET)
    plan = runs.parent / "anchor.toml"
    plan.write_text(
        'batch = "b"\n[defaults]\ndataset = "ml-1m"\nsplit = "loo"\nphase = "anchor"\n'
        'max_epochs = 50\npatience = 3\n[[runs]]\narm = "pop"\nseeds = [1]\ntag = "a"\n',
        encoding="utf-8",
    )
    missing = runs.parent / "nowhere"
    with pytest.raises(FileNotFoundError, match="nowhere"):
        test_batch(plan, runs, data, CPU, earlier=[missing], **QUIET)
    assert not (runs / "b" / _cfg("anchor", tag="a").run_id / "test_access.jsonl").exists()
    with pytest.raises(FileNotFoundError, match="nowhere"):
        export(runs, runs.parent / "results", earlier=[missing])
    args = ["--runs-dir", str(runs), "--data-dir", str(data), "--device", "cpu"]
    with pytest.raises(FileNotFoundError, match="nowhere"):
        cli_main(["test", str(plan), *args, "--earlier-exports", str(missing)])


def test_the_cli_counts_the_laptop_exports_unless_told_otherwise(tmp_path):
    """The default must be what is committed: `seqrec test` in the workflow names no directory.
    In the repository's `results/` that is the laptop's exports, and no directory that would
    stop every job for holding none."""
    from pathlib import Path

    from seqrec.runner.export import (
        NOT_EXPORTS,
        earlier_exports,
        read_batch_files,
        read_earlier_exports,
    )

    root = Path(__file__).resolve().parents[1]
    committed = earlier_exports(root / "results")
    assert root / "results/laptop" in committed
    assert read_earlier_exports(committed)
    for name in (*NOT_EXPORTS, "old-pod"):
        (tmp_path / name).mkdir()
    (tmp_path / "notes.txt").write_text("a file is not a directory of exports")
    assert earlier_exports(tmp_path) == [tmp_path / "laptop", tmp_path / "old-pod"]
    assert earlier_exports(tmp_path / "nowhere") == [tmp_path / "nowhere" / "laptop"]
    rows = read_batch_files(root / "results/laptop")
    reads = {(r["dataset"], r["arm"]): r["reads_anchor"] for r in rows if r["phase"] == "anchor"}
    assert reads and set(reads.values()) == {1}


def test_export_before_any_run_exists_writes_nothing_and_reports_nothing(tmp_path):
    """The workflow exports before a plan starts, on a new volume too."""
    out = tmp_path / "results"
    assert export(tmp_path / "runs", out) == []
    assert list(out.glob("*.csv")) == []
    assert cli_main(["export", "--runs-dir", str(tmp_path / "runs"), "--out-dir", str(out)]) == 0
    with pytest.raises(FileNotFoundError):  # a mistyped directory is not an empty one
        cli_main(["export", "--summaries-only", "--out-dir", str(tmp_path / "typo")])


def test_export_with_a_plan_refuses_until_every_run_is_trained_or_given_up(workspace):
    """A run that never started has no row; an export of its batch would pass for complete."""
    from seqrec.runner.attempts import TRAIN_ATTEMPTS, record_failure
    from seqrec.runner.config import load_plan
    from seqrec.runner.run import run_dir_of

    runs, data = workspace
    plan = _plan(runs.parent, seeds=(1, 2))
    out = runs.parent / "out"
    export_plan = ["export", "--runs-dir", str(runs), "--out-dir", str(out), "--plan", str(plan)]
    assert cli_main(export_plan) == 1
    assert not out.exists() or not list(out.iterdir())  # nothing is written on a refusal
    assert cli_main(["run", str(plan), "--runs-dir", str(runs), "--data-dir", str(data),
                     "--device", "cpu"]) == 0  # fmt: skip
    assert cli_main(export_plan) == 0
    lost = run_dir_of(load_plan(plan).runs[1], runs)
    (lost / "summary.json").unlink()
    assert cli_main(export_plan) == 1
    for _ in range(TRAIN_ATTEMPTS):
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            record_failure(lost, "train")
    assert cli_main(export_plan) == 0  # given up counts as finished (D031)
    plan.write_text(plan.read_text().replace("max_epochs = 6", "max_epochs = 9"))
    with pytest.raises(run_mod.RunConflict):  # not the plan these runs were trained from
        cli_main(export_plan)
    plan.write_text(plan.read_text() + '[[studies]]\narm = "pop"\ntrials = 1\n')
    with pytest.raises(ValueError, match="both"):
        cli_main(export_plan)


def test_a_run_plan_with_studies_is_refused_not_half_run(tmp_path):
    """`seqrec run` on a mixed plan would train the runs and drop the studies unnoticed."""
    from seqrec.runner.config import load_plan

    plan = _plan(tmp_path)
    plan.write_text(plan.read_text() + '[[studies]]\narm = "pop"\ntrials = 1\n')
    with pytest.raises(ValueError, match="both runs and studies"):
        load_plan(plan)


def test_a_plan_with_a_key_no_command_reads_is_refused(tmp_path):
    """A misspelt table or setting would be dropped without notice."""
    from seqrec.runner.config import load_plan

    plan = _plan(tmp_path)
    text = plan.read_text()
    known = "batch, defaults, warmup_s, order, order_seed, runs"
    for broken, key in (
        (text + '[[Runs]]\narm = "pop"\n', "Runs"),
        ("order_sed = 3\n" + text, "order_sed"),
    ):
        plan.write_text(broken)
        with pytest.raises(ValueError, match=f"unknown key {key}.*{known}"):
            load_plan(plan)
    plan.write_text('order = "shuffled"\norder_seed = 3\nwarmup_s = 1\n' + text)
    assert len(load_plan(plan).runs) == 1


@pytest.mark.parametrize("runs", ["", "runs = []\n"])
def test_a_plan_without_a_run_is_refused_not_finished(tmp_path, runs):
    """Nothing to train is nothing left to train: its export would pass for complete."""
    from seqrec.runner.config import load_plan

    plan = tmp_path / "plan.toml"
    plan.write_text(f'batch = "b"\n{runs}', encoding="utf-8")
    with pytest.raises(ValueError, match="no run and no study"):
        load_plan(plan)
    export_plan = ["export", "--runs-dir", str(tmp_path / "runs"), "--plan", str(plan)]
    with pytest.raises(ValueError, match="no run and no study"):
        cli_main([*export_plan, "--out-dir", str(tmp_path / "results"), "--earlier-exports"])


def test_an_earlier_export_without_runs_is_an_error_not_zero_reads(workspace, tmp_path):
    from seqrec.runner.export import read_batch_files, summarize

    runs, data = workspace
    out, empty = tmp_path / "out", tmp_path / "empty"
    out.mkdir()
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no exported run"):
        summarize(out, [empty])
    plan = _plan(runs.parent, phase="anchor")
    args = [str(plan), "--runs-dir", str(runs), "--data-dir", str(data), "--device", "cpu"]
    assert cli_main(["run", *args]) == 0
    with pytest.raises(FileNotFoundError, match="no exported run"):
        cli_main(["test", *args, "--earlier-exports", str(empty)])
    for truncated in ("", "\n", "run_id,batch\n"):  # a file cut short, down to its header
        (empty / "b.csv").write_text(truncated, encoding="utf-8")
        with pytest.raises(ValueError, match="no run"):
            read_batch_files(empty)


@pytest.mark.parametrize("more", [[], ["--summaries-only"]])
def test_export_reads_the_earlier_exports_it_is_given_and_the_laptop_by_default(
    tmp_path, monkeypatch, more
):
    out = tmp_path / "out"
    out.mkdir()
    export_here = ["export", "--runs-dir", str(tmp_path / "runs"), "--out-dir", str(out), *more]
    with pytest.raises(FileNotFoundError, match="elsewhere"):
        cli_main([*export_here, "--earlier-exports", str(tmp_path / "elsewhere")])
    monkeypatch.chdir(tmp_path)  # no results/laptop here
    with pytest.raises(FileNotFoundError, match="laptop"):
        cli_main(export_here)


# --- test reads recorded on main and made on another volume (D045) ------------------------


def _read_plan(root, tags="a", phase="anchor"):
    plan = root / "b.toml"
    plan.write_text(
        f'batch = "b"\n[defaults]\ndataset = "ml-1m"\nsplit = "loo"\nphase = "{phase}"\n'
        "max_epochs = 50\npatience = 3\n"
        + "".join(f'[[runs]]\narm = "pop"\nseeds = [1]\ntag = "{t}"\n' for t in tags),
        encoding="utf-8",
    )
    return plan


def _read_on_another_volume(root, data, cfg, committed="committed"):
    """Train and test `cfg` on a volume of its own; export it where main keeps batch files."""
    run_mod.execute(cfg, root / "old-volume", data, CPU, **QUIET)
    run_mod.test_run(cfg, root / "old-volume", data, CPU, **QUIET)
    assert export(root / "old-volume", root / committed) == []
    return root / committed


@pytest.mark.parametrize("phase", ["anchor", "final"])
def test_a_read_main_records_and_this_volume_lacks_stops_the_test_task(workspace, phase):
    """Counted nowhere, it would let a fourth anchor read or a second final read through."""
    from seqrec.runner.batch import test_batch

    runs, data = workspace
    cfg = _cfg(phase, tag="a")
    committed = _read_on_another_volume(runs.parent, data, cfg)
    plan = _read_plan(runs.parent, phase=phase)
    run_mod.execute(cfg, runs, data, CPU, **QUIET)  # trained again on a new volume
    with pytest.raises(run_mod.TestRefused, match=r"b/.*results/<name>/"):
        test_batch(plan, runs, data, CPU, committed=committed, **QUIET)
    assert not (runs / "b" / cfg.run_id / "test_access.jsonl").exists()
    same_volume = runs.parent / "old-volume"
    assert test_batch(plan, same_volume, data, CPU, committed=committed, **QUIET) == []
    nothing_committed = runs.parent / "absent"
    assert test_batch(plan, runs, data, CPU, committed=nothing_committed, **QUIET) == []


def test_a_read_recorded_for_another_batch_is_missed_just_the_same(workspace):
    """The limit is per dataset and arm across batches, so every batch file on main counts."""
    from seqrec.runner.batch import test_batch

    runs, data = workspace
    other_batch = _cfg("anchor", batch="earlier", tag="a")
    committed = _read_on_another_volume(runs.parent, data, other_batch)
    cfg = _cfg("anchor", tag="a")
    run_mod.execute(cfg, runs, data, CPU, **QUIET)
    with pytest.raises(run_mod.TestRefused, match=f"earlier/{other_batch.run_id}"):
        test_batch(_read_plan(runs.parent), runs, data, CPU, committed=committed, **QUIET)
    assert not (runs / "b" / cfg.run_id / "test_access.jsonl").exists()


def test_export_keeps_a_batch_file_whose_reads_this_volume_lacks(workspace, capsys):
    """Replacing it would take the record of those reads off main."""
    from seqrec.runner.export import UnbackedReads

    runs, data = workspace
    cfg = _cfg("anchor", tag="a")
    committed = _read_on_another_volume(runs.parent, data, cfg)
    run_mod.execute(cfg, runs, data, CPU, **QUIET)

    def tree():
        return {p: p.read_bytes() for p in runs.parent.rglob("*") if p.is_file()}

    before = tree()
    with pytest.raises(UnbackedReads, match=cfg.run_id):
        export(runs, committed)
    assert tree() == before
    check = ["export", "--check-reads", "--out-dir", str(committed), "--runs-dir"]
    assert cli_main([*check, str(runs)]) == 1
    assert "results/<name>/" in capsys.readouterr().err  # what to do about it
    assert cli_main([*check, str(runs.parent / "new-volume")]) == 1
    assert cli_main([*check, str(runs.parent / "old-volume")]) == 0
    not_yet = runs.parent / "no-export-yet"
    assert cli_main(["export", "--check-reads", "--out-dir", str(not_yet)]) == 0
    assert tree() == before  # a check writes nothing, whatever it finds
    assert not not_yet.exists() and not (runs.parent / "new-volume").exists()


def test_the_test_task_checks_results_runs_unless_told_otherwise(workspace, monkeypatch):
    runs, data = workspace
    root = runs.parent
    cfg = _cfg("anchor", tag="a")
    _read_on_another_volume(root, data, cfg, committed="results/runs")
    _anchor_export(root / "results" / "laptop", 0)
    monkeypatch.chdir(root)
    args = [str(_read_plan(root)), "--runs-dir", str(runs), "--data-dir", str(data)]
    args += ["--device", "cpu"]
    assert cli_main(["run", *args]) == 0
    with pytest.raises(run_mod.TestRefused):
        cli_main(["test", *args])
    assert not (runs / "b" / cfg.run_id / "test_access.jsonl").exists()
    assert cli_main(["test", *args, "--committed", str(root / "absent")]) == 0


def test_exports_set_aside_under_results_count_without_being_named(workspace, monkeypatch):
    """Moving a lost volume's batch files to results/<name>/ is all it takes (D045)."""
    runs, data = workspace
    root = runs.parent
    _anchor_export(root / "results" / "laptop", 1)
    _anchor_export(root / "results" / "old-pod", 1)
    for name in ("data", "env"):  # hold no batch files and are not asked for any
        (root / "results" / name).mkdir()
    monkeypatch.chdir(root)
    args = [str(_read_plan(root, "ab")), "--runs-dir", str(runs), "--data-dir", str(data)]
    args += ["--device", "cpu"]
    assert cli_main(["run", *args]) == 0
    assert cli_main(["test", *args]) == 1  # the laptop's read, the old pod's and one here
    first, second = (runs / "b" / _cfg("anchor", tag=t).run_id for t in "ab")
    assert (first / "test_access.jsonl").exists()
    assert not (second / "test_access.jsonl").exists()
    assert cli_main(["export", "--runs-dir", str(runs)]) == 0
    _anchor_export(root / "results" / "third", 1)
    assert cli_main(["export", "--runs-dir", str(runs)]) == 1  # four reads of one arm
    assert cli_main(["export", "--summaries-only"]) == 1
    (root / "results" / "figures").mkdir()
    for command in (["test", *args], ["export", "--runs-dir", str(runs)]):
        with pytest.raises(FileNotFoundError, match="figures.*NOT_EXPORTS"):
            cli_main(command)


def test_export_lists_the_batch_files_it_wrote(workspace):
    """The workflow commits a batch file only if this export wrote it."""
    runs, data = workspace
    out, written = runs.parent / "results", runs.parent / "written"
    _batch_file(out, "stale", [_trial_row("t00", 0.2)])
    export_here = ["export", "--out-dir", str(out), "--earlier-exports", "--written", str(written)]
    assert cli_main([*export_here, "--runs-dir", str(runs)]) == 0
    assert written.read_text(encoding="utf-8") == ""
    run_mod.execute(_cfg("tune"), runs, data, CPU, **QUIET)
    assert cli_main([*export_here, "--runs-dir", str(runs)]) == 0
    assert written.read_text(encoding="utf-8") == "b.csv\n"


def test_a_final_read_set_aside_with_an_earlier_export_is_not_made_again(workspace):
    """A final run reads test data once, on whichever machine (METHODS 11.6)."""
    from seqrec.runner.batch import test_batch

    runs, data = workspace
    cfg = _cfg("final", tag="a")
    earlier = _read_on_another_volume(runs.parent, data, cfg, committed="pod1")
    plan = _read_plan(runs.parent, phase="final")
    run_mod.execute(cfg, runs, data, CPU, **QUIET)
    assert test_batch(plan, runs, data, CPU, earlier=[earlier], **QUIET) == [cfg.run_id]
    assert not (runs / "b" / cfg.run_id / "test_access.jsonl").exists()
    run_mod.test_run(cfg, runs, data, CPU, **QUIET)  # made all the same: the export says so
    problems = export(runs, runs.parent / "results", earlier=[earlier])
    assert problems == [f"{cfg.run_id}: final run read test data on another machine too"]


def test_reads_are_unbacked_when_the_volume_has_fewer_not_only_none(workspace):
    from seqrec.runner.export import unbacked_reads

    runs, data = workspace
    cfg = _cfg("anchor", tag="a")
    run_mod.execute(cfg, runs, data, CPU, **QUIET)
    run_mod.test_run(cfg, runs, data, CPU, **QUIET)
    row = _trial_row("a", 0.2) | {"run_id": cfg.run_id, "arm": "pop", "phase": "anchor"}
    committed = runs.parent / "committed"
    _batch_file(committed, "b", [row | {"reads_anchor": 2, "reads_completed": 2}])
    assert len(unbacked_reads(committed, runs)) == 1
    _batch_file(committed, "b", [row | {"reads_anchor": 1, "reads_completed": 1}])
    assert unbacked_reads(committed, runs) == []
