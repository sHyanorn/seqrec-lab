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
