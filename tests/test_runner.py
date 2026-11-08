import json

import numpy as np
import pytest
import torch
from conftest import random_frame

from seqrec.cli import main as cli_main
from seqrec.data.invariants import item_popularity
from seqrec.data.preprocess import build_log
from seqrec.data.store import load_log, log_path
from seqrec.eval import test_access
from seqrec.runner import run as run_mod
from seqrec.runner.config import RunConfig
from seqrec.runner.export import export

CPU = torch.device("cpu")


class PopularityModel:
    """Test double: popularity scores plus fixed noise; one 'epoch' changes nothing."""

    crash_at: int | None = None

    def __init__(self, cfg, splits, device):
        counts = item_popularity(splits.train).counts.astype(np.float32)
        noise = np.random.default_rng(cfg.seed).random(len(counts)).astype(np.float32)
        self.scores = torch.from_numpy(counts + 1e-3 * noise)
        self.epochs = 0

    def train_epoch(self, train, epoch):
        if PopularityModel.crash_at == epoch:
            PopularityModel.crash_at = None
            raise RuntimeError("simulated crash")
        self.epochs += 1
        return {"loss": 0.0}

    def score(self, inputs):
        return self.scores.repeat(len(inputs), 1)

    def state_dict(self):
        return {"scores": self.scores, "epochs": self.epochs}

    def load_state_dict(self, state):
        self.scores, self.epochs = state["scores"], state["epochs"]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setitem(run_mod.ARMS, "pop", PopularityModel)
    data_dir = tmp_path / "data"
    build_log(random_frame()).save(log_path(data_dir, "ml-1m"))
    load_log.cache_clear()
    yield tmp_path, data_dir
    load_log.cache_clear()


def _cfg(phase="final", **kw):
    base = dict(batch="b", dataset="ml-1m", split="loo", arm="pop", seed=1, phase=phase)
    return RunConfig(**(base | {"max_epochs": 50, "patience": 3} | kw))


def test_final_run_reads_test_once_and_stops_on_patience(workspace):
    root, data_dir = workspace
    summary = run_mod.execute(_cfg(), root / "runs", data_dir, CPU, log=lambda _: None)
    assert summary["stop_reason"] == "patience" and summary["epochs"] == 4
    assert summary["test_reads"] == {"final": 1, "anchor": 0}
    assert 0 <= summary["test"]["ndcg@10"] <= 1
    assert "ndcg@10" in summary["test_inputs_shuffled"]
    again = run_mod.execute(_cfg(), root / "runs", data_dir, CPU, log=lambda _: None)
    assert again == summary  # finished runs are not repeated


def test_tuning_run_never_touches_test_data(workspace, monkeypatch):
    root, data_dir = workspace

    def forbidden(*args, **kwargs):
        raise AssertionError("tuning must not read test data")

    monkeypatch.setattr(run_mod, "read_test", forbidden)
    summary = run_mod.execute(_cfg("tune"), root / "runs", data_dir, CPU, log=lambda _: None)
    assert "test" not in summary
    assert summary["test_reads"] == {"final": 0, "anchor": 0}


def test_interrupted_run_resumes(workspace):
    root, data_dir = workspace
    PopularityModel.crash_at = 2
    with pytest.raises(RuntimeError):
        run_mod.execute(_cfg(), root / "runs", data_dir, CPU, log=lambda _: None)
    messages = []
    summary = run_mod.execute(_cfg(), root / "runs", data_dir, CPU, log=messages.append)
    assert any("resuming at epoch 2" in m for m in messages)
    lines = (root / "runs" / "b" / _cfg().run_id / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(x)["epoch"] for x in lines] == [0, 1, 2, 3]
    assert summary["epochs"] == 4


def test_test_reader_rejects_other_phases(tmp_path):
    with pytest.raises(PermissionError):
        test_access.read_test(
            None, phase="tune", log_path=tmp_path / "t.jsonl", run_id="r", git_commit="c",
            checkpoint_hash="h",
        )  # fmt: skip


def test_batch_and_export_through_cli(workspace, monkeypatch):
    root, data_dir = workspace
    plan = root / "plan.toml"
    plan.write_text(
        'batch = "b"\norder = "shuffled"\n[defaults]\ndataset = "ml-1m"\nsplit = "loo"\n'
        'phase = "final"\nmax_epochs = 10\npatience = 2\n'
        '[[runs]]\narm = "pop"\nseeds = [1, 2]\n'
        '[[runs]]\narm = "pop"\nsplit = "gts"\nseeds = [1]\n'
        '[[runs]]\narm = "pop"\nphase = "tune"\nseeds = [3]\n'
        '[[runs]]\narm = "pop"\nshuffle = "train_eval"\nseeds = [1]\n'
    )
    args = ["--runs-dir", str(root / "runs"), "--data-dir", str(data_dir), "--device", "cpu"]
    assert cli_main(["run", str(plan), *args]) == 0
    out = root / "results"
    assert cli_main(["export", "--runs-dir", str(root / "runs"), "--out-dir", str(out)]) == 0
    rows = (out / "b.csv").read_text().splitlines()
    assert len(rows) == 1 + 5

    tune_dir = next((root / "runs" / "b").glob("*s3*"))
    (tune_dir / "test_access.jsonl").write_text(json.dumps({"phase": "final"}) + "\n")
    summary = json.loads((tune_dir / "summary.json").read_text())
    summary["test_reads"] = {"final": 1, "anchor": 0}
    (tune_dir / "summary.json").write_text(json.dumps(summary))
    problems = export(root / "runs", out)
    assert any("tuning run read test data" in p for p in problems)
