import math
import os
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from conftest import random_frame

from seqrec.data.preprocess import build_log
from seqrec.data.store import load_splits, log_path
from seqrec.models.trainer import BATCH_SIZE
from seqrec.runner.config import RunConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import profile_step  # noqa: E402


def test_profile_step_reports_time_per_step(tmp_path, capsys):
    data = tmp_path / "data"
    build_log(random_frame(users=300)).save(log_path(data, "ml-1m"))
    load_splits.cache_clear()
    args = ["--dataset", "ml-1m", "--arm", "sasrec-bce", "--steps", "2", "--warmup", "1"]
    assert profile_step.main([*args, "--data-dir", str(data), "--device", "cpu"]) == 0
    load_splits.cache_clear()
    out = capsys.readouterr().out
    fields = ("wall_ms_per_step", "cpu_ms_per_step", "gpu_ms_per_step", "steps 2",
              "torch_reserved_gib_max", "alloc_retries")  # fmt: skip
    for field in fields:
        assert field in out
    assert not list(tmp_path.glob("runs*"))  # writes nothing


def test_gpu_time_counts_each_gpu_event_once():
    """CPU ops also report the device time of the kernels they launch, and GPU user
    annotations span other events; only the kernels themselves are summed."""
    from torch.autograd import DeviceType

    def event(device, us, annotation=False):
        return SimpleNamespace(device_type=device, device_time_total=us,
                               is_user_annotation=annotation)  # fmt: skip

    events = [
        event(DeviceType.CPU, 30.0),  # aten::mm, which launched the kernel below
        event(DeviceType.CUDA, 30.0),  # the kernel
        event(DeviceType.CUDA, 5.0),  # a memset
        event(DeviceType.CUDA, 50.0, annotation=True),  # a ProfilerStep range on the GPU
    ]
    assert profile_step.device_events(events) == (35.0, 2)


def test_fewer_than_two_steps_are_refused(tmp_path):
    for steps in ("0", "1"):
        with pytest.raises(SystemExit):
            profile_step.main(["--dataset", "ml-1m", "--arm", "sasrec-bce", "--steps", steps])


class _FixedCostModel:
    """Sleeps 60 ms per epoch plus `per_step` s per step; 10 full batches and a partial one."""

    per_step = 0.002

    def __init__(self, cfg, train, device):
        self.data = SimpleNamespace(rows=torch.zeros(10 * BATCH_SIZE + 24, 2, dtype=torch.long))

    def train_epoch(self, train, epoch):
        time.sleep(0.06 + self.per_step * math.ceil(len(self.data.rows) / BATCH_SIZE))
        return {"loss": 0.0}


def test_time_per_step_leaves_out_the_cost_per_epoch(tmp_path, monkeypatch):
    data = tmp_path / "data"
    build_log(random_frame(users=300)).save(log_path(data, "ml-1m"))
    load_splits.cache_clear()
    monkeypatch.setitem(profile_step.ARMS, "fixed", _FixedCostModel)
    cfg = RunConfig("profile", "ml-1m", "loo", "fixed", 0, "tune")
    result = profile_step.measure(cfg, data, torch.device("cpu"), steps=8, warmup=1)
    assert result["wall_ms_per_step"] == pytest.approx(2.0, abs=1.5)
    assert result["epoch_overhead_ms"] == pytest.approx(60.0, abs=10.0)
    whole = profile_step.measure(cfg, data, torch.device("cpu"), steps=50, warmup=1)
    assert whole["steps"] == 10  # full batches only: a partial one has another shape
    monkeypatch.setattr(_FixedCostModel, "per_step", -0.002)  # timing noise beyond a step
    with pytest.raises(ValueError, match="more steps"):
        profile_step.measure(cfg, data, torch.device("cpu"), steps=8, warmup=1)
    load_splits.cache_clear()


def test_a_run_directory_supplies_the_configuration(tmp_path, capsys):
    """--from-run reads dataset, arm and parameters from a run's config.json, so commands
    need no JSON on the command line."""
    import json

    data = tmp_path / "data"
    build_log(random_frame(users=300)).save(log_path(data, "ml-1m"))
    params = {"dim": 32, "blocks": 1, "heads": 1, "dropout": 0.2}
    cfg = RunConfig("m2-anchor", "ml-1m", "loo", "sasrec-ce", 0, "anchor", params=params)
    run_dir = tmp_path / "runs" / cfg.batch / cfg.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(cfg.to_json())
    load_splits.cache_clear()
    args = ["--from-run", str(run_dir), "--steps", "2", "--warmup", "1"]
    assert profile_step.main([*args, "--data-dir", str(data), "--device", "cpu"]) == 0
    load_splits.cache_clear()
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert (result["dataset"], result["arm"], result["params"]) == ("ml-1m", "sasrec-ce", params)
    assert not (run_dir / "test_access.jsonl").exists()
    with pytest.raises(SystemExit):
        profile_step.main(["--from-run", str(run_dir), "--arm", "sasrec-bce"])


def test_results_outside_the_assumptions_carry_a_note():
    """D039: GPU time comes from a separate profiled pass, and the split into per-step and
    per-epoch time assumes every step takes as long."""
    clean = {"gpu_busy_share": 0.2, "epoch_overhead_ms": 1.0}
    assert profile_step.notes(clean) == []
    assert profile_step.notes(clean | {"gpu_busy_share": None}) == []
    gpu_bound = profile_step.notes(clean | {"gpu_busy_share": 1.15})
    assert len(gpu_bound) == 1 and "GPU-bound" in gpu_bound[0]
    uneven = profile_step.notes(clean | {"epoch_overhead_ms": -2.0})
    assert len(uneven) == 1 and "epoch_overhead_ms" in uneven[0]


def test_memory_readings_separate_memory_in_use_from_cached_blocks():
    """D042: the allocator's reserved peak, its retries after a failed device allocation and
    its device allocations show whether cached blocks, not live tensors, fill the device."""
    stats = {
        "allocated_bytes.all.peak": 5 * 2**30,
        "reserved_bytes.all.peak": 27 * 2**30,
        "num_alloc_retries": 0,
        "num_device_alloc": 9,
    }
    assert profile_step.memory_readings(stats) == {
        "torch_allocated_gib_max": 5.0,
        "torch_reserved_gib_max": 27.0,
        "alloc_retries": 0,
        "device_allocs": 9,
    }
    assert set(profile_step.memory_readings(None).values()) == {None}  # no CUDA device


class _ScriptedModel:
    """Each step warns of `syncs` synchronisations, and each epoch of one more."""

    syncs, loss = 0, 0.5
    threaded = []

    def __init__(self, cfg, train, device):
        self.data = SimpleNamespace(rows=torch.zeros(10 * BATCH_SIZE, 2, dtype=torch.long))

    def train_epoch(self, train, epoch):
        self.threaded.append(torch.autograd.is_multithreading_enabled())
        for _ in range(bool(self.syncs) + self.syncs * (len(self.data.rows) // BATCH_SIZE)):
            warnings.warn("called a synchronizing CUDA operation", stacklevel=1)
        return {"loss": self.loss}


def test_synchronisations_are_counted_per_step_on_the_calling_thread(monkeypatch):
    """Backward nodes of a GPU run on an autograd thread unless multithreading is off, and
    warnings raised there do not reach the count."""
    monkeypatch.setattr(torch.cuda, "set_sync_debug_mode", lambda mode: None)
    monkeypatch.setattr(_ScriptedModel, "syncs", 2)
    monkeypatch.setattr(_ScriptedModel, "threaded", [])
    model = _ScriptedModel(None, None, None)
    assert profile_step.syncs_per_step(model, None, model.data.rows) == 2.0
    assert model.threaded == [False, False] and torch.autograd.is_multithreading_enabled()


def _scripted(tmp_path, monkeypatch):
    data = tmp_path / "data"
    build_log(random_frame(users=300)).save(log_path(data, "ml-1m"))
    load_splits.cache_clear()
    monkeypatch.setitem(profile_step.ARMS, "scripted", _ScriptedModel)
    return data, RunConfig("profile", "ml-1m", "loo", "scripted", 0, "tune")


def test_a_non_finite_loss_stops_the_measurement(tmp_path, monkeypatch):
    """A step that produces NaN would otherwise be timed and printed like a working one."""
    data, cfg = _scripted(tmp_path, monkeypatch)
    result = profile_step.measure(cfg, data, torch.device("cpu"), steps=3, warmup=1)
    assert result["loss"] == 0.5
    monkeypatch.setattr(_ScriptedModel, "loss", float("nan"))
    with pytest.raises(ValueError, match="loss"):
        profile_step.measure(cfg, data, torch.device("cpu"), steps=3, warmup=1)
    load_splits.cache_clear()


def test_timings_are_printed_before_the_profiled_pass(tmp_path, monkeypatch, capsys):
    """The profiled pass can end the process under a CUDA graph; the timings must be out."""
    data, _ = _scripted(tmp_path, monkeypatch)

    def crash(*args, **kwargs):
        raise RuntimeError("profiler")

    monkeypatch.setattr(profile_step, "gpu_readings", crash)
    args = ["--dataset", "ml-1m", "--arm", "scripted", "--steps", "3", "--warmup", "1"]
    with pytest.raises(RuntimeError, match="profiler"):
        profile_step.main([*args, "--data-dir", str(data), "--device", "cpu"])
    load_splits.cache_clear()
    timing = [x for x in capsys.readouterr().out.splitlines() if x.startswith("timing ")]
    assert len(timing) == 1 and "wall_ms_per_step" in timing[0] and "loss" in timing[0]


def test_gpu_readings_give_time_and_events_per_step(monkeypatch):
    passes = iter([(100.0, 10)] * 5 + [(2100.0, 50)])  # five one-step epochs, then 5 steps
    monkeypatch.setattr(profile_step, "_gpu", lambda *args: next(passes))
    cuda, cpu = torch.device("cuda"), torch.device("cpu")
    got = profile_step.gpu_readings(None, None, None, 5, 5, wall_ms=1.0, device=cuda)
    assert got == {"gpu_ms_per_step": 0.5, "gpu_busy_share": 0.5, "gpu_events_per_step": 10.0}
    none = profile_step.gpu_readings(None, None, None, 5, 5, wall_ms=1.0, device=cpu)
    assert set(none) == set(got) and set(none.values()) == {None}


def test_a_cuda_graph_result_says_what_its_gpu_time_covers():
    clean = {"gpu_busy_share": 0.2, "epoch_overhead_ms": 1.0, "modes": ["fixed_shapes"]}
    assert profile_step.notes(clean) == []
    graph = profile_step.notes(clean | {"modes": ["fixed_shapes", "cuda_graph"]})
    assert len(graph) == 1 and "gpu_events_per_step" in graph[0]


def test_the_profiler_is_set_up_before_a_graph_is_captured(monkeypatch):
    """Without these the profiler's re-initialisation after a capture can end the process
    (torch.profiler sets them itself only for compiled graphs)."""
    monkeypatch.setenv("TEARDOWN_CUPTI", "1")
    monkeypatch.delenv("DISABLE_CUPTI_LAZY_REINIT", raising=False)
    profile_step.prepare_profiler_for_graphs()
    assert os.environ["TEARDOWN_CUPTI"] == "0" and os.environ["DISABLE_CUPTI_LAZY_REINIT"] == "1"
