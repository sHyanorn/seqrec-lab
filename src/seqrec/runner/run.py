"""Train one run with early stopping on validation; read test data in a separate step.

Run directory (runs/<batch>/<run_id>/):
    config.json, manifest.json   what was run, on which code and data, invariants passed
    metrics.jsonl                one line per finished epoch
    last.pt                      resume state, including the best weights so far; removed
                                 when training ends (local only)
    best.pt                      the selected weights, written when training ends; removed
                                 for tuning runs, which are never tested
    nvml.jsonl                   GPU telemetry at 1 Hz
    summary.json                 written when training ends; marks the run as trained
    test_access.jsonl, test.json written by `test_run` (phases final and anchor only)

Test data are read by `test_run` only after every run of its plan has finished training
(METHODS 5.5), so no test score exists while seeds are still being added.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from seqrec.data.invariants import check_batch_positions, check_split
from seqrec.data.registry import DATASETS
from seqrec.data.splits import SealedOff, SplitSet, TrainSplit, shuffle_eval_inputs, shuffle_order
from seqrec.data.store import load_splits
from seqrec.eval.ranking import NonFiniteScores, evaluate
from seqrec.eval.test_access import count_reads, mark_completed, read_test
from seqrec.runner.config import RunConfig, default_start_limit
from seqrec.telemetry import NvmlSampler, nvml_static, summarize_samples

REPO_ROOT = Path(__file__).resolve().parents[3]
MAX_ANCHOR_READS = 3
CPU_QUOTA = Path("/sys/fs/cgroup/cpu.max")  # cgroup v2: "<quota> <period>" or "max <period>"
# cgroup v1: cpu.cfs_quota_us (-1 without a quota) and cpu.cfs_period_us in one of these
CPU_QUOTA_V1 = (Path("/sys/fs/cgroup/cpu"), Path("/sys/fs/cgroup/cpu,cpuacct"))
START_FACTOR = 1.5  # D035: a run starts once validation NDCG@10 exceeds this times MostPop's


class Model(Protocol):
    """state_dict returns a snapshot of everything training depends on: weights, optimizer,
    scaler and any random generator the model owns. Optional attributes: `hparams` (resolved
    hyperparameters, recorded in the manifest), `sample_positions()` (full-log positions
    used by one sampled training batch, checked against the splits) and `weights()` (a
    snapshot of what inference needs, used for the best checkpoint)."""

    def train_epoch(self, train: TrainSplit, epoch: int) -> dict: ...

    def score(self, inputs: torch.Tensor) -> torch.Tensor: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...


# Factories see only the training split: validation and test data stay with the runner.
ARMS: dict[str, Callable[[RunConfig, TrainSplit, torch.device], Model]] = {}


def register_arm(name: str):
    def wrap(factory):
        ARMS[name] = factory
        return factory

    return wrap


class TestRefused(PermissionError):
    """A test read refused before any data were read (METHODS 11.5, 11.6)."""


class RunConflict(RuntimeError):
    """A run directory holds a different configuration."""


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_dir_of(cfg: RunConfig, runs_dir: Path) -> Path:
    return runs_dir / cfg.batch / cfg.run_id


def _replace(tmp: Path, path: Path, attempts: int = 5) -> None:
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: a virus scanner or indexer can hold a file briefly
            if attempt == attempts - 1:
                raise
            time.sleep(0.2 * 2**attempt)


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())  # survives a power loss as a whole file or not at all
    _replace(tmp, path)


def _save_atomic(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    _replace(tmp, path)


def _differences(config_path: Path, cfg: RunConfig) -> list[str]:
    """Fields in which a stored config.json differs from cfg, as "field stored, now new"."""
    stored = json.loads(config_path.read_text(encoding="utf-8"))
    current = json.loads(cfg.to_json())
    if "start_limit" not in stored:
        # Claimed before the start rule (D035), so trained without it. A plan written before
        # then resolves to the arm's default; one may also state 0. Either keeps it off (D040).
        same = cfg.start_limit in (0, default_start_limit(cfg.arm))
        stored["start_limit"] = cfg.start_limit if same else "absent"
    return [
        f"{k} {stored.get(k)}, now {current.get(k)}"
        for k in sorted(stored.keys() | current.keys())
        if stored.get(k) != current.get(k)
    ]


def config_conflicts(cfgs: list[RunConfig], runs_dir: Path) -> list[str]:
    """Runs whose existing directory holds a different configuration."""
    out = []
    for cfg in cfgs:
        path = run_dir_of(cfg, runs_dir) / "config.json"
        if path.exists() and _differences(path, cfg):
            out.append(cfg.run_id)
    return out


def _claim(cfg: RunConfig, run_dir: Path) -> RunConfig:
    """Create the run directory, or confirm it belongs to this exact configuration. Returns
    the configuration the run trains with: a run from before the start rule keeps it off."""
    config_path = run_dir / "config.json"
    if config_path.exists():
        if diff := _differences(config_path, cfg):
            raise RunConflict(f"{run_dir} holds a different configuration: {'; '.join(diff)}")
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        return cfg if "start_limit" in stored else replace(cfg, start_limit=0)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(config_path, cfg.to_json())
    return cfg


def _prepare_splits(cfg: RunConfig, data_dir: Path) -> tuple[SplitSet, list[str]]:
    splits, checks = load_splits(data_dir, cfg.dataset, cfg.split)
    if cfg.shuffle == "train_eval":
        splits = shuffle_order(splits, seed=cfg.seed)
        checks = checks + check_split(splits)
    if cfg.phase == "tune":
        splits = replace(splits, test=SealedOff())
    return splits, checks


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _pruning_step(epoch: int, state: dict) -> int | None:
    """Epochs since the run started, the step at which tuning compares trials (D044); None
    before the start. A run without the start rule (D035) counts from epoch 0."""
    if not state.get("started", True):
        return None
    step = epoch - (state.get("start_epoch") or 0)
    return step if step >= 0 else None


def _stop_reason(state: dict, cfg: RunConfig) -> str | None:
    if state.get("stop"):  # decided inside the loop (pruned, diverged, converged) and saved
        return state["stop"]
    started = state.get("started", True)  # runs without the start rule (D035) count as started
    if started and state["bad"] >= cfg.patience:
        return "patience"
    if not started and state["epoch"] >= cfg.start_limit:
        return "no_start"
    if state["wall_s"] >= cfg.guard_hours * 3600:
        return "guard"
    if state["epoch"] >= cfg.max_epochs:
        return "max_epochs"
    return None


def _keep_metrics_before(path: Path, epoch: int) -> None:
    """Drop metric lines of epochs that will be repeated after a resume."""
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if json.loads(line)["epoch"] < epoch]
        _write_atomic(path, "".join(line + "\n" for line in kept))


def _weights(model: Model) -> dict:
    """What a restored checkpoint needs: inference weights if the model separates them."""
    return model.weights() if hasattr(model, "weights") else model.state_dict()


def _epoch_times(metrics_path: Path) -> dict:
    """Median seconds per epoch for training and for validation scoring."""
    lines = metrics_path.read_text(encoding="utf-8").splitlines() if metrics_path.exists() else []
    records = [json.loads(line) for line in lines]
    return {
        f"{key}_median": round(float(np.median([r[key] for r in records])), 3)
        if records else None
        for key in ("epoch_train_s", "epoch_eval_s")
    }  # fmt: skip


def release_device_memory(device: torch.device) -> None:
    """Drop unreachable objects and return the allocator's cached blocks to the driver.
    Where the driver may place memory in system RAM once device memory is full (Windows),
    cached blocks would otherwise push new allocations there (D041)."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def epoch_memory(device: torch.device) -> dict:
    """Peak memory PyTorch allocated and reserved since the last reset; the reserved peak also
    counts blocks the allocator keeps cached (D041)."""
    if device.type != "cuda":
        return {}
    return {
        "train_peak_memory_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
        "train_peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 3),
    }


def _memory_peaks(metrics_path: Path) -> dict:
    """Largest per-epoch memory peaks of a run; None when its epochs recorded none."""
    lines = metrics_path.read_text(encoding="utf-8").splitlines() if metrics_path.exists() else []
    records = [json.loads(line) for line in lines]
    peaks = {}
    for key in ("train_peak_memory_gib", "train_peak_reserved_gib"):
        values = [r[key] for r in records if r.get(key) is not None]
        peaks[f"{key}_max"] = max(values) if values else None
    return peaks


def max_len_of(cfg: RunConfig) -> int:
    return cfg.max_len or DATASETS[cfg.dataset].max_len


def popularity_reference(cfg: RunConfig, splits: SplitSet, device: torch.device) -> float:
    """MostPop's (B1) validation NDCG@10 on the run's own splits, the level a run must beat
    by START_FACTOR before patience counts (D035)."""
    from seqrec.models.baselines import MostPop  # it imports this module to register itself

    floor = MostPop(replace(cfg, arm="mostpop", params={}), splits.train, device)
    return evaluate(floor.score, splits.validation, max_len_of(cfg), device).metrics["ndcg@10"]


def host_signature(device: torch.device) -> str:
    """GPU model, driver branch, CPU model, CPU count and data centre of this machine, as a
    description. Run times are compared only between runs made on one Pod (D046)."""
    gpu, driver = "cpu", "none"
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
        driver = str(nvml_static().get("driver_version")).split(".")[0]
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        models = [x for x in cpuinfo.read_text().splitlines() if x.startswith("model name")]
        cpu = models[0].split(":", 1)[1].strip() if models else cpu
    cpus, _ = cpu_allocation()
    return " | ".join([gpu, driver, cpu, str(cpus), os.environ.get("RUNPOD_DC_ID", "")])


def cpu_allocation() -> tuple[int, str]:
    """CPUs this process may use, and where the number comes from. A container can run on
    every thread of its host and still be limited to fewer by its cgroup's quota: cgroup v2
    is read first, then cgroup v1; without a quota the count is the CPUs it may run on."""
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    v1 = [[d / "cpu.cfs_quota_us", d / "cpu.cfs_period_us"] for d in CPU_QUOTA_V1]
    for source, files in [("cgroup v2", [CPU_QUOTA]), *(("cgroup v1", f) for f in v1)]:
        try:
            quota, period = " ".join(f.read_text() for f in files).split()
            if int(quota) > 0:
                cpus = min(count, math.ceil(int(quota) / int(period)))
                return cpus, f"{source} quota in {files[0].parent}: {quota} {period}"
        except (OSError, ValueError):  # no such file, or "max" in cgroup v2: no quota
            pass
    return count, f"affinity: {count}"


def execute(
    cfg: RunConfig,
    runs_dir: Path,
    data_dir: Path,
    device: torch.device,
    log=print,
    report: Callable[[int, float], bool] | None = None,
) -> dict:
    """Train until a stop rule fires and write summary.json. No test data are read here.

    `report(step, validation NDCG@10)` is called after every epoch from the run's start
    epoch on, with the epochs since then as the step; returning True stops the run as
    pruned (tuning, METHODS 5.4, D044)."""
    run_dir = run_dir_of(cfg, runs_dir)
    summary_path = run_dir / "summary.json"
    cfg = _claim(cfg, run_dir)
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    release_device_memory(device)  # blocks an earlier run of this process left cached (D041)
    commit = git_commit()
    seed_everything(cfg.seed)
    splits, checks = _prepare_splits(cfg, data_dir)
    if len(splits.validation) == 0:
        raise ValueError(f"{cfg.dataset}/{cfg.split}: empty validation split")
    max_len = max_len_of(cfg)
    model = ARMS[cfg.arm](cfg, splits.train, device)
    if hasattr(model, "sample_positions"):
        checks = checks + check_batch_positions(splits, model.sample_positions())
    reference = popularity_reference(cfg, splits, device) if cfg.start_limit else None
    manifest = {
        "git_commit": commit,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "train_fingerprint": splits.train.fingerprint,
        "full_fingerprint": splits.full.fingerprint(),
        "hparams": getattr(model, "hparams", None),
        "invariants_passed": checks,
        "start_reference": reference,
        "hosts": [host_signature(device)],
        "pods": [os.environ.get("RUNPOD_POD_ID", "")],  # the rented machine of this session
    }
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["train_fingerprint"] != manifest["train_fingerprint"]:
            raise RunConflict(f"{run_dir}: training data changed since the run started")
        if (
            old.get("full_fingerprint", manifest["full_fingerprint"])
            != manifest["full_fingerprint"]
        ):
            raise RunConflict(f"{run_dir}: full data changed since the run started")
        manifest["sessions"] = old.get("sessions", [old["git_commit"]]) + [commit]
        manifest["hosts"] = old.get("hosts", []) + manifest["hosts"]
        manifest["pods"] = old.get("pods", []) + manifest["pods"]
    _write_atomic(manifest_path, json.dumps(manifest, indent=1))

    state = {"epoch": 0, "best": -1.0, "best_epoch": -1, "bad": 0, "train_s": 0.0, "wall_s": 0.0}
    best_weights = None
    last_path, metrics_path = run_dir / "last.pt", run_dir / "metrics.jsonl"
    if last_path.exists():
        saved = torch.load(last_path, weights_only=False)
        model.load_state_dict(saved["model"])
        best_weights, state = saved["best_model"], saved["state"]
        _set_rng_state(saved["rng"])
        log(f"[run] {cfg.run_id}: resuming at epoch {state['epoch']}")
    state.setdefault("checkpoint_s", 0.0)
    _keep_metrics_before(metrics_path, state["epoch"])
    if reference is not None and "started" not in state:  # a resumed run keeps its own
        if last_path.exists():
            raise RunConflict(f"{run_dir}: last.pt has no record of the start rule the run uses")
        state |= {"started": False, "start_reference": reference, "start_epoch": None}

    def checkpoint() -> None:
        t0 = time.perf_counter()
        _save_atomic(
            {"model": model.state_dict(), "best_model": best_weights, "state": state,
             "rng": _rng_state()},
            last_path,
        )  # fmt: skip
        state["checkpoint_s"] += time.perf_counter() - t0

    if report is not None and metrics_path.exists():  # a resumed trial re-reports its steps
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if (step := _pruning_step(record["epoch"], state)) is not None:
                report(step, record["val_ndcg@10"])

    sampler = NvmlSampler(interval_s=1.0) if device.type == "cuda" else None
    if sampler and sampler.available:
        sampler.start()
    try:
        while (stop := _stop_reason(state, cfg)) is None:
            t_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            stats = model.train_epoch(splits.train, state["epoch"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            stats |= epoch_memory(device)
            t_trained = time.perf_counter()
            try:
                val = evaluate(model.score, splits.validation, max_len, device)
            except NonFiniteScores:
                state["stop"] = "diverged"
                checkpoint()  # so that a retry does not train on
                break
            t_evaluated = time.perf_counter()
            state["train_s"] += t_trained - t_start
            state["wall_s"] += t_evaluated - t_start
            ndcg = val.metrics["ndcg@10"]
            if not state.get("started", True) and ndcg > START_FACTOR * state["start_reference"]:
                state |= {"started": True, "start_epoch": state["epoch"]}
            if ndcg > state["best"]:
                state |= {"best": ndcg, "best_epoch": state["epoch"], "bad": 0}
                best_weights = _weights(model)
            else:
                state["bad"] += 1
            record = {
                "epoch": state["epoch"],
                "train_s": round(state["train_s"], 3),
                "val_ndcg@10": ndcg,
                "val_hr@10": val.metrics["hr@10"],
                "val_tie_rate": val.tie_rate,
                "epoch_train_s": round(t_trained - t_start, 3),
                "epoch_eval_s": round(t_evaluated - t_trained, 3),
            } | stats
            state["epoch"] += 1
            # Log first, then checkpoint: a crash in between resumes before this epoch, and
            # the resume drops its line before repeating it.
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())
            step = _pruning_step(record["epoch"], state)
            if stats.get("converged"):  # models without training, e.g. the floor baselines
                state["stop"] = "converged"
            elif report is not None and step is not None and report(step, ndcg):
                state["stop"] = "pruned"
            checkpoint()  # with the stop decision, so that a retry does not train on
            if state.get("stop"):
                break
    finally:
        samples = sampler.stop() if sampler and sampler.available else []
        if samples:
            with (run_dir / "nvml.jsonl").open("a", encoding="utf-8") as f:
                f.writelines(json.dumps(s) + "\n" for s in samples)

    stop = _stop_reason(state, cfg)
    summary = {
        "run_id": cfg.run_id,
        "config": json.loads(cfg.to_json()),
        "git_commit": commit,
        "host": " + ".join(dict.fromkeys(manifest["hosts"])),  # every machine it trained on
        "pod": " + ".join(dict.fromkeys(manifest["pods"])),
        "status": "trained" if best_weights is not None else "diverged",
        "epochs": state["epoch"],
        "best_epoch": state["best_epoch"],
        "stop_reason": stop,
        "truncated": stop in ("max_epochs", "guard", "no_start"),
        "start_limit": cfg.start_limit,
        "start_reference": state.get("start_reference"),
        "start_epoch": state.get("start_epoch"),
        "train_s": round(state["train_s"], 2),
        "wall_s": round(state["wall_s"], 2),
        "checkpoint_s": round(state["checkpoint_s"], 2),  # outside wall_s; kept for budgeting
    } | _epoch_times(metrics_path)
    summary |= _memory_peaks(metrics_path)
    if best_weights is not None:
        best_path = run_dir / "best.pt"
        _save_atomic(best_weights, best_path)
        model.load_state_dict(best_weights)
        t0 = time.perf_counter()
        val = evaluate(model.score, splits.validation, max_len, device)
        summary["wall_s"] = round(state["wall_s"] + time.perf_counter() - t0, 2)
        summary["checkpoint_hash"] = file_hash(best_path)
        summary["validation"] = val.metrics | {"tie_rate": val.tie_rate, "users": val.users}
    nvml_path = run_dir / "nvml.jsonl"
    if sampler and sampler.available and nvml_path.exists():
        all_samples = [json.loads(x) for x in nvml_path.read_text(encoding="utf-8").splitlines()]
        summary["nvml"] = summarize_samples(sampler.nv, all_samples)
    _write_atomic(run_dir / "summary.json", json.dumps(summary, indent=1))
    # The resume state is no longer needed; tuning checkpoints are never tested (disk space).
    # Best effort: the run is complete once summary.json exists.
    for path in [last_path] + [run_dir / "best.pt"] * (cfg.phase == "tune"):
        try:
            path.unlink(missing_ok=True)
        except OSError as err:
            log(f"[run] {cfg.run_id}: could not remove {path.name} ({err})")
    log(f"[run] {cfg.run_id}: {stop} after {state['epoch']} epochs")
    return summary


def anchor_reads(runs_dir: Path, dataset: str, arm: str) -> int:
    """All anchor reads logged for (dataset, arm) in any batch, finished or not."""
    total = 0
    for log_path in runs_dir.glob("*/*/test_access.jsonl"):
        cfg = json.loads((log_path.parent / "config.json").read_text(encoding="utf-8"))
        if cfg["dataset"] == dataset and cfg["arm"] == arm:
            total += count_reads(log_path)["anchor"]
    return total


def test_run(
    cfg: RunConfig,
    runs_dir: Path,
    data_dir: Path,
    device: torch.device,
    log=print,
    earlier_anchor_reads: int = 0,
    earlier_final_reads: int = 0,
):
    """Read test data once for a trained final or anchor run and write test.json.
    `earlier_anchor_reads` are the anchor reads of this dataset and arm made on another
    machine; they count toward the limit. `earlier_final_reads` are the final reads of this
    run made there; a final run reads test data once, wherever."""
    if cfg.phase not in ("final", "anchor"):
        raise TestRefused("only final and anchor runs are tested")
    run_dir = run_dir_of(cfg, runs_dir)
    _claim(cfg, run_dir)
    if (run_dir / "test.json").exists():
        return json.loads((run_dir / "test.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "trained":
        raise RuntimeError(f"{cfg.run_id} has no checkpoint ({summary['status']})")
    used = anchor_reads(runs_dir, cfg.dataset, cfg.arm) + earlier_anchor_reads
    if cfg.phase == "anchor" and used >= MAX_ANCHOR_READS:
        raise TestRefused(f"{cfg.arm} on {cfg.dataset} used its {MAX_ANCHOR_READS} anchor reads")

    if cfg.phase == "final" and earlier_final_reads:
        raise TestRefused(f"{cfg.run_id}: final run read test data on another machine")

    access_log = run_dir / "test_access.jsonl"
    if cfg.phase == "final" and count_reads(access_log)["final"]:
        raise TestRefused(
            f"{cfg.run_id}: an earlier test read did not complete; final runs read test data once"
        )

    best_path = run_dir / "best.pt"
    checkpoint_hash = file_hash(best_path)
    if checkpoint_hash != summary["checkpoint_hash"]:
        raise TestRefused(f"{cfg.run_id}: checkpoint changed after training")
    seed_everything(cfg.seed)
    splits, _ = _prepare_splits(cfg, data_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    current = {"train_fingerprint": splits.train.fingerprint,
               "full_fingerprint": splits.full.fingerprint()}  # fmt: skip
    if changed := [k for k, v in current.items() if manifest.get(k) != v]:
        raise TestRefused(f"{cfg.run_id}: data differ from training ({', '.join(changed)})")
    model = ARMS[cfg.arm](cfg, splits.train, device)
    model.load_state_dict(torch.load(best_path, weights_only=False))
    test = read_test(
        splits.test,
        phase=cfg.phase,
        log_path=access_log,
        run_id=cfg.run_id,
        git_commit=git_commit(),
        checkpoint_hash=checkpoint_hash,
    )
    max_len = max_len_of(cfg)
    result = evaluate(model.score, test, max_len, device)
    shuffled = evaluate(
        model.score, shuffle_eval_inputs(test, cfg.seed, len(splits.full.items)), max_len, device
    )
    out = {
        "run_id": cfg.run_id,
        "test": result.metrics | {"tie_rate": result.tie_rate, "users": result.users},
        "test_inputs_shuffled": shuffled.metrics,
    }
    _write_atomic(run_dir / "test.json", json.dumps(out, indent=1))
    mark_completed(access_log, cfg.run_id)
    log(f"[test] {cfg.run_id}: ndcg@10 {result.metrics['ndcg@10']:.4f}")
    return out
