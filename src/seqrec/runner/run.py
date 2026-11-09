"""Train one run with early stopping on validation; read test data in a separate step.

Run directory (runs/<batch>/<run_id>/):
    config.json, manifest.json   what was run, on which code and data, invariants passed
    metrics.jsonl                one line per finished epoch
    last.pt                      resume state, including the best weights so far (local only)
    best.pt                      the selected checkpoint, written when training ends
    nvml.jsonl                   GPU telemetry at 1 Hz
    summary.json                 written when training ends; marks the run as trained
    test_access.jsonl, test.json written by `test_run` (phases final and anchor only)

Test data are read by `test_run` only after every run of its plan has finished training
(METHODS 5.5), so no test score exists while seeds are still being added.
"""

from __future__ import annotations

import hashlib
import json
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
from seqrec.runner.config import RunConfig
from seqrec.telemetry import NvmlSampler, summarize_samples

REPO_ROOT = Path(__file__).resolve().parents[3]
MAX_ANCHOR_READS = 3


class Model(Protocol):
    """state_dict returns a snapshot of everything training depends on: weights, optimizer,
    scaler and any random generator the model owns. Optional attributes: `hparams` (resolved
    hyperparameters, recorded in the manifest) and `sample_positions()` (full-log positions
    used by one sampled training batch, checked against the splits)."""

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


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _save_atomic(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _claim(cfg: RunConfig, run_dir: Path) -> None:
    """Create the run directory, or confirm it belongs to this exact configuration."""
    config_path = run_dir / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != json.loads(cfg.to_json()):
            raise RunConflict(f"{run_dir} holds a different configuration")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(config_path, cfg.to_json())


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


def _stop_reason(state: dict, cfg: RunConfig) -> str | None:
    if state["bad"] >= cfg.patience:
        return "patience"
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


def max_len_of(cfg: RunConfig) -> int:
    return cfg.max_len or DATASETS[cfg.dataset].max_len


def execute(
    cfg: RunConfig, runs_dir: Path, data_dir: Path, device: torch.device, log=print
) -> dict:
    """Train until a stop rule fires and write summary.json. No test data are read here."""
    run_dir = run_dir_of(cfg, runs_dir)
    summary_path = run_dir / "summary.json"
    _claim(cfg, run_dir)
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    commit = git_commit()
    seed_everything(cfg.seed)
    splits, checks = _prepare_splits(cfg, data_dir)
    if len(splits.validation) == 0:
        raise ValueError(f"{cfg.dataset}/{cfg.split}: empty validation split")
    max_len = max_len_of(cfg)
    model = ARMS[cfg.arm](cfg, splits.train, device)
    if hasattr(model, "sample_positions"):
        checks = checks + check_batch_positions(splits, model.sample_positions())
    manifest = {
        "git_commit": commit,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "train_fingerprint": splits.train.fingerprint,
        "full_fingerprint": splits.full.fingerprint(),
        "hparams": getattr(model, "hparams", None),
        "invariants_passed": checks,
    }
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["train_fingerprint"] != manifest["train_fingerprint"]:
            raise RunConflict(f"{run_dir}: training data changed since the run started")
        manifest["sessions"] = old.get("sessions", [old["git_commit"]]) + [commit]
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
    _keep_metrics_before(metrics_path, state["epoch"])

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
                stats["train_peak_memory_gib"] = round(
                    torch.cuda.max_memory_allocated(device) / 2**30, 3
                )
            t_trained = time.perf_counter()
            try:
                val = evaluate(model.score, splits.validation, max_len, device)
            except NonFiniteScores:
                stop = "diverged"
                break
            state["train_s"] += t_trained - t_start
            state["wall_s"] += time.perf_counter() - t_start
            ndcg = val.metrics["ndcg@10"]
            if ndcg > state["best"]:
                state |= {"best": ndcg, "best_epoch": state["epoch"], "bad": 0}
                best_weights = model.state_dict()
            else:
                state["bad"] += 1
            record = {
                "epoch": state["epoch"],
                "train_s": round(state["train_s"], 3),
                "val_ndcg@10": ndcg,
                "val_hr@10": val.metrics["hr@10"],
                "val_tie_rate": val.tie_rate,
            } | stats
            state["epoch"] += 1
            # Checkpoint first, then log: a crash in between repeats the epoch cleanly.
            _save_atomic(
                {
                    "model": model.state_dict(),
                    "best_model": best_weights,
                    "state": state,
                    "rng": _rng_state(),
                },  # fmt: skip
                last_path,
            )
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            if stats.get("converged"):  # models without training, e.g. the floor baselines
                stop = "converged"
                break
    finally:
        samples = sampler.stop() if sampler and sampler.available else []
        if samples:
            with (run_dir / "nvml.jsonl").open("a", encoding="utf-8") as f:
                f.writelines(json.dumps(s) + "\n" for s in samples)

    summary = {
        "run_id": cfg.run_id,
        "config": json.loads(cfg.to_json()),
        "git_commit": commit,
        "status": "trained" if best_weights is not None else "diverged",
        "epochs": state["epoch"],
        "best_epoch": state["best_epoch"],
        "stop_reason": stop,
        "truncated": stop in ("max_epochs", "guard"),
        "train_s": round(state["train_s"], 2),
        "wall_s": round(state["wall_s"], 2),
    }
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


def test_run(cfg: RunConfig, runs_dir: Path, data_dir: Path, device: torch.device, log=print):
    """Read test data once for a trained final or anchor run and write test.json."""
    if cfg.phase not in ("final", "anchor"):
        raise PermissionError("only final and anchor runs are tested")
    run_dir = run_dir_of(cfg, runs_dir)
    _claim(cfg, run_dir)
    if (run_dir / "test.json").exists():
        return json.loads((run_dir / "test.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "trained":
        raise RuntimeError(f"{cfg.run_id} has no checkpoint ({summary['status']})")
    if cfg.phase == "anchor" and anchor_reads(runs_dir, cfg.dataset, cfg.arm) >= MAX_ANCHOR_READS:
        raise PermissionError(
            f"{cfg.arm} on {cfg.dataset} used its {MAX_ANCHOR_READS} anchor reads"
        )

    best_path = run_dir / "best.pt"
    checkpoint_hash = file_hash(best_path)
    if checkpoint_hash != summary["checkpoint_hash"]:
        raise RunConflict(f"{cfg.run_id}: checkpoint changed after training")
    seed_everything(cfg.seed)
    splits, _ = _prepare_splits(cfg, data_dir)
    model = ARMS[cfg.arm](cfg, splits.train, device)
    model.load_state_dict(torch.load(best_path, weights_only=False))
    access_log = run_dir / "test_access.jsonl"
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
