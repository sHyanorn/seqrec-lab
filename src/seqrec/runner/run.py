"""Execute one run: train with early stopping on validation, then read test data once.

Run directory (runs/<batch>/<run_id>/):
    config.json, manifest.json   what was run, on which code and data, invariants passed
    metrics.jsonl                one line per epoch
    last.pt, best.pt             resume state and the selected checkpoint (local only)
    nvml.jsonl                   GPU telemetry at 1 Hz
    test_access.jsonl            every test read
    summary.json                 written last; its presence marks the run as finished
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from seqrec.data.invariants import check_split
from seqrec.data.registry import DATASETS
from seqrec.data.splits import SplitSet, TrainSplit, shuffle_eval_inputs, shuffle_order
from seqrec.data.store import load_splits
from seqrec.eval.ranking import evaluate
from seqrec.eval.test_access import count_reads, read_test
from seqrec.runner.config import RunConfig
from seqrec.telemetry import NvmlSampler, summarize_samples

REPO_ROOT = Path(__file__).resolve().parents[3]


class Model(Protocol):
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


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _stop_reason(state: dict, cfg: RunConfig) -> str | None:
    if state["bad"] >= cfg.patience:
        return "patience"
    if state["wall_s"] >= cfg.guard_hours * 3600:
        return "guard"
    if state["epoch"] >= cfg.max_epochs:
        return "max_epochs"
    return None


def _prepare_splits(cfg: RunConfig, data_dir: Path) -> tuple[SplitSet, list[str]]:
    splits, checks = load_splits(data_dir, cfg.dataset, cfg.split)
    if cfg.shuffle == "train_eval":
        splits = shuffle_order(splits, seed=cfg.seed)
        checks = checks + check_split(splits)
    return splits, checks


def execute(
    cfg: RunConfig, runs_dir: Path, data_dir: Path, device: torch.device, log=print
) -> dict:
    run_dir = runs_dir / cfg.batch / cfg.run_id
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(run_dir / "config.json", cfg.to_json())

    commit = git_commit()
    seed_everything(cfg.seed)
    splits, checks = _prepare_splits(cfg, data_dir)
    max_len = cfg.max_len or DATASETS[cfg.dataset].max_len
    manifest = {
        "git_commit": commit,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "train_fingerprint": splits.train.fingerprint,
        "full_fingerprint": splits.full.fingerprint(),
        "invariants_passed": checks,
    }
    _write_atomic(run_dir / "manifest.json", json.dumps(manifest, indent=1))

    model = ARMS[cfg.arm](cfg, splits.train, device)
    state = {"epoch": 0, "best": -1.0, "best_epoch": -1, "bad": 0, "train_s": 0.0, "wall_s": 0.0}
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if last_path.exists():
        saved = torch.load(last_path, weights_only=False)
        model.load_state_dict(saved["model"])
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        state = saved["state"]
        log(f"[run] {cfg.run_id}: resuming at epoch {state['epoch']}")

    sampler = NvmlSampler(interval_s=1.0) if device.type == "cuda" else None
    if sampler and sampler.available:
        sampler.start()
    try:
        while (stop_reason := _stop_reason(state, cfg)) is None:
            t_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            stats = model.train_epoch(splits.train, state["epoch"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_trained = time.perf_counter()
            val = evaluate(model.score, splits.validation, max_len, device)
            state["train_s"] += t_trained - t_start
            state["wall_s"] += time.perf_counter() - t_start
            ndcg = val.metrics["ndcg@10"]
            record = {
                "epoch": state["epoch"],
                "train_s": round(state["train_s"], 3),
                "val_ndcg@10": ndcg,
                "val_hr@10": val.metrics["hr@10"],
                "val_tie_rate": val.tie_rate,
            }
            if device.type == "cuda":
                record["peak_memory_gib"] = round(
                    torch.cuda.max_memory_allocated(device) / 2**30, 3
                )
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record | stats) + "\n")
            if ndcg > state["best"]:
                state |= {"best": ndcg, "best_epoch": state["epoch"], "bad": 0}
                torch.save(model.state_dict(), best_path)
            else:
                state["bad"] += 1
            state["epoch"] += 1
            torch.save(
                {
                    "model": model.state_dict(),
                    "state": state,
                    "torch_rng": torch.get_rng_state(),
                    "numpy_rng": np.random.get_state(),
                },
                last_path,
            )
    finally:
        samples = sampler.stop() if sampler and sampler.available else []
    if samples:
        with (run_dir / "nvml.jsonl").open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(s) + "\n" for s in samples)

    model.load_state_dict(torch.load(best_path, weights_only=False))
    checkpoint_hash = file_hash(best_path)
    val = evaluate(model.score, splits.validation, max_len, device)
    summary = {
        "run_id": cfg.run_id,
        "config": json.loads(cfg.to_json()),
        "git_commit": commit,
        "epochs": state["epoch"],
        "best_epoch": state["best_epoch"],
        "stop_reason": stop_reason,
        "truncated": stop_reason != "patience",
        "train_s": round(state["train_s"], 2),
        "wall_s": round(state["wall_s"], 2),
        "checkpoint_hash": checkpoint_hash,
        "validation": val.metrics | {"tie_rate": val.tie_rate, "users": val.users},
    }
    if cfg.phase in ("final", "anchor"):
        test = read_test(
            splits.test,
            phase=cfg.phase,
            log_path=run_dir / "test_access.jsonl",
            run_id=cfg.run_id,
            git_commit=commit,
            checkpoint_hash=checkpoint_hash,
        )
        result = evaluate(model.score, test, max_len, device)
        shuffled = evaluate(
            model.score,
            shuffle_eval_inputs(test, cfg.seed + 1, len(splits.full.items)),
            max_len,
            device,
        )
        summary["test"] = result.metrics | {"tie_rate": result.tie_rate, "users": result.users}
        summary["test_inputs_shuffled"] = shuffled.metrics
    if samples:
        summary["nvml"] = summarize_samples(sampler.nv, samples)
    summary["test_reads"] = count_reads(run_dir / "test_access.jsonl")
    _write_atomic(summary_path, json.dumps(summary, indent=1))
    log(f"[run] {cfg.run_id}: {stop_reason} after {state['epoch']} epochs")
    return summary
