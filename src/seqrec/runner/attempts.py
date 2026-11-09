"""Failure records and the one-retry rule for training (D031).

Every failed attempt appends a line, with its traceback, to failures.jsonl in the run
directory; error.txt keeps the latest traceback. Training is retried once whatever the
cause: in the same process while the device still works, otherwise when the command is
rerun. A second failure is final. A run directory that belongs to another configuration
(RunConflict) is not a failed attempt. Test reads are never retried here (METHODS 11.6).
"""

from __future__ import annotations

import datetime as dt
import gc
import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

import torch

from seqrec.runner.run import RunConflict, git_commit

TRAIN_ATTEMPTS = 2


class TrainingFailed(RuntimeError):
    """Training failed on its retry too."""


class DeviceLost(RuntimeError):
    """The device stopped working after a failure; rerunning the command retries the run."""

    def __init__(self, run_id: str):
        super().__init__(f"{run_id}: the GPU stopped working; rerun the same command to retry")
        self.run_id = run_id


def failures(run_dir: Path) -> list[dict]:
    path = run_dir / "failures.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a record torn by a power loss
            continue
    return out


def train_failures(run_dir: Path) -> int:
    return sum(f["step"] == "train" for f in failures(run_dir))


def record_failure(run_dir: Path, step: str) -> None:
    """Record the exception being handled."""
    exc = sys.exception()
    message = str(exc).strip().splitlines()
    record = {
        "time_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "step": step,
        "error": f"{type(exc).__name__}: {message[0] if message else ''}",
        "git_commit": git_commit(),
        "traceback": traceback.format_exc(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    path = run_dir / "failures.jsonl"
    torn = path.exists() and not path.read_bytes().endswith(b"\n")  # a record cut short
    with path.open("a", encoding="utf-8") as f:
        f.write(("\n" if torn else "") + json.dumps(record) + "\n")


def device_usable(device: torch.device) -> bool:
    """Free what the failed attempt held, then check that the device still runs a kernel."""
    gc.collect()
    if device.type != "cuda":
        return True
    try:
        torch.cuda.empty_cache()
        torch.ones(1, device=device).add_(1).item()
        torch.cuda.synchronize(device)
        return True
    except Exception:
        return False


def train_with_retry[T](train: Callable[[], T], run_dir: Path, device: torch.device, log) -> T:
    """Call `train`; after a failure, retry once (training resumes from its last checkpoint)."""
    tried = 0
    while True:
        finished = (run_dir / "summary.json").exists()  # train() returns the summary
        if tried >= TRAIN_ATTEMPTS or (not finished and train_failures(run_dir) >= TRAIN_ATTEMPTS):
            raise TrainingFailed(f"{run_dir.name}: training failed {TRAIN_ATTEMPTS} times")
        tried += 1
        try:
            return train()
        except RunConflict:
            raise
        except Exception:
            record_failure(run_dir, "train")
            log(f"[retry] {run_dir.name}: training failed; see error.txt")
        # Outside the except block, so the failed attempt's frames and tensors can be freed.
        # A lost device stops the command even after a final failure, so that the next run
        # is not charged an attempt for it.
        if not device_usable(device):
            raise DeviceLost(run_dir.name)
