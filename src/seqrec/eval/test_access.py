"""The only way to reach test data (METHODS 11.6). Every read is logged, and anchor reads
are limited here, so the limit binds every caller."""

from __future__ import annotations

import datetime as dt
import inspect
import json
from pathlib import Path

from seqrec.data.splits import EvalSplit, TestHandle

PHASES = ("final", "anchor")
MAX_ANCHOR_READS = 3  # per arm and dataset (METHODS 11.5)


class TestRefused(PermissionError):
    """A test read refused before any data were read (METHODS 11.5, 11.6)."""


def anchor_reads(runs_dir: Path, dataset: str, arm: str) -> int:
    """All anchor reads logged for (dataset, arm) in any batch, finished or not."""
    total = 0
    for log_path in runs_dir.glob("*/*/test_access.jsonl"):
        cfg = json.loads((log_path.parent / "config.json").read_text(encoding="utf-8"))
        if cfg["dataset"] == dataset and cfg["arm"] == arm:
            total += count_reads(log_path)["anchor"]
    return total


def read_test(
    handle: TestHandle,
    *,
    phase: str,
    log_path: Path,
    run_id: str,
    git_commit: str,
    checkpoint_hash: str,
    earlier_anchor_reads: int,
) -> EvalSplit:
    """`log_path` is `<runs>/<batch>/<run>/test_access.jsonl`. An anchor read is refused once
    its arm and dataset have MAX_ANCHOR_READS reads: those logged under `<runs>`, plus
    `earlier_anchor_reads` made on another machine (D045)."""
    if phase not in PHASES:
        raise PermissionError(f"test data are readable only in phases {PHASES}, not {phase!r}")
    if phase == "anchor":
        run = json.loads((log_path.parent / "config.json").read_text(encoding="utf-8"))
        dataset, arm = run["dataset"], run["arm"]
        used = anchor_reads(log_path.parents[2], dataset, arm) + earlier_anchor_reads
        if used >= MAX_ANCHOR_READS:
            raise TestRefused(f"{arm} on {dataset} used its {MAX_ANCHOR_READS} anchor reads")
    caller = inspect.stack()[1]
    record = {
        "time_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id,
        "phase": phase,
        "caller": f"{caller.frame.f_globals.get('__name__')}:{caller.function}:{caller.lineno}",
        "git_commit": git_commit,
        "checkpoint_hash": checkpoint_hash,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return handle.unseal("read")


def mark_completed(log_path: Path, run_id: str) -> None:
    """Record that the scores from the last read were written."""
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": run_id, "event": "completed"}) + "\n")


def count_reads(log_path: Path) -> dict[str, int]:
    """Reads per phase, and how many of them finished writing their scores."""
    counts = {phase: 0 for phase in PHASES} | {"completed": 0}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("event") == "completed":
                counts["completed"] += 1
            else:
                counts[record["phase"]] += 1
    return counts
