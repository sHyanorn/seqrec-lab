"""The only way to reach test data (METHODS 11.6). Every read is logged."""

from __future__ import annotations

import datetime as dt
import inspect
import json
from pathlib import Path

from seqrec.data.splits import EvalSplit, TestHandle

PHASES = ("final", "anchor")


def read_test(
    handle: TestHandle,
    *,
    phase: str,
    log_path: Path,
    run_id: str,
    git_commit: str,
    checkpoint_hash: str,
) -> EvalSplit:
    if phase not in PHASES:
        raise PermissionError(f"test data are readable only in phases {PHASES}, not {phase!r}")
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
