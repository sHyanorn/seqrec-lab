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
    return handle._split


def count_reads(log_path: Path) -> dict[str, int]:
    counts = {phase: 0 for phase in PHASES}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            counts[json.loads(line)["phase"]] += 1
    return counts
