"""Aggregate run directories into results/runs/ and check test-access counts.

Every run directory with a config is listed, including failed and unfinished ones
(METHODS 7.7). Test reads are recounted from test_access.jsonl.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from seqrec.data.stats import write_csv
from seqrec.eval.test_access import count_reads
from seqrec.runner.run import MAX_ANCHOR_READS

HARDWARE_EVENTS = ("hw_slowdown", "hw_thermal", "hw_power_brake")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _row(run_dir: Path) -> dict:
    cfg = _load(run_dir / "config.json")
    summary, test = _load(run_dir / "summary.json"), _load(run_dir / "test.json")
    reads = count_reads(run_dir / "test_access.jsonl")
    status = summary.get("status", "unfinished")
    if (run_dir / "error.txt").exists() and not test:
        status = "failed"
    elif test:
        status = "tested"
    nvml = summary.get("nvml") or {}
    shares = nvml.get("share_of_samples_with_reason", {})
    row = {
        "run_id": run_dir.name,
        "batch": cfg["batch"],
        "dataset": cfg["dataset"],
        "split": cfg["split"],
        "arm": cfg["arm"],
        "seed": cfg["seed"],
        "phase": cfg["phase"],
        "shuffle": cfg["shuffle"],
        "params": json.dumps(cfg["params"], sort_keys=True),
        "status": status,
        "epochs": summary.get("epochs"),
        "best_epoch": summary.get("best_epoch"),
        "stop_reason": summary.get("stop_reason"),
        "truncated": summary.get("truncated"),
        "train_s": summary.get("train_s"),
        "gpu_hours": round(summary["wall_s"] / 3600, 4) if "wall_s" in summary else None,
        "git_commit": summary.get("git_commit", "")[:12],
        "sm_mhz_median": (nvml.get("sm_mhz") or {}).get("median"),
        "hardware_slowdown": any(shares.get(k, 0) > 0 for k in HARDWARE_EVENTS),
        "reads_final": reads["final"],
        "reads_anchor": reads["anchor"],
        "reads_completed": reads["completed"],
    }
    for part, values in (("validation", summary.get("validation")),) + tuple(test.items()):
        if isinstance(values, dict):
            row |= {f"{part}_{k}": v for k, v in values.items() if k != "users"}
    return row


def access_problems(rows: list[dict]) -> list[str]:
    problems = []
    for r in rows:
        reads = r["reads_final"] + r["reads_anchor"]
        if r["phase"] == "tune" and reads:
            problems.append(f"{r['run_id']}: tuning run read test data")
        if r["phase"] == "final" and (r["reads_anchor"] or r["reads_final"] > 1):
            problems.append(f"{r['run_id']}: final run read test data more than once")
        if r["phase"] == "anchor" and r["reads_final"]:
            problems.append(f"{r['run_id']}: anchor run used a final read")
        if r["status"] == "tested" and reads != r["reads_completed"]:
            problems.append(f"{r['run_id']}: a test read did not complete")
    anchors = Counter()
    for r in rows:
        anchors[(r["dataset"], r["arm"])] += r["reads_anchor"]
    problems += [
        f"{arm} on {ds}: {n} anchor reads (limit {MAX_ANCHOR_READS})"
        for (ds, arm), n in anchors.items()
        if n > MAX_ANCHOR_READS
    ]
    return problems


def _overview(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["batch"], r["dataset"], r["split"], r["arm"], r["phase"])].append(r)
    out = []
    for (batch, dataset, split, arm, phase), rs in sorted(groups.items()):
        done = [r for r in rs if r["truncated"] is not None]
        out.append(
            {
                "batch": batch, "dataset": dataset, "split": split, "arm": arm, "phase": phase,
                "runs": len(rs),
                "unfinished_or_failed": sum(r["status"] in ("unfinished", "failed") for r in rs),
                "truncated_share": round(sum(r["truncated"] for r in done) / len(done), 3)
                if done else None,
                "gpu_hours": round(sum(r["gpu_hours"] or 0 for r in rs), 3),
            }
        )  # fmt: skip
    return out


def export(runs_dir: Path, out_dir: Path) -> list[str]:
    """Write one CSV per batch plus overview.csv; return any test-access problems."""
    rows = [_row(p.parent) for p in sorted(runs_dir.glob("*/*/config.json"))]
    for batch in sorted({r["batch"] for r in rows}):
        write_csv(out_dir / f"{batch}.csv", [r for r in rows if r["batch"] == batch])
    if rows:
        write_csv(out_dir / "overview.csv", _overview(rows))
    return access_problems(rows)
