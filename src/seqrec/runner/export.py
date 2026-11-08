"""Aggregate finished runs into results/runs/<batch>.csv and check test-access counts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from seqrec.data.stats import write_csv

MAX_ANCHOR_READS = 3
HARDWARE_EVENTS = ("hw_slowdown", "hw_thermal", "hw_power_brake")


def _row(summary: dict) -> dict:
    cfg = summary["config"]
    nvml = summary.get("nvml") or {}
    shares = nvml.get("share_of_samples_with_reason", {})
    row = {
        "run_id": summary["run_id"],
        "batch": cfg["batch"],
        "dataset": cfg["dataset"],
        "split": cfg["split"],
        "arm": cfg["arm"],
        "seed": cfg["seed"],
        "phase": cfg["phase"],
        "shuffle": cfg["shuffle"],
        "params": json.dumps(cfg["params"], sort_keys=True),
        "epochs": summary["epochs"],
        "best_epoch": summary["best_epoch"],
        "stop_reason": summary["stop_reason"],
        "truncated": summary["truncated"],
        "train_s": summary["train_s"],
        "gpu_hours": round(summary["wall_s"] / 3600, 4),
        "git_commit": summary["git_commit"][:12],
        "sm_mhz_median": (nvml.get("sm_mhz") or {}).get("median"),
        "hardware_slowdown": any(shares.get(k, 0) > 0 for k in HARDWARE_EVENTS),
        "reads_final": summary["test_reads"]["final"],
        "reads_anchor": summary["test_reads"]["anchor"],
    }
    for part in ("validation", "test", "test_inputs_shuffled"):
        for name, value in (summary.get(part) or {}).items():
            if name != "users":
                row[f"{part}_{name}"] = value
    return row


def access_problems(rows: list[dict]) -> list[str]:
    problems = []
    for r in rows:
        if r["phase"] == "tune" and r["reads_final"] + r["reads_anchor"]:
            problems.append(f"{r['run_id']}: tuning run read test data")
        if r["phase"] == "final" and (r["reads_final"] != 1 or r["reads_anchor"]):
            problems.append(f"{r['run_id']}: final run must read test data exactly once")
    anchors = Counter()
    for r in rows:
        anchors[(r["dataset"], r["arm"])] += r["reads_anchor"]
    problems += [
        f"{arm} on {ds}: {n} anchor reads (limit {MAX_ANCHOR_READS})"
        for (ds, arm), n in anchors.items()
        if n > MAX_ANCHOR_READS
    ]
    return problems


def export(runs_dir: Path, out_dir: Path) -> list[str]:
    rows = [
        _row(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(runs_dir.glob("*/*/summary.json"))
    ]
    problems = access_problems(rows)
    for batch in sorted({r["batch"] for r in rows}):
        write_csv(out_dir / f"{batch}.csv", [r for r in rows if r["batch"] == batch])
    return problems
