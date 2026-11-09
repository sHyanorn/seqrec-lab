"""Aggregate run directories into results/runs/ and check test-access counts.

Every run directory with a config is listed, including failed and unfinished ones
(METHODS 7.7). Test reads are recounted from test_access.jsonl.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from seqrec.data.stats import write_csv
from seqrec.eval.test_access import count_reads
from seqrec.runner.attempts import failures
from seqrec.runner.run import MAX_ANCHOR_READS

HARDWARE_EVENTS = ("hw_slowdown", "hw_thermal", "hw_power_brake")
TRIAL_TAG = re.compile(r"t\d+")
STOPPED = ("patience", "max_epochs", "guard")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _status(summary: dict, test: dict, failed: bool, incomplete_read: bool) -> str:
    if test:
        return "tested"
    if incomplete_read:  # a test read started and never finished
        return "test_failed"
    if summary:
        return summary["status"]
    return "failed" if failed else "unfinished"


def _row(run_dir: Path) -> dict:
    cfg = _load(run_dir / "config.json")
    summary, test = _load(run_dir / "summary.json"), _load(run_dir / "test.json")
    reads = count_reads(run_dir / "test_access.jsonl")
    n_failures = len(failures(run_dir))
    if not n_failures and (run_dir / "error.txt").exists():  # written before failures.jsonl
        n_failures = 1
    incomplete = reads["final"] + reads["anchor"] > reads["completed"]
    status = _status(summary, test, n_failures > 0, incomplete)
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
        "tag": cfg.get("tag", ""),
        "params": json.dumps(cfg["params"], sort_keys=True),
        "max_epochs": cfg.get("max_epochs"),
        "patience": cfg.get("patience"),
        "guard_hours": cfg.get("guard_hours"),
        "status": status,
        "failures": n_failures,  # failed attempts, retries included (D031)
        "epochs": summary.get("epochs"),
        "best_epoch": summary.get("best_epoch"),
        "stop_reason": summary.get("stop_reason"),
        "truncated": summary.get("truncated"),
        "train_s": summary.get("train_s"),
        "epoch_train_s_median": summary.get("epoch_train_s_median"),
        "epoch_eval_s_median": summary.get("epoch_eval_s_median"),
        "checkpoint_s": summary.get("checkpoint_s"),
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
        if reads != r["reads_completed"]:
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


def _truncation(rows: list[dict]) -> dict:
    """Truncated runs among those that stopped by patience, the epoch cap or the guard (D030).
    Pruned, diverged and failed runs stopped before any limit could apply."""
    stopped = [r for r in rows if r["stop_reason"] in STOPPED]
    truncated = sum(bool(r["truncated"]) for r in stopped)
    return {
        "stopped": len(stopped),
        "truncated": truncated,
        "truncated_share": round(truncated / len(stopped), 3) if stopped else None,
    }


def _stop_counts(rows: list[dict]) -> dict:
    return {
        "pruned": sum(r["stop_reason"] == "pruned" for r in rows),
        "diverged": sum(r["stop_reason"] == "diverged" for r in rows),
    }


def _overview(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:  # tuning trials form one group; other tags (e.g. sigma, time-small) their own
        tag = "trials" if TRIAL_TAG.fullmatch(r["tag"]) else r["tag"]
        groups[(r["batch"], r["dataset"], r["split"], r["arm"], r["phase"], tag)].append(r)
    out = []
    for (batch, dataset, split, arm, phase, tag), rs in sorted(groups.items()):
        out.append(
            {
                "batch": batch, "dataset": dataset, "split": split, "arm": arm, "phase": phase,
                "tag": tag, "runs": len(rs),
                "unfinished_or_failed": sum(
                    r["status"] in ("unfinished", "failed", "test_failed") for r in rs
                ),
            }
            | _stop_counts(rs)
            | _truncation(rs)
            | {"gpu_hours": round(sum(r["gpu_hours"] or 0 for r in rs), 3)}
        )  # fmt: skip
    return out


def tuning_best(rows: list[dict]) -> list[dict]:
    """Per study, the trial with the highest restored-checkpoint validation NDCG@10 among
    trials that finished training without being pruned; ties go to the earlier trial."""
    studies = defaultdict(list)
    for r in rows:
        if r["phase"] == "tune" and TRIAL_TAG.fullmatch(r["tag"]):
            studies[(r["batch"], r["dataset"], r["arm"])].append(r)
    out = []
    for (batch, dataset, arm), trials in sorted(studies.items()):
        eligible = [
            r for r in trials
            if r["status"] == "trained" and r["stop_reason"] != "pruned"
            and r.get("validation_ndcg@10") is not None
        ]  # fmt: skip
        best = min(
            eligible, key=lambda r: (-r["validation_ndcg@10"], int(r["tag"][1:])), default=None
        )
        out.append(
            {"batch": batch, "dataset": dataset, "arm": arm, "trials": len(trials)}
            | _stop_counts(trials)
            | {"failed_or_unfinished": sum(r["stop_reason"] is None for r in trials)}
            | _truncation(trials)  # the METHODS 5.2 rule reads truncated_share (D030)
            | {
                "best_run_id": best and best["run_id"],
                "best_validation_ndcg@10": best and best["validation_ndcg@10"],
                "best_params": best and best["params"],
            }
        )  # fmt: skip
    return out


def export(runs_dir: Path, out_dir: Path) -> list[str]:
    """Write one CSV per batch, overview.csv and, after tuning, tuning_best.csv; return any
    test-access problems."""
    rows = [_row(p.parent) for p in sorted(runs_dir.glob("*/*/config.json"))]
    for batch in sorted({r["batch"] for r in rows}):
        write_csv(out_dir / f"{batch}.csv", [r for r in rows if r["batch"] == batch])
    if rows:
        write_csv(out_dir / "overview.csv", _overview(rows))
    best = tuning_best(rows)
    if best:
        write_csv(out_dir / "tuning_best.csv", best)
    return access_problems(rows)
