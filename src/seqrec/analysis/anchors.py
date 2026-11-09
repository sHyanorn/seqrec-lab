"""Anchor runs against their published bands (METHODS 11.5, D032), and what the METHODS 5.2
limits cost against the source's training length (D033). Both read exported batch CSVs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seqrec.analysis.power import DELTA, sigma_from_export
from seqrec.analysis.table import format_table
from seqrec.runner.config import Plan, RunConfig, load_plan

ML1M_USERS = 6040  # every MovieLens-1M user has a validation and a test target (D022)
GAP_SE = 3.0  # D032: test and validation NDCG@10 further apart than this many SE are flagged
RAISE_Z = -2.0  # D033: with a drop beyond DELTA, a cap increase is recommended below this z


@dataclass(frozen=True)
class Anchor:
    source: str
    published: float
    low: float
    high: float


ANCHORS = {  # METHODS 11.5
    "sasrec-ce": Anchor("[5]", 0.1821, 0.172, 0.192),
    "sasrec-bce": Anchor("[5]", 0.1341, 0.124, 0.144),
    "bert4rec": Anchor("[5]", 0.1537, 0.144, 0.164),
    "gsasrec": Anchor("[33]", 0.1726, 0.163, 0.183),
}


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(row: dict, key: str) -> float | None:
    value = row.get(key)
    return float(value) if value not in (None, "") else None


def position(value: float | None, anchor: Anchor) -> str:
    if value is None:
        return "missing"
    if value > anchor.high:
        return "above"
    return "below" if value < anchor.low else "inside"


def gap_margin(validation: float, test: float, users: int = ML1M_USERS) -> float:
    """GAP_SE standard errors of test minus validation NDCG@10. Per-user NDCG lies in [0, 1],
    so a mean m over U users has variance at most m(1 - m)/U; the two means are treated as
    independent."""
    return GAP_SE * math.sqrt((validation * (1 - validation) + test * (1 - test)) / users)


def _arm_order(arm: str) -> int:
    return list(ANCHORS).index(arm) if arm in ANCHORS else len(ANCHORS)


def _anchor_rows(path: Path) -> list[dict]:
    rows = [r for r in _read(path) if r["phase"] == "anchor" and r["dataset"] == "ml-1m"]
    return sorted(rows, key=lambda r: (_arm_order(r["arm"]), r["tag"]))


def anchor_table(path: Path) -> list[dict]:
    out = []
    for r in _anchor_rows(path):
        anchor = ANCHORS.get(r["arm"])
        test, val = _num(r, "test_ndcg@10"), _num(r, "validation_ndcg@10")
        gap = margin = None
        flag = ""
        if test is not None and val is not None:
            gap, margin = test - val, gap_margin(val, test)
            if abs(gap) > margin:
                flag = "test above validation" if gap > 0 else "test below validation"
        best = _num(r, "best_epoch")
        out.append(
            {
                "arm": r["arm"], "tag": r["tag"], "status": r["status"],
                "source": anchor and anchor.source,
                "published": anchor and anchor.published,
                "band": anchor and f"{anchor.low:.3f}-{anchor.high:.3f}",
                "test_ndcg@10": test,
                "position": position(test, anchor) if anchor else "no anchor",
                "validation_ndcg@10": val,
                "test_minus_validation": gap, "gap_margin": margin, "gap_flag": flag,
                "epochs": r.get("epochs"),
                "best_epoch": None if best is None else int(best) + 1,  # 1-based, as in [5]
                "stop_reason": r.get("stop_reason"), "truncated": r.get("truncated"),
                "test_tie_rate": _num(r, "test_tie_rate"),
                "validation_tie_rate": _num(r, "validation_tie_rate"),
                "test_hr@10": _num(r, "test_hr@10"),
                "shuffled_test_ndcg@10": _num(r, "test_inputs_shuffled_ndcg@10"),
                "reads_anchor": r.get("reads_anchor"), "reads_completed": r.get("reads_completed"),
                "reads_final": r.get("reads_final"), "failures": r.get("failures", "-"),
                "hardware_slowdown": r.get("hardware_slowdown"),
                "git_commit": r.get("git_commit"), "run_id": r.get("run_id"),
                "params": r.get("params"),
                "max_epochs": r.get("max_epochs"), "patience": r.get("patience"),
                "guard_hours": r.get("guard_hours"),
            }
        )  # fmt: skip
    return out


def _plan_mismatch(row: dict, planned: RunConfig | None) -> str | None:
    if planned is None:
        return "not in the anchor plan"
    if json.loads(row["params"] or "{}") != planned.params:
        return "parameters differ from the anchor plan"
    for key in ("max_epochs", "patience", "guard_hours"):
        if row[key] not in (None, "") and float(row[key]) != getattr(planned, key):
            return f"{key} differs from the anchor plan"
    return None


def stop_conditions(table: list[dict], plan: Plan) -> list[str]:
    """The D032 stop conditions, one line each, for the latest anchor run of each arm; an
    earlier run (before a permitted re-read, METHODS 11.5) stays in the table as history."""
    planned = {r.run_id: r for r in plan.runs}
    latest = list({r["arm"]: r for r in table}.values())  # the table is sorted by tag
    issues = [f"{arm}: no anchor run" for arm in ANCHORS if arm not in {r["arm"] for r in table}]
    for r in latest:
        name = f"{r['arm']} {r['tag']}"
        if r["status"] != "tested":
            issues.append(f"{name}: status {r['status']}")
        if str(r["reads_anchor"]) != "1" or str(r["reads_completed"]) != "1":
            issues.append(f"{name}: {r['reads_anchor']} anchor reads, {r['reads_completed']} done")
        if str(r["reads_final"]) != "0":
            issues.append(f"{name}: final reads on an anchor run")
        if mismatch := _plan_mismatch(r, planned.get(r["run_id"])):
            issues.append(f"{name}: {mismatch}")
        if r["position"] in ("above", "below"):
            issues.append(f"{name}: test NDCG@10 {r['position']} the band {r['band']}")
        if r["gap_flag"]:
            issues.append(f"{name}: {r['gap_flag']} by more than {r['gap_margin']:.4f}")
    commits = sorted({r["git_commit"] for r in latest})
    if len(commits) > 1:
        issues.append(f"anchor runs come from several commits: {', '.join(commits)}")
    return issues


def cap_cost(anchor_csv: Path, pilot_csv: Path, tag: str = "sigma") -> tuple[float, list[dict]]:
    """Per arm: validation NDCG@10 of the seed-variance runs (METHODS 5.2 limits) against the
    latest trained anchor run (the source's limits). z divides the log difference by its
    standard error under the pooled sigma: sqrt(1 + 1/k) sigma for one anchor and k runs."""
    sigma, groups = sigma_from_export(pilot_csv, tag, "ml-1m")
    anchors = {}
    for r in _anchor_rows(anchor_csv):
        trained = r["status"] in ("trained", "tested", "test_failed")
        if trained and _num(r, "validation_ndcg@10") is not None:
            anchors[r["arm"]] = _num(r, "validation_ndcg@10")  # sorted by tag: latest wins
    runs = [
        r for r in _read(pilot_csv)
        if r["tag"] == tag and r["dataset"] == "ml-1m" and r["split"] == "loo"
        and r["shuffle"] == "none" and r["status"] in ("trained", "tested")
    ]  # fmt: skip
    out = []
    for arm, logs in sorted(groups.items(), key=lambda kv: _arm_order(kv[0])):
        mine = [r for r in runs if r["arm"] == arm]
        row = {
            "arm": arm, "runs": len(logs),
            "mean_validation": float(np.mean(np.exp(logs))),
            "epochs_mean": float(np.mean([float(r["epochs"]) for r in mine])),
            "truncated_runs": sum(r["truncated"] == "True" for r in mine),
            "anchor_validation": anchors.get(arm),
        }  # fmt: skip
        if arm in anchors:
            diff = float(np.mean(logs)) - math.log(anchors[arm])
            se = sigma * math.sqrt(1 + 1 / len(logs))
            z = diff / se if se > 0 else math.copysign(math.inf, diff)
            row |= {
                "log_diff": diff, "relative": math.exp(diff) - 1, "z": z,
                "raise_cap": diff < -DELTA and z < RAISE_Z,
            }  # fmt: skip
        out.append(row)
    return sigma, out


def report(anchor_csv: Path, plan_path: Path, pilot_csv: Path | None = None) -> str:
    table = anchor_table(anchor_csv)
    parts = [
        format_table(table, [
            "arm", "tag", "test_ndcg@10", "band", "position", "validation_ndcg@10",
            "test_minus_validation", "gap_margin", "gap_flag", "epochs", "best_epoch",
            "stop_reason", "truncated", "test_tie_rate", "validation_tie_rate",
        ]),
        format_table(table, [
            "arm", "tag", "status", "test_hr@10", "shuffled_test_ndcg@10", "reads_anchor",
            "reads_completed", "reads_final", "failures", "hardware_slowdown", "git_commit",
        ]),
    ]  # fmt: skip
    issues = stop_conditions(table, load_plan(plan_path))
    parts.append("\n".join(f"STOP {i}" for i in issues) or "no D032 stop condition")
    if pilot_csv is not None:
        sigma, rows = cap_cost(anchor_csv, pilot_csv)
        parts.append(f"pooled sigma of log validation NDCG@10: {sigma:.5f}")
        parts.append(
            format_table(rows, [
                "arm", "runs", "mean_validation", "anchor_validation", "relative", "z",
                "raise_cap", "epochs_mean", "truncated_runs",
            ])
        )  # fmt: skip
    return "\n\n".join(parts)
