"""Aggregate run directories into results/runs/ and check test-access counts.

Every run directory with a config is listed, including failed and unfinished ones
(METHODS 7.7). Test reads are recounted from test_access.jsonl. One file per batch holds the
runs; overview.csv and tuning_best.csv are computed from all batch files in the directory,
so they also cover batches whose run directories are on another machine.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from seqrec.data.stats import write_csv
from seqrec.eval.test_access import MAX_ANCHOR_READS, TestRefused, count_reads
from seqrec.runner.attempts import failures

HARDWARE_EVENTS = ("hw_slowdown", "hw_thermal", "hw_power_brake")
TRIAL_TAG = re.compile(r"t\d+")
STOPPED = ("patience", "max_epochs", "guard", "no_start")
SLOW_FACTOR = 1.5  # D038
SUMMARIES = ("overview.csv", "tuning_best.csv")
# Columns of a batch file that the summaries and the access checks read, and how each comes
# back from text. An empty field is None.
TEXT = ("run_id", "batch", "dataset", "split", "arm", "phase", "tag", "status", "stop_reason",
        "params")  # fmt: skip
PARSED = {
    "truncated": {"True": True, "False": False}.__getitem__,
    "gpu_hours": float,
    "validation_ndcg@10": float,
    "reads_final": int,
    "reads_anchor": int,
    "reads_completed": int,
}
OPTIONAL = ("validation_ndcg@10",)  # absent when no run of the batch finished training


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
        "start_limit": summary.get("start_limit", cfg.get("start_limit")),
        "guard_hours": cfg.get("guard_hours"),
        "status": status,
        "failures": n_failures,  # failed attempts, retries included (D031)
        "epochs": summary.get("epochs"),
        "best_epoch": summary.get("best_epoch"),
        "stop_reason": summary.get("stop_reason"),
        "truncated": summary.get("truncated"),
        "start_epoch": summary.get("start_epoch"),
        "train_s": summary.get("train_s"),
        "epoch_train_s_median": summary.get("epoch_train_s_median"),
        "epoch_eval_s_median": summary.get("epoch_eval_s_median"),
        "checkpoint_s": summary.get("checkpoint_s"),
        "gpu_hours": round(summary["wall_s"] / 3600, 4) if "wall_s" in summary else None,
        "git_commit": summary.get("git_commit", "")[:12],
        "host": summary.get("host"),
        "pod": summary.get("pod"),
        "sm_mhz_median": (nvml.get("sm_mhz") or {}).get("median"),
        "nvml_mem_used_mib_max": (nvml.get("mem_used_mib") or {}).get("max"),
        "torch_allocated_gib_max": summary.get("train_peak_memory_gib_max"),
        "torch_reserved_gib_max": summary.get("train_peak_reserved_gib_max"),
        "hardware_slowdown": any(shares.get(k, 0) > 0 for k in HARDWARE_EVENTS),
        "reads_final": reads["final"],
        "reads_anchor": reads["anchor"],
        "reads_completed": reads["completed"],
    }
    for part, values in (("validation", summary.get("validation")),) + tuple(test.items()):
        if isinstance(values, dict):
            row |= {f"{part}_{k}": v for k, v in values.items() if k != "users"}
    return row


def flag_slow_epochs(rows: list[dict]) -> None:
    """Set `slow_epochs` on every row (D038): True when the run's median training plus
    validation time per epoch exceeds SLOW_FACTOR times the median of the other runs with
    the same batch, dataset, split, shuffle, arm, parameters, tag and Pod (its other seeds
    made on the same Pod, D046). None when the run has no epoch time or no such other run."""
    groups = defaultdict(list)
    for r in rows:
        r["slow_epochs"] = None
        if r["epoch_train_s_median"] is not None and r["epoch_eval_s_median"] is not None:
            key = tuple(
                r[k]
                for k in ("batch", "dataset", "split", "shuffle", "arm", "params", "tag", "pod")
            )
            groups[key].append(r)
    for runs in groups.values():
        times = [r["epoch_train_s_median"] + r["epoch_eval_s_median"] for r in runs]
        for i, r in enumerate(runs):
            others = times[:i] + times[i + 1 :]
            if others:
                r["slow_epochs"] = times[i] > SLOW_FACTOR * statistics.median(others)


def read_batch_files(directory: Path) -> list[dict]:
    """The rows of every batch file in a directory, with the columns named in TEXT and PARSED
    typed as `_row` wrote them. Any other CSV file there, apart from the summaries, is an
    error: skipping it would drop runs from the summaries and reads from the limits."""
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory of exported batch files")
    rows = []
    for path in sorted(directory.glob("*.csv")):
        if path.name in SUMMARIES:
            continue
        with path.open(encoding="utf-8", newline="") as f:
            n = 1
            for n, raw in enumerate(csv.DictReader(f), start=2):
                try:
                    row = {k: raw[k] for k in TEXT}
                    for k, parse in PARSED.items():
                        text = raw.get(k, "") if k in OPTIONAL else raw[k]
                        row[k] = None if text == "" else parse(text)
                    if any(
                        row[k] is None for k in ("reads_final", "reads_anchor", "reads_completed")
                    ):
                        raise ValueError("a read count is empty")
                except (KeyError, ValueError, TypeError) as err:
                    raise ValueError(f"{path}, line {n}: not a batch file ({err!r})") from err
                if row["batch"] != path.stem:
                    raise ValueError(f"{path}, line {n}: holds batch {row['batch']!r}")
                row["stop_reason"] = row["stop_reason"] or None
                rows.append(row)
            if n == 1:  # an export writes no file without runs: this one was cut short
                raise ValueError(f"{path}: no run in it, not a batch file")
    return rows


# Directories under results/ that are not exports of another machine: results/runs holds this
# machine's, the others hold no runs.
NOT_EXPORTS = ("runs", "data", "env")


def earlier_exports(results: Path = Path("results")) -> list[Path]:
    """The directories of batch files exported from another machine: results/laptop, there or
    not, and every other directory under results/ outside NOT_EXPORTS. A lost volume's batch
    files are set aside by moving them to a directory of their own (D045); none is named
    anywhere, so none can be left out."""
    found = {d for d in results.iterdir() if d.is_dir()} if results.is_dir() else set()
    return sorted({d for d in found if d.name not in NOT_EXPORTS} | {results / "laptop"})


def read_earlier_exports(directories: Iterable[Path]) -> list[dict]:
    """The rows of batch files exported from another machine. A directory without any is an
    error: its test reads would count as none."""
    rows = []
    for directory in directories:
        found = read_batch_files(directory)
        if not found:
            raise FileNotFoundError(
                f"{directory} holds no exported run: a directory under results/ either holds "
                "batch files exported from another machine or is named in NOT_EXPORTS "
                "(seqrec/runner/export.py)"
            )
        rows += found
    return rows


class UnbackedReads(TestRefused):
    """Batch files record test reads that the run directories they are checked against lack."""


def unbacked_reads(committed: Path, runs_dir: Path) -> list[str]:
    """Runs whose batch file in `committed` records more test reads than their run directory
    in `runs_dir` logged. Such a file was exported from another volume: its reads are counted
    neither here nor among the earlier exports, and an export from here would replace their
    record. A directory that does not exist yet holds no export."""
    if not committed.exists():
        return []
    problems = []
    for r in read_batch_files(committed):
        here = count_reads(runs_dir / r["batch"] / r["run_id"] / "test_access.jsonl")
        for phase in ("final", "anchor"):
            if r[f"reads_{phase}"] > here[phase]:
                problems.append(
                    f"{r['batch']}/{r['run_id']}: {r[f'reads_{phase}']} {phase} reads recorded, "
                    f"{here[phase]} in {runs_dir}"
                )
    return problems


def require_backed_reads(committed: Path, runs_dir: Path) -> None:
    """Stop unless every test read recorded in `committed` was made in `runs_dir`."""
    problems = unbacked_reads(committed, runs_dir)
    if problems:
        raise UnbackedReads(
            f"{committed} records test reads made on another volume, e.g. {problems[0]} "
            f"({len(problems)} in all). Move its batch files to results/<name>/, where they "
            "count as an earlier export (D045)"
        )


def access_problems(rows: list[dict], earlier: Iterable[dict] = ()) -> list[str]:
    """Test reads that break METHODS 11.5 and 11.6 among `rows`. `earlier` are runs exported
    from another machine: their anchor reads count toward the limit per dataset and arm, and
    a final run read there is not read again."""
    problems = []
    earlier = list(earlier)
    read_elsewhere = {r["run_id"] for r in earlier if r["reads_final"]}
    for r in rows:
        reads = r["reads_final"] + r["reads_anchor"]
        if r["phase"] == "tune" and reads:
            problems.append(f"{r['run_id']}: tuning run read test data")
        if r["phase"] == "final" and (r["reads_anchor"] or r["reads_final"] > 1):
            problems.append(f"{r['run_id']}: final run read test data more than once")
        if r["phase"] == "final" and r["reads_final"] and r["run_id"] in read_elsewhere:
            problems.append(f"{r['run_id']}: final run read test data on another machine too")
        if r["phase"] == "anchor" and r["reads_final"]:
            problems.append(f"{r['run_id']}: anchor run used a final read")
        if reads != r["reads_completed"]:
            problems.append(f"{r['run_id']}: a test read did not complete")
    anchors = Counter()
    for r in [*rows, *earlier]:
        anchors[(r["dataset"], r["arm"])] += r["reads_anchor"]
    problems += [
        f"{arm} on {ds}: {n} anchor reads (limit {MAX_ANCHOR_READS})"
        for (ds, arm), n in anchors.items()
        if n > MAX_ANCHOR_READS
    ]
    return problems


def _truncation(rows: list[dict]) -> dict:
    """Truncated runs among those that stopped by patience, the epoch cap, the guard or the
    start limit (D030, D035). Pruned, diverged and failed runs stopped before any limit could
    apply."""
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


def summarize(out_dir: Path, earlier: Sequence[Path] = ()) -> list[str]:
    """Write overview.csv and, after tuning, tuning_best.csv from the batch files in
    `out_dir`; return any test-access problems. `earlier` are directories of batch files
    exported from another machine (see `access_problems`)."""
    rows = read_batch_files(out_dir)
    earlier_rows = read_earlier_exports(earlier)
    if rows:
        write_csv(out_dir / "overview.csv", _overview(rows))
    best = tuning_best(rows)
    if best:
        write_csv(out_dir / "tuning_best.csv", best)
    return access_problems(rows, earlier_rows)


def export(
    runs_dir: Path, out_dir: Path, earlier: Sequence[Path] = (), written: Path | None = None
) -> list[str]:
    """Write one CSV per batch in `runs_dir`, then the summaries (`summarize`); return any
    test-access problems. Nothing is written over batch files that record test reads made
    elsewhere (`require_backed_reads`). `written` is a file to list the batch files in, one
    name per line."""
    require_backed_reads(out_dir, runs_dir)
    rows = [_row(p.parent) for p in sorted(runs_dir.glob("*/*/config.json"))]
    flag_slow_epochs(rows)
    batches = sorted({r["batch"] for r in rows})
    for batch in batches:
        write_csv(out_dir / f"{batch}.csv", [r for r in rows if r["batch"] == batch])
    out_dir.mkdir(parents=True, exist_ok=True)  # no run yet: nothing to summarise, no error
    if written:
        written.write_text("".join(f"{b}.csv\n" for b in batches), encoding="utf-8")
    return summarize(out_dir, earlier)
