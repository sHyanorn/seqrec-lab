"""Aggregate dataset statistics and the comparison with published values.

Only aggregates are written; `results_are_aggregate` guards that no row-level data
reaches results/.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from seqrec.data.preprocess import Log
from seqrec.data.splits import SplitSet

FORBIDDEN_COLUMNS = {
    "user", "user_id", "username", "item", "item_id", "product_id", "parent_asin", "ts",
    "timestamp", "target", "targets", "history",
}  # fmt: skip
MAX_RESULT_ROWS = 5000  # far below any per-user table


def log_stats(dataset: str, log: Log, read_info: dict) -> dict:
    return {
        "dataset": dataset,
        "split": "log",
        "rows_read": read_info.get("rows_read"),
        "rows_skipped": read_info.get("rows_skipped"),
        "users": len(log.user_keys),
        "items": len(log.item_keys),
        "interactions": len(log),
        "fingerprint": log.fingerprint(),
    }


def split_stats(dataset: str, splits: SplitSet) -> dict:
    full = splits.full
    lengths = full.lengths()
    owner = np.repeat(np.arange(full.n_users), lengths)
    same_user = owner[1:] == owner[:-1]
    same_ts = (full.ts[1:] == full.ts[:-1])[same_user]
    row = {
        "dataset": dataset,
        "split": splits.kind,
        "users": full.n_users,
        "items": full.n_items,
        "interactions": int((full.items > 0).sum()),
        "len_p10": int(np.percentile(lengths, 10)),
        "len_p50": int(np.percentile(lengths, 50)),
        "len_p90": int(np.percentile(lengths, 90)),
        "len_max": int(lengths.max()),
        "density": round(float((full.items > 0).sum()) / (full.n_users * full.n_items), 6),
        "same_timestamp_share": round(float(same_ts.mean()) if len(same_ts) else 0.0, 4),
        "fingerprint": full.fingerprint(),
    }
    if splits.kind == "gts":
        row |= {k: v for k, v in splits.meta.items()}
    return row


def compare_published(stats_rows: list[dict], published_csv: Path) -> list[dict]:
    ours = {(r["dataset"], q): r[q] for r in stats_rows if r["split"] == "loo" for q in
            ("users", "items", "interactions")}  # fmt: skip
    out = []
    with published_csv.open(encoding="utf-8") as f:
        for ref in csv.DictReader(f):
            key = (ref["dataset"], ref["quantity"])
            if key not in ours:
                continue
            value, published, rounded = (
                float(ours[key]),
                float(ref["value"]),
                float(ref["rounded_to"]),
            )
            rel = (value - published) / published
            within_rounding = rounded > 0 and abs(value - published) <= rounded / 2
            out.append(
                {
                    "dataset": ref["dataset"],
                    "variant": ref["variant"],
                    "quantity": ref["quantity"],
                    "ours": int(value),
                    "published": ref["value"],
                    "relative_difference": round(rel, 4),
                    "needs_explanation": abs(rel) > 0.01 and not within_rounding,
                    "source": ref["source"],
                }
            )
    return out


def merge_rows(path: Path, rows: list[dict], datasets: set[str]) -> list[dict]:
    """Replace the rows of `datasets` in an existing CSV and keep the others."""
    kept = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if r["dataset"] not in datasets]
    return kept + rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write aggregate rows; refuse anything that looks like row-level data first."""
    columns: list[str] = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    _check_aggregate(path.name, columns, len(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _check_aggregate(name: str, columns: list[str], n_rows: int) -> None:
    bad = FORBIDDEN_COLUMNS.intersection(c.lower() for c in columns)
    if bad or n_rows > MAX_RESULT_ROWS:
        raise ValueError(f"{name} looks like row-level data: columns {sorted(bad)}, rows {n_rows}")


def results_are_aggregate(path: Path) -> None:
    """Check a committed CSV the same way `write_csv` checks before writing."""
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        n_rows = sum(1 for _ in reader)
    _check_aggregate(path.name, header, n_rows)
