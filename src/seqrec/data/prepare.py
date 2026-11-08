"""`seqrec data prepare`: download, preprocess, check and summarise each dataset."""

from __future__ import annotations

import time
from pathlib import Path

from seqrec.data.download import download
from seqrec.data.preprocess import build_log
from seqrec.data.readers import READERS
from seqrec.data.registry import DATASETS
from seqrec.data.stats import compare_published, log_stats, merge_rows, split_stats, write_csv
from seqrec.data.store import build_splits, log_path

PUBLISHED = Path(__file__).resolve().parents[3] / "configs" / "published_stats.csv"


def prepare(datasets: list[str], data_dir: Path, results_dir: Path, log=print) -> None:
    sources, stats = [], []
    for name in datasets:
        spec = DATASETS[name]
        t0 = time.perf_counter()
        meta = download(spec, data_dir / "raw", log)
        sources.append({k: meta[k] for k in ("dataset", "url", "file", "bytes", "sha256")})
        log(f"[data] reading {name}")
        df, read_info = READERS[name](data_dir / "raw" / name / spec.filename)
        events = build_log(df)
        del df
        events.save(log_path(data_dir, name))
        stats.append(log_stats(name, events, read_info))
        for kind in ("loo", "gts"):
            splits, checks = build_splits(events, kind)
            stats.append(split_stats(name, splits) | {"invariants_passed": len(checks)})
        log(f"[data] {name} done in {time.perf_counter() - t0:.0f} s")

    done = set(datasets)
    write_csv(results_dir / "sources.csv", merge_rows(results_dir / "sources.csv", sources, done))
    all_stats = merge_rows(results_dir / "stats.csv", stats, done)
    write_csv(results_dir / "stats.csv", all_stats)
    write_csv(results_dir / "published_comparison.csv", compare_published(all_stats, PUBLISHED))
