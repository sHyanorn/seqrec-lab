"""Command line: `seqrec data prepare`, `seqrec run|test|tune PLAN`, `seqrec seeds CSV`, and
`seqrec export`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from seqrec.data.registry import DATASETS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="seqrec")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="download and preprocess datasets")
    data_sub = data.add_subparsers(dest="action", required=True)
    prep = data_sub.add_parser("prepare")
    prep.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    prep.add_argument("--data-dir", type=Path, default=Path("data"))
    prep.add_argument("--results-dir", type=Path, default=Path("results/data"))

    for name, text in (
        ("run", "train every pending run of a plan"),
        ("test", "read test data once for each trained final or anchor run of a plan"),
        ("tune", "run the Optuna studies of a tuning plan"),
    ):
        cmd = sub.add_parser(name, help=text)
        cmd.add_argument("plan", type=Path)
        cmd.add_argument("--runs-dir", type=Path, default=Path("runs"))
        cmd.add_argument("--data-dir", type=Path, default=Path("data"))
        cmd.add_argument("--device", default=None)

    seeds = sub.add_parser("seeds", help="seed count from pilot validation scores (METHODS 7.5)")
    seeds.add_argument(
        "csv", type=Path, help="an exported batch CSV, e.g. results/runs/m2-pilot.csv"
    )
    seeds.add_argument("--tag", default="sigma")

    exp = sub.add_parser("export", help="aggregate finished runs into results/runs")
    exp.add_argument("--runs-dir", type=Path, default=Path("runs"))
    exp.add_argument("--out-dir", type=Path, default=Path("results/runs"))

    args = parser.parse_args(argv)
    if args.command == "data":
        from seqrec.data.prepare import prepare

        prepare(args.datasets, args.data_dir, args.results_dir)
        return 0
    if args.command in ("run", "test", "tune"):
        from seqrec import models  # noqa: F401  registers the arms
        from seqrec.runner.batch import run_batch, test_batch, tune_batch

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        step = {"run": run_batch, "test": test_batch, "tune": tune_batch}[args.command]
        failed = step(args.plan, args.runs_dir, args.data_dir, device)
        return 1 if failed else 0
    if args.command == "seeds":
        from seqrec.analysis.power import report

        print(report(args.csv, args.tag))
        return 0
    from seqrec.runner.export import export

    problems = export(args.runs_dir, args.out_dir)
    for p in problems:
        print(f"[export] {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
