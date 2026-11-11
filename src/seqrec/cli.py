"""Command line: `seqrec data prepare`, `seqrec run|test|tune PLAN`, `seqrec seeds CSV`,
`seqrec anchors CSV`, `seqrec budget CSV`, `seqrec final-plan CSV`, `seqrec analyze CSV`,
`seqrec bootstrap DIR` and `seqrec export`."""

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
        if name == "test":
            _earlier_exports(cmd)
            cmd.add_argument(
                "--committed",
                type=Path,
                default=Path("results/runs"),
                help="batch files exported from this machine; a test read recorded there and "
                "not made in the runs directory stops the task",
            )

    seeds = sub.add_parser("seeds", help="seed count from pilot validation scores (METHODS 7.5)")
    seeds.add_argument(
        "csv",
        type=Path,
        nargs="+",
        help="exported batch CSVs, e.g. results/runs/m2-pilot.csv; runs in a later file replace "
        "earlier runs with the same tag, dataset, split, shuffle and arm",
    )
    seeds.add_argument("--tag", default="sigma")
    seeds.add_argument("--dataset", default="ml-1m")
    seeds.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="use the runs with these seeds only and require them of every arm: 1 2 3 4 5 for "
        "the estimate after the first 5 final seeds (METHODS 7.5; final runs have --tag '')",
    )

    anchors = sub.add_parser("anchors", help="anchor runs against published bands (D032)")
    anchors.add_argument(
        "csv", type=Path, help="the anchor export, e.g. results/runs/m2-anchor.csv"
    )
    anchors.add_argument(
        "--pilot",
        type=Path,
        nargs="+",
        help="pilot exports, to compare limits (D033); as for seeds",
    )
    anchors.add_argument("--plan", type=Path, default=Path("configs/plans/m2-anchor.toml"))

    budget = sub.add_parser("budget", help="GPU-hour projection from the pilot (D034)")
    budget.add_argument(
        "csv",
        type=Path,
        nargs="+",
        help="pilot exports, e.g. results/runs/m2-pilot.csv; as for seeds",
    )
    budget.add_argument("--overview", type=Path, help="default: overview.csv next to the first")
    budget.add_argument("--seeds", type=int, help="default: the METHODS 7.5 rule on the pilot")

    analyze = sub.add_parser("analyze", help="the registered tests on final runs (METHODS 7, 8)")
    analyze.add_argument(
        "csv", type=Path, nargs="+", help="exported batches of final runs, after their test reads"
    )
    analyze.add_argument("--out-dir", type=Path, default=Path("results/analysis"))

    boot = sub.add_parser(
        "bootstrap", help="two-level bootstrap of F1 and F2 from kept test ranks (METHODS 7.6)"
    )
    boot.add_argument("batch_dir", type=Path, help="a tested batch's runs, e.g. runs/<batch>")
    boot.add_argument("--out", type=Path, required=True, help="the table of intervals")

    final = sub.add_parser("final-plan", help="write a dataset's final-run plan (METHODS 5.5)")
    final.add_argument("csv", type=Path, help="tuning_best.csv, e.g. results/runs/tuning_best.csv")
    final.add_argument("--tuning-batch", required=True, help="the batch of the dataset's studies")
    final.add_argument("--dataset", required=True, choices=list(DATASETS))
    final.add_argument("--seeds", type=int, required=True, help="leave-one-out seeds, 5 to 10")
    final.add_argument("--out", type=Path, required=True, help="the plan file, named <batch>.toml")

    exp = sub.add_parser("export", help="aggregate finished runs into results/runs")
    exp.add_argument("--runs-dir", type=Path, default=Path("runs"))
    exp.add_argument("--out-dir", type=Path, default=Path("results/runs"))
    exp.add_argument(
        "--summaries-only",
        action="store_true",
        help="read no run directories; rewrite overview.csv and tuning_best.csv from the "
        "batch files in the output directory",
    )
    exp.add_argument(
        "--plan",
        type=Path,
        help="fail, writing nothing, unless every run of this plan is trained or given up "
        "(for a tuning plan: every study has reached its trial budget)",
    )
    exp.add_argument(
        "--check-reads",
        action="store_true",
        help="write nothing; fail if a batch file in the output directory records a test "
        "read that was not made in the runs directory",
    )
    exp.add_argument(
        "--written", type=Path, help="list the batch files written in this file, one per line"
    )
    _earlier_exports(exp)

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
        more = {}
        if args.command == "test":
            more = {"earlier": _earlier(args), "committed": args.committed}
        failed = step(args.plan, args.runs_dir, args.data_dir, device, **more)
        return 1 if failed else 0
    if args.command in ("seeds", "anchors", "budget"):
        from seqrec.analysis.power import IncompleteRerun

        try:
            print(_analysis(args))
        except IncompleteRerun as e:
            print(f"[{args.command}] {e}", file=sys.stderr)
            return 1
        return 0
    if args.command == "analyze":
        from seqrec.analysis.confirm import report as confirm_report

        print(confirm_report(args.csv, args.out_dir))
        return 0
    if args.command == "bootstrap":
        from seqrec.analysis.bootstrap import report as bootstrap_report

        print(bootstrap_report(args.batch_dir, args.out))
        return 0
    if args.command == "final-plan":
        import csv

        from seqrec.runner.final_plan import final_plan

        with args.csv.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        text = final_plan(rows, args.dataset, args.tuning_batch, args.seeds, args.out.stem)
        args.out.write_text(text, encoding="utf-8")
        return 0
    from seqrec.runner.export import export, summarize, unbacked_reads

    if args.check_reads:
        problems = unbacked_reads(args.out_dir, args.runs_dir)
        for p in problems:
            print(f"[export] {p}", file=sys.stderr)
        if problems:
            print(
                f"[export] {args.out_dir} records test reads made on another volume; move its "
                "batch files to results/<name>/ (D045)",
                file=sys.stderr,
            )
        return 1 if problems else 0
    if args.plan:
        from seqrec import models  # noqa: F401  registers the arms
        from seqrec.runner.batch import unfinished

        left = unfinished(args.plan, args.runs_dir)
        if left:
            print(f"[export] {args.plan.name}: {len(left)} not finished, e.g. {left[0]}",
                  file=sys.stderr)  # fmt: skip
            return 1
    if args.summaries_only:
        problems = summarize(args.out_dir, _earlier(args))
    else:
        problems = export(args.runs_dir, args.out_dir, _earlier(args), args.written)
    for p in problems:
        print(f"[export] {p}", file=sys.stderr)
    return 1 if problems else 0


def _earlier_exports(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument(
        "--earlier-exports",
        type=Path,
        nargs="*",
        help="directories of batch files exported from another machine, whose anchor reads "
        "count toward the limit of METHODS 11.5 and whose runs stay out of the summaries "
        "(default: results/laptop and every other directory under results/ but runs, data "
        "and env; give no directory for none)",
    )


def _earlier(args) -> list[Path]:
    from seqrec.runner.export import earlier_exports

    return earlier_exports() if args.earlier_exports is None else args.earlier_exports


def _analysis(args) -> str:
    if args.command == "seeds":
        from seqrec.analysis.power import report

        return report(args.csv, args.tag, args.dataset, args.seeds)
    if args.command == "anchors":
        from seqrec.analysis.anchors import report as anchor_report

        return anchor_report(args.csv, args.plan, args.pilot)
    from seqrec.analysis.budget import report as budget_report

    overview = args.overview or args.csv[0].with_name("overview.csv")
    return budget_report(args.csv, overview, args.seeds)


if __name__ == "__main__":
    sys.exit(main())
