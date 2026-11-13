"""The confirmatory analysis of final runs (METHODS 7 and 8).

Input: exported batch files with the final runs of one or more datasets, read after their
test reads. A dataset is either absent or complete: a registered comparison on a dataset
that has final runs must find both of its groups.

Every family is corrected over all its registered comparisons (`families.registered`), so
with some datasets still to run a comparison is rejected, not rejected or pending
(`inference.holm`), and so is every conclusion drawn from several comparisons. Pending means
that the datasets still to run decide.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from scipy import stats

from seqrec.analysis import families, inference, timing
from seqrec.analysis.families import DIRECTIONAL, MINIMUM_EFFECT, NON_INFERIORITY, TIME, TOST
from seqrec.analysis.inference import NOT_REJECTED, PENDING, REJECTED
from seqrec.analysis.power import (
    ALPHA,
    DELTA,
    TARGET,
    directional_power,
    equivalence_power,
    pooled_sd,
    sigma_of,
)
from seqrec.analysis.table import format_table
from seqrec.data.stats import write_csv
from seqrec.runner.export import STOPPED
from seqrec.runner.final_plan import H2A_ARMS, H2A_SEEDS

PRIMARY_METRIC = "ndcg@10"
PRIMARY = "primary"
UNFLAGGED = "timing without flagged runs"  # the repeat of METHODS 7.6
# (variant, split, metric, families): the primary analysis; the temporal split (secondary, no
# confirmatory claim, METHODS 7.4); the repeats on HR@10 and without flagged runs (robustness,
# METHODS 7.6).
VARIANTS = (
    (PRIMARY, "loo", PRIMARY_METRIC, ("F1", "F1-NI", "F1-ME", "F5", "F2", "F4", "F6")),
    ("temporal split", "gts", PRIMARY_METRIC, ("F1", "F1-NI", "F1-ME", "F5", "F2")),
    ("HR@10", "loo", "hr@10", ("F1", "F2")),
    (UNFLAGGED, "loo", PRIMARY_METRIC, ("F6",)),
)
CLOCK_SHARE = 0.95  # METHODS 8: a median SM clock more than 5% below the median of all runs
SUPPORTED, NOT_SUPPORTED = "supported", "not supported"
GAP_CLASSES = (
    "reversed below SESOI", "reversed", "confirmed below SESOI", "confirmed", "equivalent",
    "inconclusive",
)  # fmt: skip
METRICS = ("test_ndcg@10", "test_hr@10", "test_ndcg@20", "test_hr@20", "validation_ndcg@10")


def read_runs(paths: Iterable[Path]) -> list[dict]:
    """The rows of exported batch files."""
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as f:
            rows += csv.DictReader(f)
    return rows


def quality(rows: list[dict], split: str, metric: str) -> dict[tuple, list[float]]:
    """y = log of the test metric (METHODS 7.1) of the tested runs, by (dataset, arm,
    shuffled), in seed order."""
    groups = defaultdict(dict)
    for r in rows:
        if r["split"] == split and r["status"] == "tested" and not r["tag"]:
            key = (r["dataset"], r["arm"], r["shuffle"] != "none")
            if int(r["seed"]) in groups[key]:
                raise ValueError(f"{r['run_id']}: a second run with its arm, split and seed")
            groups[key][int(r["seed"])] = math.log(float(r[f"test_{metric}"]))
    return {key: [y for _, y in sorted(by_seed.items())] for key, by_seed in groups.items()}


@dataclass(frozen=True)
class Outcome:
    """One comparison in one variant. `p` is None while its dataset has no final runs."""

    variant: str
    comparison: families.Comparison
    p: float | None = None
    status: str = PENDING
    n_a: int | None = None
    n_b: int | None = None
    estimate: float | None = None  # mean y of a minus b; for time to target, median b / median a
    low: float | None = None  # 95% Welch interval of the difference, descriptive
    high: float | None = None
    power: float | None = None
    median_a: float | None = None  # time to target only, in seconds
    median_b: float | None = None
    # The same function at the sigma of the two arms compared; outside every decision (D055).
    supplementary_power: float | None = None

    @property
    def underpowered(self) -> bool | None:
        return None if self.power is None else self.power < TARGET


def _power(kind: str, sigma: float | None, n: int) -> float | None:
    """The power functions of METHODS 7.5 at a sigma and seed count. The one-sided tests have
    no registered function."""
    if sigma is None:
        return None
    if kind == DIRECTIONAL:
        return directional_power(sigma, n)
    return equivalence_power(sigma, n) if kind == TOST else None


def _quality_outcome(
    variant: str, c: families.Comparison, a, b, sigma: float, pair_sigma: float | None
) -> Outcome:
    w = inference.welch(a, b)
    half = stats.t.ppf(1 - ALPHA / 2, w.df) * w.se
    p = {
        DIRECTIONAL: lambda: inference.p_two_sided(a, b),
        NON_INFERIORITY: lambda: inference.p_above(a, b, -DELTA),
        MINIMUM_EFFECT: lambda: inference.p_above(a, b, DELTA),
        TOST: lambda: inference.p_tost(a, b),
    }[c.kind]()
    n = min(len(a), len(b))
    return Outcome(
        variant, c, p, n_a=len(a), n_b=len(b), estimate=w.diff, low=w.diff - half,
        high=w.diff + half, power=_power(c.kind, sigma, n),
        supplementary_power=_power(c.kind, pair_sigma, n),
    )  # fmt: skip


def flagged(row: dict, clock: float | None) -> bool:
    """A run left out of the repeat of the timing analysis (METHODS 7.6): a hardware slowdown
    event, a median SM clock more than 5% below `clock`, the median over all runs (METHODS
    8), or epochs far slower than its other seeds' (D038)."""
    slow_clock = bool(clock and row["sm_mhz_median"]) and (
        float(row["sm_mhz_median"]) < CLOCK_SHARE * clock
    )
    return row["hardware_slowdown"] == "True" or row["slow_epochs"] == "True" or slow_clock


def time_group(
    rows: list[dict], dataset: str, clock: float | None = None
) -> tuple[float, dict[str, list[float]], dict[str, list[bool]]]:
    """The target quality of a dataset and, by arm and in seed order, the times to target of
    its registered group and which of its runs are flagged (METHODS 8, 9.4). The group is the
    unshuffled leave-one-out runs of A1 to A4 with seeds 1 to 5, all trained on one Pod. The
    target is 0.95 times the highest arm mean of their best validation NDCG@10. A run's
    `validation_records` hold its cumulative training time and validation NDCG@10 at every
    epoch that set a new best, which is where any target is first reached."""
    members = {}
    for r in rows:
        if (
            r["dataset"] == dataset and r["split"] == "loo" and r["shuffle"] == "none"
            and not r["tag"] and r["arm"] in H2A_ARMS and int(r["seed"]) in H2A_SEEDS
        ):  # fmt: skip
            if (r["arm"], int(r["seed"])) in members:
                raise ValueError(f"{r['run_id']}: a second run with its arm and seed in the group")
            members[r["arm"], int(r["seed"])] = r
    missing = [(a, s) for a in H2A_ARMS for s in H2A_SEEDS if (a, s) not in members]
    if missing:
        raise ValueError(f"{dataset}: the time-to-target group lacks {missing[0]}")
    pods = {r["pod"] for r in members.values()}
    if len(pods) != 1 or not next(iter(pods)):
        raise ValueError(
            f"{dataset}: the time-to-target group ran on Pods {sorted(pods)}; times from "
            "different Pods are not compared (METHODS 9.4)"
        )
    curves = {  # per run: (cumulative training seconds, validation NDCG@10)
        key: tuple(zip(*json.loads(r["validation_records"]), strict=True))
        for key, r in members.items()
    }
    best = {a: [max(curves[a, s][1]) for s in H2A_SEEDS] for a in H2A_ARMS}
    target = timing.target_quality(best)
    times = {a: [timing.time_to_target(*curves[a, s], target) for s in H2A_SEEDS] for a in H2A_ARMS}
    flags = {a: [flagged(members[a, s], clock) for s in H2A_SEEDS] for a in H2A_ARMS}
    return target, times, flags


def _time_outcome(variant: str, c: families.Comparison, times: Mapping[str, list]) -> Outcome:
    a, b = times[c.a], times[c.b]
    return Outcome(
        variant, c, inference.mann_whitney_less(a, b), n_a=len(a), n_b=len(b),
        estimate=timing.median_ratio(b, a), median_a=statistics.median(a),
        median_b=statistics.median(b),
    )  # fmt: skip


def analyze(
    rows: list[dict],
    sigma: Mapping[str, float],
    clock: float | None = None,
    validation: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
) -> list[Outcome]:
    """Every registered comparison in every variant, with its Holm status. `rows` are final
    runs; `sigma` holds, per dataset among them, the estimate of METHODS 7.5 from its first
    five seeds; `clock` is the median SM clock that flags a run (`flagged`); `validation`
    holds, per dataset and arm, the values that estimate pooled, and gives each comparison
    with a power function its supplementary power (D055). A run that is trained and not
    tested yet stops the analysis: it comes after the test reads. Runs that failed, or
    diverged without a checkpoint (D031), stay out of their groups."""
    waiting = [r for r in rows if not r["tag"] and r["status"] in ("trained", "unfinished")]
    if waiting:
        raise ValueError(f"{waiting[0]['run_id']} is {waiting[0]['status']}, not tested")
    present = {r["dataset"] for r in rows}
    times = {PRIMARY: {}, UNFLAGGED: {}}
    for d in sorted(present):
        _, every, flags = time_group(rows, d, clock)
        times[PRIMARY][d] = every
        times[UNFLAGGED][d] = {
            arm: [t for t, flag in zip(ts, flags[arm], strict=True) if not flag]
            for arm, ts in every.items()
        }
    out: list[Outcome] = []
    for variant, split, metric, names in VARIANTS:
        groups = quality(rows, split, metric)
        for name in names:
            members = []
            for c in families.family(name):
                if c.dataset not in present:
                    members.append(Outcome(variant, c))
                elif c.kind == TIME:
                    members.append(_time_outcome(variant, c, times[variant][c.dataset]))
                else:
                    try:
                        a = groups[c.dataset, c.a, False]
                        b = groups[c.dataset, c.b, c.shuffled_b]
                    except KeyError as missing:
                        raise ValueError(
                            f"{variant}, {c.name}: no tested run of {missing}"
                        ) from None
                    pair = None
                    if validation is not None and c.kind in (DIRECTIONAL, TOST):
                        pair = pooled_sd([validation[c.dataset][x] for x in (c.a, c.b)])
                    members.append(_quality_outcome(variant, c, a, b, sigma[c.dataset], pair))
            status = inference.holm({o.comparison.name: o.p for o in members})
            out += [replace(o, status=status[o.comparison.name]) for o in members]
    return out


def all_rejected(statuses: Iterable[str]) -> str:
    """A hypothesis that holds when each of its tests rejects (METHODS 7.4, 8)."""
    statuses = list(statuses)
    if NOT_REJECTED in statuses:
        return NOT_SUPPORTED
    return SUPPORTED if all(s == REJECTED for s in statuses) else PENDING


def _first_class(with_sign: bool, directional: bool, tost: bool) -> str:
    """The first class of METHODS 7.3 that applies, given which tests are significant."""
    if directional:
        return GAP_CLASSES[2 * with_sign + (not tost)]
    return GAP_CLASSES[4 + (not tost)]


def gap_class(sign: int, directional: Outcome, tost: Outcome) -> list[str]:
    """The classes of METHODS 7.3 a published gap can still fall into, in their order; one
    class when it is decided. `sign` is the published sign of A - B. A pending test may end
    either way, and a dataset without runs has no sign of its difference yet."""
    either = {REJECTED: (True,), NOT_REJECTED: (False,), PENDING: (True, False)}
    estimate = directional.estimate
    with_sign = (True, False) if estimate is None else ((estimate > 0) == (sign > 0),)
    reachable = {
        _first_class(w, d, t)
        for w in with_sign
        for d in either[directional.status]
        for t in either[tost.status]
    }
    return [name for name in GAP_CLASSES if name in reachable]


def conclusions(outcomes: list[Outcome]) -> list[dict]:
    """Results of the primary analysis per dataset: H1a, H1b, H2a, the three clauses of H2b,
    the shuffle control and the class of each published gap."""
    primary = {
        (o.comparison.family, o.comparison.dataset, o.comparison.a, o.comparison.b): o
        for o in outcomes
        if o.variant == PRIMARY
    }

    def status(family, dataset, a, b):
        return primary[family, dataset, a, b].status

    A1, A2, A3, A4, A5 = families.A1, families.A2, families.A3, families.A4, families.A5
    out = []
    for d in families.DATASETS:
        if d in families.MOVIELENS:
            out.append(("H1a", d, all_rejected([status("F1-NI", d, A2, A5)])))
        out.append(("H1b", d, all_rejected(status("F1-ME", d, a, A1) for a in (A2, A4, A5))))
        out.append(("H2a", d, all_rejected(status("F6", d, A2, b) for b in (A1, A3, A4))))
        out.append(("H2b sampled softmax", d, all_rejected([status("F2", d, A3, A2)])))
        out.append(("H2b gBCE", d, all_rejected([status("F2", d, A4, A2)])))
        out.append(("H2b BCE", d, all_rejected([status("F1-ME", d, A2, A1)])))
        for arm in (A2, A5):
            if ("F4", d, arm, arm) in primary:
                control = all_rejected([status("F4", d, arm, arm)])
                out.append((f"shuffle control {arm}", d, control))
    for gap, ((a, b), signs) in families.GAPS.items():
        for d, sign in signs.items():
            classes = gap_class(sign, primary["F1", d, a, b], primary["F5", d, a, b])
            decided = classes[0] if len(classes) == 1 else f"{PENDING}: {' or '.join(classes)}"
            out.append((gap, d, decided))
    return [{"question": q, "dataset": d, "result": result} for q, d, result in out]


def rank_agreement(rows: list[dict], dataset: str) -> float:
    """Kendall's tau between the arm rankings by mean y on leave-one-out and on the temporal
    split (METHODS 7.4), over the five neural arms."""
    arms = (families.A1, families.A2, families.A3, families.A4, families.A5)
    means = [
        [
            statistics.fmean(quality(rows, split, PRIMARY_METRIC)[dataset, arm, False])
            for arm in arms
        ]
        for split in ("loo", "gts")
    ]
    return float(stats.kendalltau(*means).statistic)


def tests_table(outcomes: list[Outcome]) -> list[dict]:
    """One row per registered comparison and variant. For quality tests `estimate` is the
    difference in mean y, a minus b, and `relative` is exp(estimate) - 1; for time to target
    it is the median of b over the median of a, with its reading under METHODS 9.4."""
    table = []
    for o in outcomes:
        c, timed = o.comparison, o.comparison.kind == TIME
        known = o.estimate is not None
        table.append(
            {
                "variant": o.variant, "family": c.family, "dataset": c.dataset, "test": c.kind,
                "a": c.a, "b": ("C1-" if c.shuffled_b else "") + c.b, "n_a": o.n_a, "n_b": o.n_b,
                "estimate": o.estimate,
                "relative": math.expm1(o.estimate) if known and not timed else None,
                "low": o.low, "high": o.high, "p": o.p, "status": o.status, "power": o.power,
                "underpowered": o.underpowered,
                "reading": timing.reading(o.estimate) if known and timed else None,
                "median_a": o.median_a, "median_b": o.median_b,
                "supplementary_power": o.supplementary_power,
            }
        )  # fmt: skip
    return table


def arms_table(rows: list[dict]) -> list[dict]:
    """Per dataset, split, shuffle and arm: runs, mean and SD of every metric over the tested
    runs, mean tie rate, share of truncated runs among those that stopped by patience, the
    epoch cap, the guard or the start limit (D030, D035, as in overview.csv), and GPU hours
    (METHODS 7.7)."""
    groups = defaultdict(list)
    for r in rows:
        if not r["tag"]:
            groups[r["dataset"], r["split"], r["shuffle"], r["arm"]].append(r)
    table = []
    for (dataset, split, shuffle, arm), runs in sorted(groups.items()):
        tested = [r for r in runs if r["status"] == "tested"]
        row = {
            "dataset": dataset, "split": split, "shuffle": shuffle, "arm": arm,
            "runs": len(runs), "tested": len(tested),
        }  # fmt: skip
        for metric in METRICS:
            values = [float(r[metric]) for r in tested]
            row[f"{metric}_mean"] = statistics.fmean(values) if values else None
            row[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else None
        ties = [float(r["test_tie_rate"]) for r in tested]
        row["test_tie_rate_mean"] = statistics.fmean(ties) if ties else None
        stopped = [r["truncated"] == "True" for r in runs if r["stop_reason"] in STOPPED]
        row["truncated_share"] = statistics.fmean(stopped) if stopped else None
        row["gpu_hours"] = round(sum(float(r["gpu_hours"] or 0) for r in runs), 4)
        table.append(row)
    return table


def times_table(rows: list[dict], clock: float | None = None) -> list[dict]:
    """Per dataset with final runs: the target quality, every run's time to target and
    whether the run is flagged."""
    table = []
    for dataset in sorted({r["dataset"] for r in rows}):
        target, times, flags = time_group(rows, dataset, clock)
        for arm in H2A_ARMS:
            for seed, t, flag in zip(H2A_SEEDS, times[arm], flags[arm], strict=True):
                table.append(
                    {
                        "dataset": dataset, "target_ndcg@10": target, "arm": arm, "seed": seed,
                        "time_to_target_s": t, "flagged": flag,
                    }
                )  # fmt: skip
    return table


def report(paths: Sequence[Path], out_dir: Path) -> str:
    """Analyze the final runs in exported batch files; write tests.csv, conclusions.csv,
    arms.csv and times.csv to `out_dir` and return the conclusions as text. Runs of other
    phases in the files enter only the median SM clock that flags a run."""
    every = read_runs(paths)
    clocks = [float(r["sm_mhz_median"]) for r in every if r["sm_mhz_median"]]
    clock = statistics.median(clocks) if clocks else None
    rows = [r for r in every if r["phase"] == "final"]
    datasets = sorted({r["dataset"] for r in rows})
    estimates = {d: sigma_of(rows, "", d, seeds=H2A_SEEDS) for d in datasets}
    sigma = {d: s for d, (s, _) in estimates.items()}
    outcomes = analyze(rows, sigma, clock, {d: v for d, (_, v) in estimates.items()})
    concluded = conclusions(outcomes)
    for name, table in (
        ("tests", tests_table(outcomes)),
        ("conclusions", concluded),
        ("arms", arms_table(rows)),
        ("times", times_table(rows, clock)),
    ):
        write_csv(out_dir / f"{name}.csv", table)
    lines = [f"sigma of {d} from its first five seeds: {v:.5f}" for d, v in sigma.items()]
    lines += [
        f"{d}: Kendall tau of arm rankings, leave-one-out and temporal split: "
        f"{rank_agreement(rows, d):.2f}"
        for d in datasets
    ]
    for hypothesis in ("H1a", "H1b"):  # METHODS 7.4: the count of datasets supporting each
        results = [c["result"] for c in concluded if c["question"] == hypothesis]
        lines.append(
            f"{hypothesis} supported on {results.count(SUPPORTED)} dataset, not supported on "
            f"{results.count(NOT_SUPPORTED)}, pending on {results.count(PENDING)}"
        )
    lines.append(format_table(concluded, ["question", "dataset", "result"]))
    return "\n".join(lines)
