"""GPU-hour projection from the M2 pilot (METHODS 12, D012, D034).

Per dataset and arm, one run costs epochs x epoch time plus one evaluation of the restored
checkpoint. Epoch time is the median training plus validation time per epoch of the pilot's
timing runs (`time-small`, the anchor size; `time-large`, the largest size of the search
space), plus their checkpoint time per epoch. Epochs per run are the best epoch of the
MovieLens-1M seed-variance runs, scaled to other datasets by the best-epoch ratios in [5]
Table 2, plus the patience, at most the cap. The core counts 5 leave-one-out seeds; seeds
beyond 5 (METHODS 7.5) come from the reserve in the priority order of METHODS 7.5. Pruning
savings, floor baselines and test reads are left out.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from seqrec.analysis.power import NEURAL_ARMS, seed_count, sigma_from_export
from seqrec.analysis.table import format_table
from seqrec.runner.config import default_limits

DATASETS = ("ml-1m", "video-games", "steam", "ml-20m")  # also the seed priority (METHODS 7.5)
TRIALS = {"ml-1m": 20, "video-games": 20, "steam": 20, "ml-20m": 10}  # METHODS 5.4
CORE_SEEDS = 5  # METHODS 12: final runs with 5 seeds on both splits
SHUFFLED = {"sasrec-ce": DATASETS, "bert4rec": ("ml-1m", "ml-20m")}  # METHODS 3.3
RQ3_HOURS = 15.0  # METHODS 12; measured in M4
CAP_HOURS = 200.0  # D012
WARMUP_S = 180.0  # per batch (METHODS 5.5)
PILOT_BATCH_PREFIX = "m2-"  # hours already spent: the M2 batches

# Best epoch on each dataset in [5] Table 2: SASRec+ (cross-entropy), SASRec (BCE), BERT4Rec.
REF_BEST_EPOCH = {
    "ml-1m": {"ce": 63, "bce": 53, "bert4rec": 197},
    "ml-20m": {"ce": 46, "bce": 30, "bert4rec": 68},
    "steam": {"ce": 16, "bce": 39, "bert4rec": 74},
}
FAMILY = {
    "sasrec-ce": "ce", "sasrec-ss": "ce", "sasrec-bce": "bce", "gsasrec": "bce",
    "bert4rec": "bert4rec",
}  # fmt: skip


def epoch_ratio(dataset: str, arm: str) -> float:
    """Best epoch on `dataset` relative to MovieLens-1M in [5]; 1 where [5] has no value."""
    ref = REF_BEST_EPOCH.get(dataset)
    family = FAMILY[arm]
    return ref[family] / REF_BEST_EPOCH["ml-1m"][family] if ref else 1.0


@dataclass(frozen=True)
class Pilot:
    epoch_s: dict  # (dataset, arm, "small" | "large") -> (train + checkpoint s, eval s)
    best_epoch: dict  # arm -> mean 1-based best epoch of the MovieLens-1M seed-variance runs
    spent_hours: float  # GPU hours of the M2 batches


def load_pilot(pilot_csv: Path, overview_csv: Path, tag: str = "sigma") -> Pilot:
    with pilot_csv.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "loo" and r["shuffle"] == "none"]
    epoch_s, best = {}, defaultdict(list)
    for r in rows:
        if r["status"] not in ("trained", "tested"):
            continue
        if r["tag"] in ("time-small", "time-large"):
            key = (r["dataset"], r["arm"], r["tag"].removeprefix("time-"))
            per_epoch_checkpoint = float(r["checkpoint_s"]) / int(r["epochs"])
            train = float(r["epoch_train_s_median"]) + per_epoch_checkpoint
            epoch_s[key] = (train, float(r["epoch_eval_s_median"]))
        elif r["tag"] == tag and r["dataset"] == "ml-1m" and r["arm"] in NEURAL_ARMS:
            best[r["arm"]].append(int(r["best_epoch"]) + 1)
    missing = [
        f"{d}/{a}/{s}" for d in DATASETS for a in NEURAL_ARMS for s in ("small", "large")
        if (d, a, s) not in epoch_s
    ] + [f"{tag}/{a}" for a in NEURAL_ARMS if a not in best]  # fmt: skip
    if missing:
        raise ValueError(f"pilot export lacks: {', '.join(missing)}")
    with overview_csv.open(encoding="utf-8") as f:
        spent = sum(
            float(r["gpu_hours"] or 0)
            for r in csv.DictReader(f)
            if r["batch"].startswith(PILOT_BATCH_PREFIX)
        )
    return Pilot(epoch_s, {a: float(np.mean(v)) for a, v in best.items()}, spent)


@dataclass(frozen=True)
class Scope:
    """What remains to run; the cuts of D012 shrink it."""

    trials: dict = field(default_factory=lambda: dict(TRIALS))
    datasets: tuple = DATASETS
    temporal: tuple = DATASETS


@dataclass(frozen=True)
class Scenario:
    name: str
    scale_epochs: bool  # scale MovieLens-1M epochs by the [5] ratios
    time: str  # "mid" (mean of the small and large sizes) or "large"


SCENARIOS = (
    Scenario("central", True, "mid"),
    Scenario("ml-1m epochs", False, "mid"),
    Scenario("upper", False, "large"),
)


def run_hours(pilot: Pilot, dataset: str, arm: str, scenario: Scenario) -> tuple[float, float]:
    """Epochs per run and GPU hours per run."""
    sizes = ("large",) if scenario.time == "large" else ("small", "large")
    train_s, eval_s = np.mean([pilot.epoch_s[(dataset, arm, s)] for s in sizes], axis=0)
    cap, patience = default_limits(arm)
    ratio = epoch_ratio(dataset, arm) if scenario.scale_epochs else 1.0
    epochs = min(cap, pilot.best_epoch[arm] * ratio + patience)
    return epochs, (epochs * (train_s + eval_s) + eval_s) / 3600


def _hours(pilot: Pilot, scope: Scope, scenario: Scenario) -> dict:
    return {
        (d, a): run_hours(pilot, d, a, scenario)[1] for d in scope.datasets for a in NEURAL_ARMS
    }


def _shuffled(scope: Scope) -> list[tuple[str, str]]:
    return [(d, a) for a, ds in SHUFFLED.items() for d in ds if d in scope.datasets]


def project(pilot: Pilot, scope: Scope, scenario: Scenario) -> dict:
    hours = _hours(pilot, scope, scenario)
    batches = len(scope.datasets) * 2 + len(scope.temporal) + 1  # tuning, two splits, control
    blocks = {
        "spent": pilot.spent_hours,
        "tuning": sum(scope.trials[d] * h for (d, _), h in hours.items()),
        "final_loo": CORE_SEEDS * sum(hours.values()),
        "final_temporal": CORE_SEEDS * sum(h for (d, _), h in hours.items() if d in scope.temporal),
        "shuffle_control": CORE_SEEDS * sum(hours[k] for k in _shuffled(scope)),
        "warmup": batches * WARMUP_S / 3600,
        "rq3": RQ3_HOURS,
    }  # fmt: skip
    blocks["core"] = sum(blocks.values())
    blocks["reserve"] = CAP_HOURS - blocks["core"]
    return blocks


def extra_seed_hours(pilot: Pilot, scope: Scope, scenario: Scenario) -> dict[str, float]:
    """Cost of one more leave-one-out seed for every arm of a dataset, with its shuffle-control
    runs (METHODS 3.3 uses the same seeds)."""
    hours = _hours(pilot, scope, scenario)
    shuffled = _shuffled(scope)
    return {
        d: sum(hours[(d, a)] for a in NEURAL_ARMS)
        + sum(hours[(d, a)] for (sd, a) in shuffled if sd == d)
        for d in scope.datasets
    }


def allocate_seeds(reserve: float, costs: dict[str, float], n: int) -> dict[str, int]:
    """Leave-one-out seeds per dataset: 5, plus up to n - 5 extra seeds paid from the reserve
    in the priority order of METHODS 7.5. Once a dataset cannot get all of them, the datasets
    after it get none."""
    seeds, extra = {}, max(0, n - CORE_SEEDS)
    for d in (d for d in DATASETS if d in costs):
        affordable = math.floor(max(reserve, 0.0) / costs[d]) if costs[d] > 0 else extra
        seeds[d] = CORE_SEEDS + min(extra, affordable)
        reserve -= (seeds[d] - CORE_SEEDS) * costs[d]
        if affordable < extra:
            reserve = 0.0
    return seeds


CUTS = (
    ("halve MovieLens-20M tuning trials",
     lambda s: replace(s, trials=s.trials | {"ml-20m": s.trials["ml-20m"] // 2})),
    ("drop the MovieLens-20M temporal split",
     lambda s: replace(s, temporal=tuple(d for d in s.temporal if d != "ml-20m"))),
    ("drop Steam",
     lambda s: replace(s, datasets=tuple(d for d in s.datasets if d != "steam"),
                       temporal=tuple(d for d in s.temporal if d != "steam"))),
)  # fmt: skip


def cuts(pilot: Pilot, scope: Scope, scenario: Scenario = SCENARIOS[0]) -> tuple[Scope, list]:
    """The D012 cuts, applied in order while the projected core exceeds the cap."""
    steps = []
    for name, cut in CUTS:
        if project(pilot, scope, scenario)["core"] <= CAP_HOURS:
            break
        scope = cut(scope)
        steps.append((name, project(pilot, scope, scenario)["core"]))
    return scope, steps


def seeds_from_pilot(pilot_csv: Path, tag: str = "sigma") -> int:
    return seed_count(sigma_from_export(pilot_csv, tag, "ml-1m")[0])


def report(pilot_csv: Path, overview_csv: Path, seeds: int | None = None) -> str:
    pilot = load_pilot(pilot_csv, overview_csv)
    n = seeds or seeds_from_pilot(pilot_csv)
    central, ml1m, upper = SCENARIOS
    rows = []
    for d in DATASETS:
        for a in NEURAL_ARMS:
            small, large = (sum(pilot.epoch_s[(d, a, s)]) for s in ("small", "large"))
            epochs, hours = run_hours(pilot, d, a, central)
            rows.append(
                {
                    "dataset": d, "arm": a,
                    "epoch_s_small": f"{small:.2f}", "epoch_s_large": f"{large:.2f}",
                    "epochs": f"{epochs:.0f}", "run_h": f"{hours:.3f}",
                    "epochs_ml1m": f"{run_hours(pilot, d, a, ml1m)[0]:.0f}",
                    "run_h_upper": f"{run_hours(pilot, d, a, upper)[1]:.3f}",
                }
            )  # fmt: skip
    projections = {s.name: project(pilot, Scope(), s) for s in SCENARIOS}
    blocks = [
        {"block": b} | {name: f"{p[b]:.1f}" for name, p in projections.items()}
        for b in projections["central"]
    ]
    scope, steps = cuts(pilot, Scope())
    reserve = project(pilot, scope, central)["reserve"]
    costs = extra_seed_hours(pilot, scope, central)
    seeds_per = allocate_seeds(reserve, costs, n)
    extra = [
        {"dataset": d, "extra_seed_h": f"{costs[d]:.1f}", "seeds": seeds_per[d]} for d in costs
    ]
    left = reserve - sum((seeds_per[d] - CORE_SEEDS) * costs[d] for d in costs)
    return "\n\n".join(
        [
            f"seeds from the METHODS 7.5 rule: {n}; the core counts {CORE_SEEDS}",
            format_table(rows, list(rows[0])),
            format_table(blocks, ["block", *projections]),
            "\n".join(f"cut: {name}; central core {core:.1f} h" for name, core in steps)
            or f"central core within the {CAP_HOURS:.0f} h cap; no cut",
            format_table(extra, ["dataset", "extra_seed_h", "seeds"]),
            f"central reserve {reserve:.1f} h; left after extra seeds (RQ4 next) {left:.1f} h",
        ]
    )
