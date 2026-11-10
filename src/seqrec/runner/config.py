"""Run configuration and batch plans.

A plan is a TOML file:

    batch = "pilot"
    warmup_s = 180          # continuous GPU load before the first pending run
    order = "shuffled"      # or "listed"; shuffled interleaves arms and seeds
    order_seed = 0

    [defaults]
    dataset = "ml-1m"
    split = "loo"
    phase = "final"

    [[runs]]
    arm = "sasrec-ce"
    seeds = [1, 2, 3]
    params = { lr = 0.001 }
"""

from __future__ import annotations

import json
import random
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from seqrec.data.registry import DATASETS

PHASES = ("tune", "final", "anchor")
SHUFFLES = ("none", "train_eval")
SPLITS = ("loo", "gts")
# Epoch cap and patience per arm family (METHODS 5.2); a plan may override them.
LIMITS = {"bert4rec": (1000, 100)}
DEFAULT_LIMITS = (300, 30)


# Epochs a run may take to beat 1.5 times MostPop's validation NDCG@10 before patience
# counts (D035); the floor baselines do not train, so the rule is off for them.
START_LIMIT = 300
FLOOR_ARMS = ("mostpop", "transition")


def default_limits(arm: str) -> tuple[int, int]:
    return next((v for prefix, v in LIMITS.items() if arm.startswith(prefix)), DEFAULT_LIMITS)


def default_start_limit(arm: str) -> int:
    return 0 if arm in FLOOR_ARMS else START_LIMIT


@dataclass(frozen=True)
class RunConfig:
    batch: str
    dataset: str
    split: str
    arm: str
    seed: int
    phase: str
    shuffle: str = "none"
    params: dict = field(default_factory=dict)
    max_epochs: int | None = None  # None: the arm's default (METHODS 5.2)
    patience: int | None = None
    start_limit: int | None = None  # None: the arm's default (D035); 0 turns the rule off
    guard_hours: float = 8.0
    max_len: int | None = None  # defaults to the dataset's length
    tag: str = ""  # distinguishes runs that share arm and seed, e.g. tuning trials

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        if self.shuffle not in SHUFFLES:
            raise ValueError(f"shuffle must be one of {SHUFFLES}")
        if self.dataset not in DATASETS or self.split not in SPLITS:
            raise ValueError(f"unknown dataset {self.dataset!r} or split {self.split!r}")
        epochs, patience = default_limits(self.arm)
        if self.max_epochs is None:
            object.__setattr__(self, "max_epochs", epochs)
        if self.patience is None:
            object.__setattr__(self, "patience", patience)
        if self.start_limit is None:
            object.__setattr__(self, "start_limit", default_start_limit(self.arm))
        if self.start_limit < 0:
            raise ValueError(f"start_limit must be 0 or more, got {self.start_limit}")

    @property
    def run_id(self) -> str:
        parts = [self.arm, self.dataset, self.split, f"s{self.seed}"]
        if self.shuffle != "none":
            parts.append(self.shuffle)
        if self.tag:
            parts.append(self.tag)
        return "-".join(parts)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)


@dataclass(frozen=True)
class Plan:
    batch: str
    warmup_s: float
    runs: list[RunConfig]


def read_plan(path: Path) -> dict:
    """Parse a plan file. It holds runs or studies; each command would silently drop the other."""
    raw = tomllib.loads(path.read_text(encoding="utf-8-sig"))  # tolerate a BOM
    if "runs" in raw and "studies" in raw:
        raise ValueError(f"{path.name} has both runs and studies")
    return raw


def load_plan(path: Path) -> Plan:
    raw = read_plan(path)
    batch = raw["batch"]
    defaults = raw.get("defaults", {})
    runs = []
    for entry in raw["runs"]:
        params = defaults.get("params", {}) | entry.get("params", {})
        entry = defaults | entry | {"params": params}
        seeds = entry.pop("seeds", [entry.pop("seed", 0)])
        runs += [RunConfig(batch=batch, seed=s, **entry) for s in seeds]
    ids = [r.run_id for r in runs]
    if len(set(ids)) != len(ids):
        raise ValueError("plan contains duplicate run ids")
    if raw.get("order", "listed") == "shuffled":
        random.Random(raw.get("order_seed", 0)).shuffle(runs)
    return Plan(batch, float(raw.get("warmup_s", 0)), runs)
