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

PHASES = ("tune", "final", "anchor")
SHUFFLES = ("none", "train_eval")


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
    max_epochs: int = 300
    patience: int = 30
    guard_hours: float = 8.0
    max_len: int | None = None  # defaults to the dataset's length
    tag: str = ""  # distinguishes runs that share arm and seed, e.g. tuning trials

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        if self.shuffle not in SHUFFLES:
            raise ValueError(f"shuffle must be one of {SHUFFLES}")

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


def load_plan(path: Path) -> Plan:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    batch = raw["batch"]
    defaults = raw.get("defaults", {})
    runs = []
    for entry in raw["runs"]:
        entry = defaults | entry
        seeds = entry.pop("seeds", [entry.pop("seed", 0)])
        runs += [RunConfig(batch=batch, seed=s, **entry) for s in seeds]
    ids = [r.run_id for r in runs]
    if len(set(ids)) != len(ids):
        raise ValueError("plan contains duplicate run ids")
    if raw.get("order", "listed") == "shuffled":
        random.Random(raw.get("order_seed", 0)).shuffle(runs)
    return Plan(batch, float(raw.get("warmup_s", 0)), runs)
