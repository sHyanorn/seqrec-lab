"""Hyperparameter tuning (METHODS 5.3, 5.4): one Optuna study per (arm, dataset).

A tuning plan is a TOML file:

    batch = "tune-ml-1m"
    warmup_s = 180
    [defaults]
    dataset = "ml-1m"
    trials = 20
    [[studies]]
    arm = "sasrec-ce"

Each trial is an ordinary run with phase "tune", seed 0 and tag tNN, so the test split is
sealed and the run directory records every parameter. Studies are stored in
runs/<batch>/optuna.sqlite3. After an interruption a running trial resumes from its
checkpoint with its own parameters. TPE is reseeded from (sampler seed, trials so far)
before every proposal, so an interrupted study proposes what an uninterrupted one would.
Failed and pruned trials count toward the trial budget.
"""

from __future__ import annotations

import tomllib
import traceback
from dataclasses import dataclass
from pathlib import Path

import optuna
from optuna.distributions import CategoricalDistribution as Choice
from optuna.distributions import FloatDistribution as Float
from optuna.trial import TrialState

from seqrec.runner.config import RunConfig, default_limits
from seqrec.runner.run import ARMS, execute, run_dir_of

SHARED_SPACE = {
    "lr": Float(1e-4, 1e-2, log=True),
    "dim": Choice([64, 128, 256]),
    "blocks": Choice([1, 2, 3]),
    "heads": Choice([1, 2]),
    "dropout": Float(0.1, 0.6),
    "weight_decay": Choice([0.0, 1e-4, 1e-2]),
}
OUTPUT = {"shared_output": Choice([True, False])}
SPACES = {
    "sasrec-bce": SHARED_SPACE | OUTPUT,
    "sasrec-ce": SHARED_SPACE | OUTPUT,
    "sasrec-ss": SHARED_SPACE | OUTPUT | {"negatives": Choice([64, 256, 1024, 4096])},
    "gsasrec": SHARED_SPACE | OUTPUT | {"negatives": Choice([64, 128, 256]), "t": Float(0.5, 1.0)},
    "bert4rec": SHARED_SPACE | {"mask_prob": Float(0.1, 0.4)},
}
STARTUP_TRIALS = 5
TRAINING_SEED = 0

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class Study:
    batch: str
    dataset: str
    arm: str
    trials: int
    sampler_seed: int = 0
    max_epochs: int | None = None  # None: the arm's cap (METHODS 5.2)
    patience: int | None = None
    max_len: int | None = None

    @property
    def name(self) -> str:
        return f"{self.arm}-{self.dataset}"

    def config(self, number: int, params: dict) -> RunConfig:
        return RunConfig(
            batch=self.batch, dataset=self.dataset, split="loo", arm=self.arm,
            seed=TRAINING_SEED, phase="tune", params=params, max_epochs=self.max_epochs,
            patience=self.patience, max_len=self.max_len, tag=f"t{number:02d}",
        )  # fmt: skip


def load_tuning_plan(path: Path) -> tuple[str, float, list[Study]]:
    raw = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    batch, defaults = raw["batch"], raw.get("defaults", {})
    studies = [Study(batch=batch, **(defaults | entry)) for entry in raw["studies"]]
    for s in studies:
        if s.arm not in SPACES:
            raise ValueError(f"no search space for arm {s.arm!r}")
    if len({s.name for s in studies}) != len(studies):
        raise ValueError("plan contains duplicate studies")
    return batch, float(raw.get("warmup_s", 0)), studies


def _storage(runs_dir: Path, batch: str) -> str:
    path = (runs_dir / batch / "optuna.sqlite3").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def _run_trial(study, opt: optuna.Study, trial: optuna.Trial, runs_dir, data_dir, device, log):
    cfg = study.config(trial.number, dict(trial.params))
    trial.set_user_attr("run_id", cfg.run_id)

    def report(epoch: int, value: float) -> bool:
        trial.report(value, epoch)
        return trial.should_prune()

    try:
        summary = execute(cfg, runs_dir, data_dir, device, log=log, report=report)
    except Exception:
        run_dir = run_dir_of(cfg, runs_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        log(f"[tune] {cfg.run_id} failed; see error.txt")
        opt.tell(trial, state=TrialState.FAIL)
        return
    if summary["stop_reason"] == "pruned":
        opt.tell(trial, state=TrialState.PRUNED)
    elif summary["status"] == "trained":
        opt.tell(trial, summary["validation"]["ndcg@10"])
    else:
        opt.tell(trial, state=TrialState.FAIL)


def run_study(study: Study, runs_dir: Path, data_dir: Path, device, log=print) -> optuna.Study:
    if study.arm not in ARMS:
        raise ValueError(f"unknown arm {study.arm!r}")
    cap = study.max_epochs or default_limits(study.arm)[0]
    opt = optuna.create_study(
        study_name=study.name,
        storage=_storage(runs_dir, study.batch),
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(STARTUP_TRIALS, n_warmup_steps=cap // 10),
    )
    for frozen in opt.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        log(f"[tune] {study.name}: resuming trial {frozen.number}")
        trial = optuna.trial.Trial(opt, frozen._trial_id)
        _run_trial(study, opt, trial, runs_dir, data_dir, device, log)
    while (done := len(opt.trials)) < study.trials:
        opt.sampler = optuna.samplers.TPESampler(
            seed=study.sampler_seed * 10_000 + done, n_startup_trials=STARTUP_TRIALS
        )
        trial = opt.ask(SPACES[study.arm])
        log(f"[tune] {study.name}: trial {trial.number} of {study.trials}")
        _run_trial(study, opt, trial, runs_dir, data_dir, device, log)
    return opt
