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
A trial is compared with others by epochs since its start epoch and cannot be pruned before
it starts (D044). Failed and pruned trials count toward the trial budget; a trial
interrupted before all its parameters were stored is marked failed. A trial whose training
raises is retried once before it counts as failed (D031).
"""

from __future__ import annotations

import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path

import optuna
from optuna.distributions import CategoricalDistribution as Choice
from optuna.distributions import FloatDistribution as Float
from optuna.trial import TrialState
from sqlalchemy.engine import URL

from seqrec.runner.attempts import TrainingFailed, train_with_retry
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
MEDIAN_TRIALS = 3  # complete trials with a value at a step before that step can prune (D044)
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


def pruner(cap: int) -> optuna.pruners.BasePruner:
    """Median pruning after 5 complete trials, from step cap // 10 on (METHODS 5.4). A step
    is an epoch counted from the trial's start epoch, and a step prunes only once 3 complete
    trials have a value at it (D044)."""
    return optuna.pruners.MedianPruner(
        STARTUP_TRIALS, n_warmup_steps=cap // 10, n_min_trials=MEDIAN_TRIALS
    )


def _database(runs_dir: Path, batch: str) -> Path:
    return (runs_dir / batch / "optuna.sqlite3").resolve()


def _storage(runs_dir: Path, batch: str) -> str:
    path = _database(runs_dir, batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    return URL.create("sqlite", database=str(path)).render_as_string(hide_password=False)


def unfinished_studies(studies: list[Study], runs_dir: Path) -> list[str]:
    """Studies short of their trial budget or with a trial still running. Creates no database."""
    out = []
    for study in studies:
        trials = []
        if _database(runs_dir, study.batch).exists():
            try:
                loaded = optuna.load_study(
                    study_name=study.name, storage=_storage(runs_dir, study.batch)
                )
                trials = loaded.get_trials(deepcopy=False)
            except KeyError:  # the database has other studies only
                pass
        finished = sum(t.state.is_finished() for t in trials)
        if finished < study.trials or finished < len(trials):
            running = len(trials) - finished
            out.append(
                f"{study.name} ({finished} of {study.trials} trials finished, {running} running)"
            )
    return out


def _run_trial(study, opt: optuna.Study, trial: optuna.Trial, runs_dir, data_dir, device, log):
    cfg = study.config(trial.number, dict(trial.params))
    trial.set_user_attr("run_id", cfg.run_id)

    def report(step: int, value: float) -> bool:
        with warnings.catch_warnings():  # a resumed trial re-reports finished steps
            warnings.simplefilter("ignore", UserWarning)
            trial.report(value, step)
        return trial.should_prune()

    def train():
        return execute(cfg, runs_dir, data_dir, device, log=log, report=report)

    try:  # DeviceLost propagates: the trial stays running and resumes on the next run
        summary = train_with_retry(train, run_dir_of(cfg, runs_dir), device, log)
    except TrainingFailed:
        log(f"[tune] {cfg.run_id} failed; see error.txt")
        opt.tell(trial, state=TrialState.FAIL)
        return cfg.run_id
    if summary["stop_reason"] == "pruned":
        opt.tell(trial, state=TrialState.PRUNED)
    elif summary["status"] == "trained":
        opt.tell(trial, summary["validation"]["ndcg@10"])
    else:
        opt.tell(trial, state=TrialState.FAIL)
        return cfg.run_id
    return None


def run_study(
    study: Study, runs_dir: Path, data_dir: Path, device, log=print
) -> tuple[optuna.Study, list[str]]:
    """Run a study to its trial budget; returns it and the run ids of trials that failed now."""
    if study.arm not in ARMS:
        raise ValueError(f"unknown arm {study.arm!r}")
    cap = study.max_epochs or default_limits(study.arm)[0]
    opt = optuna.create_study(
        study_name=study.name,
        storage=_storage(runs_dir, study.batch),
        direction="maximize",
        load_if_exists=True,
        pruner=pruner(cap),
    )
    failed: list[str] = []
    for frozen in opt.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        if set(frozen.params) != set(SPACES[study.arm]):
            # interrupted while its parameters were being stored: it never trained
            log(f"[tune] {study.name}: trial {frozen.number} has incomplete parameters; failed")
            opt.tell(frozen.number, state=TrialState.FAIL)
            failed.append(study.config(frozen.number, {}).run_id)
            continue
        log(f"[tune] {study.name}: resuming trial {frozen.number}")
        trial = optuna.trial.Trial(opt, frozen._trial_id)
        failed += filter(None, [_run_trial(study, opt, trial, runs_dir, data_dir, device, log)])
    while (done := len(opt.trials)) < study.trials:
        opt.sampler = optuna.samplers.TPESampler(
            seed=study.sampler_seed * 10_000 + done, n_startup_trials=STARTUP_TRIALS
        )
        trial = opt.ask(SPACES[study.arm])
        log(f"[tune] {study.name}: trial {trial.number} of {study.trials}")
        failed += filter(None, [_run_trial(study, opt, trial, runs_dir, data_dir, device, log)])
    return opt, failed
