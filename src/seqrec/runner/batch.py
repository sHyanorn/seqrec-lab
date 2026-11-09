"""Batches: `run` trains every pending run of a plan; `test` scores the trained ones."""

from __future__ import annotations

from pathlib import Path

import torch

from seqrec.data.store import load_splits
from seqrec.runner.attempts import (
    TRAIN_ATTEMPTS,
    DeviceLost,
    TrainingFailed,
    record_failure,
    train_failures,
    train_with_retry,
)
from seqrec.runner.config import Plan, load_plan
from seqrec.runner.run import ARMS, execute, run_dir_of, test_run
from seqrec.runner.tune import load_tuning_plan, run_study
from seqrec.telemetry import warmup_gpu


def _check_arms(plan: Plan) -> None:
    unknown = sorted({r.arm for r in plan.runs} - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arms in plan: {unknown}")


def _train(cfg, runs_dir: Path, data_dir: Path, device, log) -> None:
    run_dir = run_dir_of(cfg, runs_dir)
    train_with_retry(
        lambda: execute(cfg, runs_dir, data_dir, device, log=log), run_dir, device, log
    )


def _test(cfg, runs_dir: Path, data_dir: Path, device, log, failed: list[str]) -> None:
    try:
        test_run(cfg, runs_dir, data_dir, device, log=log)
    except Exception:
        failed.append(cfg.run_id)
        record_failure(run_dir_of(cfg, runs_dir), "test")
        log(f"[batch] {cfg.run_id}: test read failed; see error.txt")


def _warm_up(warmup_s: float, dataset: str, split: str, data_dir: Path, device, log) -> None:
    if warmup_s > 0 and device.type == "cuda":
        load_splits(data_dir, dataset, split)  # build on CPU before the GPU warms up
        log(f"[batch] warming up the GPU for {warmup_s:.0f} s")
        warmup_gpu(warmup_s, device)


def run_batch(plan_path: Path, runs_dir: Path, data_dir: Path, device: torch.device, log=print):
    plan = load_plan(plan_path)
    _check_arms(plan)
    pending = [r for r in plan.runs if not (run_dir_of(r, runs_dir) / "summary.json").exists()]
    failed = [
        r.run_id for r in pending if train_failures(run_dir_of(r, runs_dir)) >= TRAIN_ATTEMPTS
    ]
    pending = [r for r in pending if r.run_id not in failed]
    log(f"[batch] {plan.batch}: {len(pending)} of {len(plan.runs)} runs pending")
    for run_id in failed:
        log(f"[batch] {run_id} failed twice; not retried")
    if pending:
        _warm_up(plan.warmup_s, pending[0].dataset, pending[0].split, data_dir, device, log)
    for cfg in pending:
        try:
            _train(cfg, runs_dir, data_dir, device, log)
        except TrainingFailed:
            failed.append(cfg.run_id)
        except DeviceLost as err:
            failed.append(err.run_id)
            log(f"[batch] {err}")
            break
    log(f"[batch] training done; {len(failed)} failed")
    return failed


def test_batch(plan_path: Path, runs_dir: Path, data_dir: Path, device: torch.device, log=print):
    """Refuses until every run of the plan has finished training (METHODS 5.5)."""
    plan = load_plan(plan_path)
    _check_arms(plan)
    untrained = [
        r.run_id for r in plan.runs if not (run_dir_of(r, runs_dir) / "summary.json").exists()
    ]
    if untrained:
        raise RuntimeError(f"{len(untrained)} runs are not trained yet, e.g. {untrained[0]}")
    failed: list[str] = []
    for cfg in plan.runs:
        if cfg.phase in ("final", "anchor"):
            _test(cfg, runs_dir, data_dir, device, log, failed)
    log(f"[batch] testing done; {len(failed)} failed")
    return failed


def tune_batch(plan_path: Path, runs_dir: Path, data_dir: Path, device: torch.device, log=print):
    """Run every study of a tuning plan to its trial budget; resumes after interruption."""
    batch, warmup_s, studies = load_tuning_plan(plan_path)
    if studies:
        _warm_up(warmup_s, studies[0].dataset, "loo", data_dir, device, log)
    failed: list[str] = []
    for study in studies:
        try:
            opt, failed_now = run_study(study, runs_dir, data_dir, device, log)
        except DeviceLost as err:  # the trial stays running and resumes on the next run
            log(f"[tune] {err}")
            return failed + [err.run_id]
        failed += failed_now
        complete = [t for t in opt.trials if t.state.name == "COMPLETE"]
        best = max(complete, key=lambda t: t.value, default=None)
        best_id = best.user_attrs["run_id"] if best else None
        log(f"[tune] {study.name}: {len(complete)} of {len(opt.trials)} complete; best {best_id}")
    log(f"[tune] done; {len(failed)} trials failed in this session")
    return failed
