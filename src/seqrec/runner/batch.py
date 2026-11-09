"""Batches: `run` trains every pending run of a plan; `test` scores the trained ones."""

from __future__ import annotations

import traceback
from pathlib import Path

import torch

from seqrec.data.store import load_splits
from seqrec.runner.config import Plan, load_plan
from seqrec.runner.run import ARMS, execute, run_dir_of, test_run
from seqrec.runner.tune import load_tuning_plan, run_study
from seqrec.telemetry import warmup_gpu


def _check_arms(plan: Plan) -> None:
    unknown = sorted({r.arm for r in plan.runs} - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arms in plan: {unknown}")


def _guarded(step, cfg, runs_dir: Path, log, failed: list[str], *args) -> None:
    try:
        step(cfg, runs_dir, *args, log=log)
    except Exception:
        failed.append(cfg.run_id)
        run_dir = run_dir_of(cfg, runs_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        log(f"[batch] {cfg.run_id} failed; see error.txt")


def _warm_up(warmup_s: float, dataset: str, split: str, data_dir: Path, device, log) -> None:
    if warmup_s > 0 and device.type == "cuda":
        load_splits(data_dir, dataset, split)  # build on CPU before the GPU warms up
        log(f"[batch] warming up the GPU for {warmup_s:.0f} s")
        warmup_gpu(warmup_s, device)


def run_batch(plan_path: Path, runs_dir: Path, data_dir: Path, device: torch.device, log=print):
    plan = load_plan(plan_path)
    _check_arms(plan)
    pending = [r for r in plan.runs if not (run_dir_of(r, runs_dir) / "summary.json").exists()]
    log(f"[batch] {plan.batch}: {len(pending)} of {len(plan.runs)} runs pending")
    if pending:
        _warm_up(plan.warmup_s, pending[0].dataset, pending[0].split, data_dir, device, log)
    failed: list[str] = []
    for cfg in pending:
        _guarded(execute, cfg, runs_dir, log, failed, data_dir, device)
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
            _guarded(test_run, cfg, runs_dir, log, failed, data_dir, device)
    log(f"[batch] testing done; {len(failed)} failed")
    return failed


def tune_batch(plan_path: Path, runs_dir: Path, data_dir: Path, device: torch.device, log=print):
    """Run every study of a tuning plan to its trial budget; resumes after interruption."""
    batch, warmup_s, studies = load_tuning_plan(plan_path)
    if studies:
        _warm_up(warmup_s, studies[0].dataset, "loo", data_dir, device, log)
    for study in studies:
        opt = run_study(study, runs_dir, data_dir, device, log)
        complete = [t for t in opt.trials if t.state.name == "COMPLETE"]
        best = max(complete, key=lambda t: t.value, default=None)
        best_id = best.user_attrs["run_id"] if best else None
        log(f"[tune] {study.name}: {len(complete)} of {len(opt.trials)} complete; best {best_id}")
    return []
