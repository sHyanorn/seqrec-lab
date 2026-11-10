"""Batches: `run` trains every pending run of a plan; `test` scores the trained ones."""

from __future__ import annotations

import json
import tomllib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import torch

from seqrec.data.store import load_splits, log_path
from seqrec.runner.attempts import (
    TRAIN_ATTEMPTS,
    DeviceLost,
    TrainingFailed,
    device_usable,
    record_failure,
    train_failures,
    train_with_retry,
)
from seqrec.runner.config import Plan, load_plan
from seqrec.runner.export import read_batch_files
from seqrec.runner.run import (
    ARMS,
    RunConflict,
    TestRefused,
    config_conflicts,
    execute,
    run_dir_of,
    test_run,
)
from seqrec.runner.tune import load_tuning_plan, run_study, unfinished_studies
from seqrec.telemetry import warmup_gpu


def _check_plan(plan: Plan, runs_dir: Path) -> None:
    unknown = sorted({r.arm for r in plan.runs} - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arms in plan: {unknown}")
    conflicts = config_conflicts(plan.runs, runs_dir)
    if conflicts:  # an edited plan would otherwise mix configurations (METHODS 5.5)
        raise RunConflict(
            f"the plan differs from {len(conflicts)} existing run directories, e.g. "
            f"{conflicts[0]}; restore the plan or use a new batch name"
        )


def require_data(datasets, data_dir: Path) -> None:
    """Fail before any attempt when a dataset has not been prepared."""
    missing = sorted({d for d in datasets if not log_path(data_dir, d).exists()})
    if missing:
        raise FileNotFoundError(
            f"not prepared in {data_dir}: {', '.join(missing)} (run `seqrec data prepare`)"
        )


def _given_up(cfg, runs_dir: Path) -> bool:
    return train_failures(run_dir_of(cfg, runs_dir)) >= TRAIN_ATTEMPTS


def _untrained(plan: Plan, runs_dir: Path) -> list[str]:
    """Runs neither trained nor given up after repeated failures (D031)."""
    return [
        r.run_id
        for r in plan.runs
        if not (run_dir_of(r, runs_dir) / "summary.json").exists() and not _given_up(r, runs_dir)
    ]


def unfinished(plan_path: Path, runs_dir: Path) -> list[str]:
    """What is left of a plan of either kind: runs to train, or studies short of their trial
    budget. An export of the plan's batch is complete only when this is empty."""
    if "studies" in tomllib.loads(plan_path.read_text(encoding="utf-8-sig")):
        return unfinished_studies(load_tuning_plan(plan_path)[2], runs_dir)
    return _untrained(load_plan(plan_path), runs_dir)


def _train(cfg, runs_dir: Path, data_dir: Path, device, log) -> None:
    run_dir = run_dir_of(cfg, runs_dir)
    train_with_retry(
        lambda: execute(cfg, runs_dir, data_dir, device, log=log), run_dir, device, log
    )


def _test(cfg, runs_dir: Path, data_dir: Path, device, log, earlier_anchor_reads: int) -> bool:
    """Read test data for one run; a refusal is reported but not recorded as a failure."""
    try:
        test_run(
            cfg, runs_dir, data_dir, device, log=log, earlier_anchor_reads=earlier_anchor_reads
        )
        return True
    except TestRefused as err:
        log(f"[batch] {cfg.run_id}: test read refused: {err}")
    except Exception:
        record_failure(run_dir_of(cfg, runs_dir), "test")
        log(f"[batch] {cfg.run_id}: test read failed; see error.txt")
    return False


def _warm_up(warmup_s: float, dataset: str, split: str, data_dir: Path, device, log) -> None:
    if warmup_s > 0 and device.type == "cuda":
        load_splits(data_dir, dataset, split)  # build on CPU before the GPU warms up
        log(f"[batch] warming up the GPU for {warmup_s:.0f} s")
        warmup_gpu(warmup_s, device)


def run_batch(plan_path: Path, runs_dir: Path, data_dir: Path, device: torch.device, log=print):
    plan = load_plan(plan_path)
    _check_plan(plan, runs_dir)
    pending = [r for r in plan.runs if not (run_dir_of(r, runs_dir) / "summary.json").exists()]
    require_data({r.dataset for r in pending}, data_dir)
    failed = [r.run_id for r in pending if _given_up(r, runs_dir)]
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
        except RunConflict as err:  # the plan changed under an existing run directory
            failed.append(cfg.run_id)
            log(f"[batch] {err}")
        except DeviceLost as err:
            failed.append(err.run_id)
            log(f"[batch] {err}")
            break
    log(f"[batch] training done; {len(failed)} failed")
    return failed


def test_batch(
    plan_path: Path,
    runs_dir: Path,
    data_dir: Path,
    device: torch.device,
    log=print,
    earlier: Sequence[Path] = (),
):
    """Refuses until every run of the plan has finished training (METHODS 5.5); a run whose
    training failed twice has finished (D031) and is not tested, nor is a diverged run.
    `earlier` are directories of batch files exported from another machine; the anchor reads
    in them count toward the limit of METHODS 11.5."""
    earlier_reads = Counter()
    for directory in earlier:  # read before any test data: a missing directory must stop here
        for r in read_batch_files(directory):
            earlier_reads[(r["dataset"], r["arm"])] += r["reads_anchor"]
    plan = load_plan(plan_path)
    _check_plan(plan, runs_dir)
    require_data({r.dataset for r in plan.runs}, data_dir)
    summaries = {r.run_id: run_dir_of(r, runs_dir) / "summary.json" for r in plan.runs}
    untrained = _untrained(plan, runs_dir)
    if untrained:
        raise RuntimeError(f"{len(untrained)} runs are not trained yet, e.g. {untrained[0]}")
    failed: list[str] = []
    for cfg in plan.runs:
        if cfg.phase not in ("final", "anchor"):
            continue
        if not summaries[cfg.run_id].exists():
            log(f"[batch] {cfg.run_id}: training failed twice; not tested")
            continue
        status = json.loads(summaries[cfg.run_id].read_text(encoding="utf-8"))["status"]
        if status != "trained":
            log(f"[batch] {cfg.run_id}: {status}; not tested")
            continue
        if not _test(cfg, runs_dir, data_dir, device, log, earlier_reads[(cfg.dataset, cfg.arm)]):
            failed.append(cfg.run_id)
            if not device_usable(device):
                log("[batch] the GPU stopped working; rerun the same command to continue")
                break
    log(f"[batch] testing done; {len(failed)} failed")
    return failed


def tune_batch(plan_path: Path, runs_dir: Path, data_dir: Path, device: torch.device, log=print):
    """Run every study of a tuning plan to its trial budget; resumes after interruption."""
    batch, warmup_s, studies = load_tuning_plan(plan_path)
    require_data({s.dataset for s in studies}, data_dir)
    if studies:
        _warm_up(warmup_s, studies[0].dataset, "loo", data_dir, device, log)
    failed: list[str] = []
    for study in studies:
        try:
            opt, failed_now = run_study(study, runs_dir, data_dir, device, log)
        except DeviceLost as err:  # the trial stays running and resumes on the next run
            log(f"[tune] {err}")
            return failed + [err.run_id]
        except RunConflict as err:  # the plan changed under an existing trial directory
            log(f"[tune] {study.name}: {err}")
            failed.append(study.name)
            continue
        failed += failed_now
        complete = [t for t in opt.trials if t.state.name == "COMPLETE"]
        best = max(complete, key=lambda t: t.value, default=None)
        best_id = best.user_attrs["run_id"] if best else None
        log(f"[tune] {study.name}: {len(complete)} of {len(opt.trials)} complete; best {best_id}")
    log(f"[tune] done; {len(failed)} trials failed in this session")
    return failed
