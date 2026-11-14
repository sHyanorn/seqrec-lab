"""Batches: `run` trains every pending run of a plan; `test` scores the trained ones."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import torch

from seqrec.analysis.power import NEURAL_ARMS
from seqrec.data.store import load_splits, log_path
from seqrec.eval.test_access import TestRefused
from seqrec.runner.attempts import (
    TRAIN_ATTEMPTS,
    DeviceLost,
    TrainingFailed,
    device_usable,
    record_failure,
    train_failures,
    train_with_retry,
)
from seqrec.runner.config import Plan, load_plan, read_plan
from seqrec.runner.export import read_earlier_exports, require_backed_reads
from seqrec.runner.run import (
    ARMS,
    RunConflict,
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


class SeedCountRefused(RuntimeError):
    """A final plan whose seeds do not follow the committed estimate of METHODS 7.5 (D064)."""


SEED_RECORD = "seeds-{dataset}.json"  # written by `seqrec seeds --write`


def plan_seed_problems(plan: Plan, dataset: str, n: int) -> list[str]:
    """Leave-one-out arms of a final plan whose seeds are not 1 to n: every neural arm, and
    every shuffle control, which has the seeds of the unshuffled runs (METHODS 3.3, 7.5)."""
    seeds: dict[tuple[str, str], set[int]] = {(arm, "none"): set() for arm in NEURAL_ARMS}
    for r in plan.runs:
        if (r.dataset, r.split, r.phase) == (dataset, "loo", "final") and r.arm in NEURAL_ARMS:
            seeds.setdefault((r.arm, r.shuffle), set()).add(r.seed)
    want = set(range(1, n + 1))
    return [
        f"{arm}{'' if shuffle == 'none' else f' ({shuffle})'}: seeds {sorted(have)}"
        for (arm, shuffle), have in sorted(seeds.items())
        if have != want
    ]


def require_seed_record(plan: Plan, runs_dir: Path, committed: Path | None) -> None:
    """Before any test read of final runs (D064): each dataset's committed estimate of
    METHODS 7.5 rests on 25 distinct runs of this plan, each with the arm and seed its entry
    gives and the validation value it records on this volume, and the plan has seeds 1 to n
    on leave-one-out. Raises before any read, so a refusal is not a read (D031)."""
    finals = [r for r in plan.runs if r.phase == "final" and r.arm in NEURAL_ARMS]
    by_id = {r.run_id: r for r in finals}
    for dataset in sorted({r.dataset for r in finals}):
        name = SEED_RECORD.format(dataset=dataset)
        if committed is None or not (committed / name).exists():
            raise SeedCountRefused(
                f"{dataset}: no {name} among the committed batch files; write it with "
                "`seqrec seeds --write` after the first five final seeds (METHODS 7.5)"
            )
        record = json.loads((committed / name).read_text(encoding="utf-8"))
        if record.get("dataset") != dataset:
            raise SeedCountRefused(f"{name} is the estimate for {record.get('dataset')}")
        runs = record["runs"]
        expected = {(arm, s) for arm in NEURAL_ARMS for s in range(1, 6)}
        if sorted((r["arm"], r["seed"]) for r in runs) != sorted(expected):
            raise SeedCountRefused(f"{name}: not one run for seeds 1 to 5 of every neural arm")
        if len({r["run_id"] for r in runs}) != len(runs):
            raise SeedCountRefused(f"{name}: a run is named twice")
        for run in runs:
            cfg = by_id.get(run["run_id"])
            want = (dataset, "loo", "none", run["arm"], run["seed"])
            if cfg is None or (cfg.dataset, cfg.split, cfg.shuffle, cfg.arm, cfg.seed) != want:
                raise SeedCountRefused(
                    f"{name}: {run['run_id']} is not a run of this plan with arm {run['arm']} "
                    f"and seed {run['seed']} on leave-one-out without shuffle"
                )
            path = run_dir_of(cfg, runs_dir) / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            value = (summary.get("validation") or {}).get("ndcg@10")
            if summary.get("status") != "trained" or value != run["validation_ndcg@10"]:
                raise SeedCountRefused(
                    f"{name}: {run['run_id']} has validation NDCG@10 {value} on this volume, "
                    f"{run['validation_ndcg@10']} in the record"
                )
        problems = plan_seed_problems(plan, dataset, record["n"])
        if problems:
            raise SeedCountRefused(
                f"{dataset}: METHODS 7.5 gives n = {record['n']}, so leave-one-out needs seeds "
                f"1 to {record['n']}; the plan has {'; '.join(problems)}"
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
    if "studies" in read_plan(plan_path):
        return unfinished_studies(load_tuning_plan(plan_path)[2], runs_dir)
    plan = load_plan(plan_path)
    _check_plan(plan, runs_dir)  # an edited plan is not the plan these runs were trained from
    return _untrained(plan, runs_dir)


def _train(cfg, runs_dir: Path, data_dir: Path, device, log) -> None:
    run_dir = run_dir_of(cfg, runs_dir)
    train_with_retry(
        lambda: execute(cfg, runs_dir, data_dir, device, log=log), run_dir, device, log
    )


def _test(cfg, runs_dir: Path, data_dir: Path, device, log, earlier: dict) -> bool:
    """Read test data for one run; a refusal is reported but not recorded as a failure.
    `earlier` are the reads made on another machine, as `test_run` takes them."""
    try:
        test_run(cfg, runs_dir, data_dir, device, log=log, **earlier)
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
    committed: Path | None = None,
):
    """Refuses until every run of the plan has finished training (METHODS 5.5); a run whose
    training failed twice has finished (D031) and is not tested, nor is a diverged run.
    `earlier` are directories of batch files exported from another machine; the anchor reads
    in them count toward the limit of METHODS 11.5. `committed` is the directory of batch
    files exported from this machine: a read recorded there and not made in `runs_dir` would
    be in neither count, so it stops the task. It also holds the seed record that final runs
    need before their read (`require_seed_record`, D064)."""
    if committed:  # before any test data, as the earlier exports below
        require_backed_reads(committed, runs_dir)
    earlier_anchor, earlier_final = Counter(), Counter()
    for r in read_earlier_exports(earlier):  # before any test data: a bad directory stops here
        earlier_anchor[(r["dataset"], r["arm"])] += r["reads_anchor"]
        earlier_final[r["run_id"]] += r["reads_final"]
    plan = load_plan(plan_path)
    _check_plan(plan, runs_dir)
    require_data({r.dataset for r in plan.runs}, data_dir)
    summaries = {r.run_id: run_dir_of(r, runs_dir) / "summary.json" for r in plan.runs}
    untrained = _untrained(plan, runs_dir)
    if untrained:
        raise RuntimeError(f"{len(untrained)} runs are not trained yet, e.g. {untrained[0]}")
    require_seed_record(plan, runs_dir, committed)
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
        elsewhere = {
            "earlier_anchor_reads": earlier_anchor[(cfg.dataset, cfg.arm)],
            "earlier_final_reads": earlier_final[cfg.run_id],
        }
        if not _test(cfg, runs_dir, data_dir, device, log, elsewhere):
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
