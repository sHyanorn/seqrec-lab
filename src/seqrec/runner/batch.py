"""Run every pending run of a plan; finished runs are skipped, interrupted ones resume."""

from __future__ import annotations

import traceback
from pathlib import Path

import torch

from seqrec.runner.config import load_plan
from seqrec.runner.run import execute
from seqrec.telemetry import warmup_gpu


def run_batch(plan_path: Path, runs_dir: Path, data_dir: Path, device: torch.device, log=print):
    plan = load_plan(plan_path)
    pending = [
        r for r in plan.runs if not (runs_dir / r.batch / r.run_id / "summary.json").exists()
    ]
    log(f"[batch] {plan.batch}: {len(pending)} of {len(plan.runs)} runs pending")
    if pending and plan.warmup_s > 0 and device.type == "cuda":
        log(f"[batch] warming up the GPU for {plan.warmup_s:.0f} s")
        warmup_gpu(plan.warmup_s, device)
    failed = []
    for cfg in pending:
        try:
            execute(cfg, runs_dir, data_dir, device, log)
        except Exception:
            failed.append(cfg.run_id)
            run_dir = runs_dir / cfg.batch / cfg.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            log(f"[batch] {cfg.run_id} failed; see error.txt")
    log(f"[batch] done; {len(failed)} failed")
    return failed
