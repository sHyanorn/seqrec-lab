"""Time training steps of one arm on one prepared dataset (D039).

Prints wall time, process CPU time and GPU kernel time per step, so that the share of each
step the GPU spends computing can be read off. Per-step times are the difference between an
epoch of n steps and an epoch of one step, divided by n - 1, so the fixed cost of an epoch
(shuffle, first-batch check, reading the loss) is reported apart as `epoch_overhead_ms`.
Reads only the training split and writes nothing.

    uv run python scripts/profile_step.py --dataset steam --arm sasrec-bce
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from seqrec import models  # noqa: F401  registers the arms
from seqrec.data.registry import DATASETS
from seqrec.models.trainer import BATCH_SIZE
from seqrec.runner.config import RunConfig
from seqrec.runner.run import ARMS, _prepare_splits, seed_everything


def _steps(model, train, rows: torch.Tensor, n: int) -> None:
    """n training steps through the arm's own epoch loop, over the first n batches of rows.
    The loop ends by reading the loss, so the GPU has finished when it returns."""
    model.data.rows = rows[: n * BATCH_SIZE]
    model.train_epoch(train, 0)


def device_us(events) -> float:
    """GPU time of profiler events, each kernel counted once: CPU ops also report the device
    time of the kernels they launch, and GPU user annotations span other events."""
    return sum(
        e.device_time_total
        for e in events
        if e.device_type == DeviceType.CUDA and not e.is_user_annotation
    )


def _wall_cpu(model, train, rows, n: int) -> tuple[float, float]:
    wall, cpu = time.perf_counter(), time.process_time()
    _steps(model, train, rows, n)
    return time.perf_counter() - wall, time.process_time() - cpu


def _gpu_us(model, train, rows, n: int) -> float:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _steps(model, train, rows, n)
    return device_us(prof.events())


def measure(
    cfg: RunConfig, data_dir: Path, device: torch.device, steps: int, warmup: int, repeats=5
):
    seed_everything(cfg.seed)
    splits, _ = _prepare_splits(cfg, data_dir)  # phase "tune": the test split stays sealed
    model = ARMS[cfg.arm](cfg, splits.train, device)
    rows = model.data.rows
    available = math.ceil(len(rows) / BATCH_SIZE)
    steps, warmup = min(steps, available), min(warmup, available)
    if steps < 2:
        raise ValueError(f"{cfg.dataset} has {available} batches; at least 2 are needed")
    _steps(model, splits.train, rows, warmup)

    # The median of several one-step epochs, against one epoch of `steps` steps.
    one = [_wall_cpu(model, splits.train, rows, 1) for _ in range(repeats)]
    wall_1, cpu_1 = (statistics.median(x) for x in zip(*one, strict=True))
    wall_n, cpu_n = _wall_cpu(model, splits.train, rows, steps)
    wall_ms = (wall_n - wall_1) * 1000 / (steps - 1)
    cpu_ms = (cpu_n - cpu_1) * 1000 / (steps - 1)

    gpu_ms = None
    if device.type == "cuda":  # separate, profiled passes: the profiler slows the CPU side
        gpu_1 = statistics.median(_gpu_us(model, splits.train, rows, 1) for _ in range(repeats))
        gpu_ms = (_gpu_us(model, splits.train, rows, steps) - gpu_1) / 1000 / (steps - 1)
    return {
        "dataset": cfg.dataset, "arm": cfg.arm, "params": cfg.params, "steps": steps,
        "wall_ms_per_step": round(wall_ms, 3),
        "cpu_ms_per_step": round(cpu_ms, 3),
        "gpu_kernel_ms_per_step": None if gpu_ms is None else round(gpu_ms, 3),
        "gpu_busy_share": None if gpu_ms is None else round(gpu_ms / wall_ms, 3),
        "epoch_overhead_ms": round(wall_1 * 1000 - wall_ms, 3),
    }  # fmt: skip


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--params", default="{}", help="JSON, e.g. '{\"dim\": 256}'")
    parser.add_argument("--steps", type=int, default=200, help="at least 2")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = RunConfig(
        "profile", args.dataset, "loo", args.arm, 0, "tune", params=json.loads(args.params)
    )
    result = measure(cfg, args.data_dir, device, args.steps, args.warmup)
    print(f"steps {result['steps']}")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
