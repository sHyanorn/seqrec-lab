"""Time training steps of one arm on one prepared dataset (D039).

Prints wall time, process CPU time and GPU time (kernels, copies and memsets) per step, so
that the share of each step the GPU is busy can be read off. Per-step times are the
difference between an epoch of n steps and an epoch of one step, divided by n - 1, so the
fixed cost of an epoch (shuffle, first-batch check, reading the loss) is reported apart as
`epoch_overhead_ms`. Only full batches are timed. On a GPU it also prints the peaks of memory
PyTorch allocated and reserved, and how often its allocator asked the driver for memory or
retried after a failed request (D042). Reads only the training split and writes nothing.

    uv run python scripts/profile_step.py --dataset steam --arm sasrec-bce
    uv run python scripts/profile_step.py --from-run runs/m2-anchor/gsasrec-ml-1m-loo-s0-r1

`--fixed-shapes`, `--fused-adam` and `--cuda-graph` time the SASRec arms under the execution
modes of `step_modes.py`; `--sync-debug` counts the CPU-GPU synchronisations per step, and
`--check` compares one step under the modes with one step of the arm's own code.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from seqrec import models  # noqa: F401  registers the arms
from seqrec.data.registry import DATASETS
from seqrec.models.trainer import BATCH_SIZE
from seqrec.runner.config import RunConfig
from seqrec.runner.run import ARMS, _prepare_splits, release_device_memory, seed_everything

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_modes import Stepper  # noqa: E402

MODES = ("fixed_shapes", "fused_adam", "cuda_graph")


def _steps(model, train, rows: torch.Tensor, n: int) -> None:
    """n training steps through the arm's own epoch loop, over the first n batches of rows.
    The loop ends by reading the loss, so the GPU has finished when it returns."""
    model.data.rows = rows[: n * BATCH_SIZE]
    model.train_epoch(train, 0)


def device_us(events) -> float:
    """GPU time of profiler events, each GPU event (kernel, copy, memset) counted once: CPU ops
    also report the device time of the kernels they launch, and GPU user annotations span
    other events."""
    return sum(
        e.device_time_total
        for e in events
        if e.device_type == DeviceType.CUDA and not e.is_user_annotation
    )


def memory_readings(stats: dict | None) -> dict:
    """Allocator peaks and counts from `torch.cuda.memory_stats`; None values without CUDA. A
    reserved peak far above the allocated one means cached blocks fill the device (D042)."""
    stats = stats or {}

    def gib(key):
        return round(stats[key] / 2**30, 3) if key in stats else None

    return {
        "torch_allocated_gib_max": gib("allocated_bytes.all.peak"),
        "torch_reserved_gib_max": gib("reserved_bytes.all.peak"),
        "alloc_retries": stats.get("num_alloc_retries"),
        "device_allocs": stats.get("num_device_alloc"),
    }


def _wall_cpu(model, train, rows, n: int) -> tuple[float, float]:
    wall, cpu = time.perf_counter(), time.process_time()
    _steps(model, train, rows, n)
    return time.perf_counter() - wall, time.process_time() - cpu


def _gpu_us(model, train, rows, n: int) -> float:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _steps(model, train, rows, n)
    return device_us(prof.events())


def syncs_per_step(model, train, rows) -> float:
    """CPU-GPU synchronisations per step: an epoch of 3 steps against one of 1 step."""
    counts = []
    for n in (1, 3):
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _steps(model, train, rows, n)
        finally:
            torch.cuda.set_sync_debug_mode(0)
        counts.append(sum("synchroniz" in str(w.message) for w in caught))
    return (counts[1] - counts[0]) / 2


def build(cfg: RunConfig, data_dir: Path, device: torch.device, modes: dict):
    seed_everything(cfg.seed)
    splits, _ = _prepare_splits(cfg, data_dir)  # phase "tune": the test split stays sealed
    model = ARMS[cfg.arm](cfg, splits.train, device)
    if any(modes.values()):
        model.train_epoch = Stepper(model, **modes).train_epoch
    return model, splits


def measure(
    cfg: RunConfig, data_dir: Path, device: torch.device, steps: int, warmup: int, repeats=5,
    modes: dict | None = None, sync_debug: bool = False,
):  # fmt: skip
    modes = modes or {}
    release_device_memory(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.reset_accumulated_memory_stats(device)
    model, splits = build(cfg, data_dir, device, modes)
    rows = model.data.rows
    available = len(rows) // BATCH_SIZE  # full batches: a partial one has another shape
    steps, warmup = min(steps, available), min(warmup, available)
    if steps < 2:
        raise ValueError(f"{cfg.dataset} has {available} full batches; at least 2 are needed")
    _steps(model, splits.train, rows, warmup)

    # The median of several one-step epochs, against one epoch of `steps` steps.
    one = [_wall_cpu(model, splits.train, rows, 1) for _ in range(repeats)]
    wall_1, cpu_1 = (statistics.median(x) for x in zip(*one, strict=True))
    wall_n, cpu_n = _wall_cpu(model, splits.train, rows, steps)
    wall_ms = (wall_n - wall_1) * 1000 / (steps - 1)
    cpu_ms = (cpu_n - cpu_1) * 1000 / (steps - 1)
    if wall_ms <= 0:
        raise ValueError(f"timing noise exceeds the step time over {steps} steps; use more steps")

    stats = torch.cuda.memory_stats(device) if device.type == "cuda" else None
    gpu_ms = None
    if device.type == "cuda":  # separate, profiled passes: the profiler slows the CPU side
        gpu_1 = statistics.median(_gpu_us(model, splits.train, rows, 1) for _ in range(repeats))
        gpu_ms = (_gpu_us(model, splits.train, rows, steps) - gpu_1) / 1000 / (steps - 1)
    syncs = syncs_per_step(model, splits.train, rows) if sync_debug and gpu_ms is not None else None
    return {
        "dataset": cfg.dataset, "arm": cfg.arm, "params": cfg.params, "steps": steps,
        "modes": [m for m in MODES if modes.get(m)], "syncs_per_step": syncs,
        "wall_ms_per_step": round(wall_ms, 3),
        "cpu_ms_per_step": round(cpu_ms, 3),
        "gpu_ms_per_step": None if gpu_ms is None else round(gpu_ms, 3),
        "gpu_busy_share": None if gpu_ms is None else round(gpu_ms / wall_ms, 3),
        "epoch_overhead_ms": round(wall_1 * 1000 - wall_ms, 3),
    } | memory_readings(stats)  # fmt: skip


def check(cfg: RunConfig, data_dir: Path, device: torch.device, modes: dict) -> dict:
    """Four steps on one fixed batch with dropout 0, under the modes and under the arm's own
    code, from the same initial state; the largest differences of the last loss and of the
    weights. gSASRec draws negatives for every position under fixed shapes, so its reference
    keeps fixed shapes and checks only fused Adam and the CUDA graph."""
    cfg = dataclasses.replace(cfg, params=cfg.params | {"dropout": 0.0})
    ref_modes = {"fixed_shapes": modes["fixed_shapes"] and cfg.arm == "gsasrec"}
    out = []
    for m in (ref_modes, modes):
        model, _ = build(cfg, data_dir, device, {})
        stepper = Stepper(model, **m)
        batch = model.make_batch(model.data.rows[:BATCH_SIZE], model.gen)
        model.net.train()
        torch.set_float32_matmul_precision("high")
        if m.get("cuda_graph"):  # the capture runs three eager steps, then one replay
            loss = stepper.step(batch)
        else:
            for _ in range(4):
                loss = stepper.eager_step(batch)
        out.append((loss.float(), [p.detach().float() for p in model.net.parameters()]))
    (loss_a, params_a), (loss_b, params_b) = out
    return {
        "check_reference": [k for k, v in ref_modes.items() if v],
        "check_loss_diff": abs(loss_a - loss_b).item(),
        "check_weight_diff": max(
            (a - b).abs().max().item() for a, b in zip(params_a, params_b, strict=True)
        ),
    }


def notes(result: dict) -> list[str]:
    """Warnings for results outside the script's assumptions (D039)."""
    out = []
    busy = result.get("gpu_busy_share")
    if busy is not None and busy > 1:
        out.append(
            "GPU-bound; GPU time comes from a separate profiled pass, so the busy share only "
            "says that the GPU limits the step"
        )
    if result["epoch_overhead_ms"] < 0:
        out.append(
            "a one-step epoch was faster than a step of the long epoch (a higher clock after an "
            "idle GPU, or a slowdown during the long epoch); epoch_overhead_ms is not meaningful"
        )
    return out


def config_from_run(run_dir: Path) -> RunConfig:
    """The configuration of an existing run, as a training-only profiling run."""
    c = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return RunConfig(
        "profile", c["dataset"], c["split"], c["arm"], c["seed"], "tune",
        shuffle=c.get("shuffle", "none"), params=c["params"], max_len=c.get("max_len"),
    )  # fmt: skip


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--from-run", type=Path, help="a run directory; its config.json replaces the next three"
    )
    parser.add_argument("--dataset", choices=list(DATASETS))
    parser.add_argument("--arm")
    parser.add_argument("--params", help="JSON, e.g. '{\"dim\": 256}'")
    parser.add_argument("--steps", type=int, default=200, help="at least 2")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--fixed-shapes", action="store_true", help="loss over all positions")
    parser.add_argument("--fused-adam", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true", help="needs --fixed-shapes")
    parser.add_argument("--sync-debug", action="store_true", help="count syncs per step")
    parser.add_argument("--check", action="store_true", help="compare with the arm's own code")
    args = parser.parse_args(argv)
    if args.cuda_graph and not args.fixed_shapes:
        parser.error("--cuda-graph needs --fixed-shapes")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    given = [f for f in ("dataset", "arm", "params") if getattr(args, f) is not None]
    if args.from_run and given:
        parser.error(f"--from-run takes the place of --{', --'.join(given)}")
    if not args.from_run and not (args.dataset and args.arm):
        parser.error("give --from-run, or --dataset and --arm")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.from_run:
        cfg = config_from_run(args.from_run)
    else:
        params = json.loads(args.params or "{}")
        cfg = RunConfig("profile", args.dataset, "loo", args.arm, 0, "tune", params=params)
    modes = {m: getattr(args, m) for m in MODES}
    if args.check:
        print(json.dumps(check(cfg, args.data_dir, device, modes)))
        return 0
    result = measure(cfg, args.data_dir, device, args.steps, args.warmup, modes=modes,
                     sync_debug=args.sync_debug)  # fmt: skip
    print(f"steps {result['steps']}")
    for note in notes(result):
        print(f"note: {note}")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
