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
`--check` compares steps on several batches under the modes with the same steps of the arm's
own code; its loss difference is the one to read, since Adam turns rounding differences in
near-zero gradients into weight differences up to the size of the learning rate. The timings
are printed in a `timing` line before the profiled pass that measures GPU time. Under a CUDA
graph the GPU time covers only the kernels the profiler reports, so `gpu_events_per_step` is
printed next to it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
from torch.profiler._utils import _init_for_cuda_graphs

from seqrec import models  # noqa: F401  registers the arms
from seqrec.data.registry import DATASETS
from seqrec.models.trainer import BATCH_SIZE
from seqrec.runner.config import RunConfig
from seqrec.runner.run import ARMS, _prepare_splits, release_device_memory, seed_everything

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_modes import CAPTURE_WARMUP_STEPS, Stepper  # noqa: E402

MODES = ("fixed_shapes", "fused_adam", "cuda_graph")
CHECK_BATCHES = 3


def _steps(model, train, rows: torch.Tensor, n: int) -> float:
    """n training steps through the arm's own epoch loop, over the first n batches of rows;
    returns the mean loss. The loop ends by reading the loss, so the GPU has finished when
    it returns."""
    model.data.rows = rows[: n * BATCH_SIZE]
    loss = model.train_epoch(train, 0)["loss"]
    if not math.isfinite(loss):
        raise ValueError(f"the loss of a timed epoch is {loss}; its steps are not training")
    return loss


def device_events(events) -> tuple[float, int]:
    """GPU time and number of profiler events, each GPU event (kernel, copy, memset) counted
    once: CPU ops also report the device time of the kernels they launch, and GPU user
    annotations span other events."""
    times = [
        e.device_time_total
        for e in events
        if e.device_type == DeviceType.CUDA and not e.is_user_annotation
    ]
    return sum(times), len(times)


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


def _wall_cpu(model, train, rows, n: int) -> tuple[float, float, float]:
    wall, cpu = time.perf_counter(), time.process_time()
    loss = _steps(model, train, rows, n)
    return time.perf_counter() - wall, time.process_time() - cpu, loss


def _gpu(model, train, rows, n: int) -> tuple[float, int, bool]:
    """GPU time and events of n steps, and whether the profiler stopped collecting GPU events
    before the steps ended (a full buffer), which leaves both too low."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _steps(model, train, rows, n)
        stopped = torch.autograd._is_kineto_stopped()
    return *device_events(prof.events()), stopped


def gpu_readings(model, train, rows, steps: int, repeats: int, wall_ms: float, device) -> dict:
    """GPU time and GPU events per step from separate, profiled passes: the profiler slows
    the CPU side."""
    keys = ("gpu_ms_per_step", "gpu_busy_share", "gpu_events_per_step", "gpu_trace_truncated")
    if device.type != "cuda":
        return dict.fromkeys(keys)
    one = [_gpu(model, train, rows, 1) for _ in range(repeats)]
    us_1, events_1, _ = (statistics.median(x) for x in zip(*one, strict=True))
    us_n, events_n, stopped = _gpu(model, train, rows, steps)
    gpu_ms = (us_n - us_1) / 1000 / (steps - 1)
    readings = (
        round(gpu_ms, 3),
        round(gpu_ms / wall_ms, 3),
        round((events_n - events_1) / (steps - 1), 1),
        stopped or any(x[2] for x in one),
    )
    return dict(zip(keys, readings, strict=True))


def prepare_profiler_for_graphs() -> None:
    """PyTorch's workaround for compiled CUDA graphs on CUDA before 12.6, applied on every
    version to a graph captured by hand: the profiler is initialised before the capture and
    keeps CUPTI loaded between sessions."""
    os.environ["DISABLE_CUPTI_LAZY_REINIT"] = "1"
    os.environ["TEARDOWN_CUPTI"] = "0"
    _init_for_cuda_graphs()


def syncs_per_step(model, train, rows) -> tuple[float, int]:
    """CPU-GPU synchronisations per step: an epoch of 3 steps against one of 1 step. The
    backward pass runs on the calling thread meanwhile, where its warnings are counted. Also
    returns the count of the one-step epoch: every epoch reads its loss, so 0 there means
    that no warning reached the count."""
    counts = []
    for n in (1, 3):
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with (
                torch.autograd.set_multithreading_enabled(False),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                _steps(model, train, rows, n)
        finally:
            torch.cuda.set_sync_debug_mode(0)
        counts.append(sum("synchroniz" in str(w.message) for w in caught))
    return (counts[1] - counts[0]) / 2, counts[0]


def build(cfg: RunConfig, data_dir: Path, device: torch.device, modes: dict):
    seed_everything(cfg.seed)
    splits, _ = _prepare_splits(cfg, data_dir)  # phase "tune": the test split stays sealed
    model = ARMS[cfg.arm](cfg, splits.train, device)
    if any(modes.values()):
        model.train_epoch = Stepper(model, **modes).train_epoch
    return model, splits


def measure(
    cfg: RunConfig, data_dir: Path, device: torch.device, steps: int, warmup: int, repeats=5,
    modes: dict | None = None, sync_debug: bool = False, log=lambda timing: None,
):  # fmt: skip
    """`log` receives the timings and memory readings before the profiled pass starts."""
    modes = modes or {}
    if modes.get("cuda_graph"):
        prepare_profiler_for_graphs()
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
    if warmup:  # an epoch over no rows is one empty batch, whose loss is NaN
        _steps(model, splits.train, rows, warmup)

    # The median of several one-step epochs, against one epoch of `steps` steps.
    one = [_wall_cpu(model, splits.train, rows, 1) for _ in range(repeats)]
    wall_1, cpu_1, _ = (statistics.median(x) for x in zip(*one, strict=True))
    wall_n, cpu_n, loss = _wall_cpu(model, splits.train, rows, steps)
    wall_ms = (wall_n - wall_1) * 1000 / (steps - 1)
    cpu_ms = (cpu_n - cpu_1) * 1000 / (steps - 1)
    if wall_ms <= 0:
        raise ValueError(f"timing noise exceeds the step time over {steps} steps; use more steps")

    stats = torch.cuda.memory_stats(device) if device.type == "cuda" else None
    result = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dataset": cfg.dataset, "arm": cfg.arm, "params": cfg.params, "steps": steps,
        "modes": [m for m in MODES if modes.get(m)], "loss": loss,
        "wall_ms_per_step": round(wall_ms, 3),
        "cpu_ms_per_step": round(cpu_ms, 3),
        "epoch_overhead_ms": round(wall_1 * 1000 - wall_ms, 3),
    } | memory_readings(stats)  # fmt: skip
    log(result)
    result |= gpu_readings(model, splits.train, rows, steps, repeats, wall_ms, device)
    count_syncs = sync_debug and device.type == "cuda"
    syncs = syncs_per_step(model, splits.train, rows) if count_syncs else (None, None)
    return result | dict(zip(("syncs_per_step", "syncs_one_step_epoch"), syncs, strict=True))


def largest_diff(params_a, params_b) -> float:
    """The largest absolute difference between two lists of tensors; NaN if any is NaN."""
    diffs = [(a - b).abs().max() for a, b in zip(params_a, params_b, strict=True)]
    return torch.stack(diffs).max().item()


def _check_run(cfg: RunConfig, data_dir: Path, device: torch.device, modes: dict):
    """Losses of one step on each of CHECK_BATCHES batches, and the weights afterwards. The
    first batch gets the steps a graph capture takes on it beforehand, whatever the modes."""
    model, _ = build(cfg, data_dir, device, {})
    stepper = Stepper(model, **modes)
    rows = model.data.rows
    if len(rows) < CHECK_BATCHES * BATCH_SIZE:
        raise ValueError(f"--check needs {CHECK_BATCHES} full batches")
    made = [model.make_batch(b, model.gen) for b in rows.split(BATCH_SIZE)[:CHECK_BATCHES]]
    model.net.train()
    torch.set_float32_matmul_precision("high")
    if not modes.get("cuda_graph"):
        for _ in range(CAPTURE_WARMUP_STEPS):
            stepper.eager_step(made[0])
    losses = torch.stack([stepper.step(batch).float() for batch in made])
    return losses, [p.detach().float() for p in model.net.parameters()]


def check(cfg: RunConfig, data_dir: Path, device: torch.device, modes: dict) -> dict:
    """Steps on several batches with dropout 0, under the modes and under the arm's own
    code, from the same initial state: the largest differences of the losses and of the
    final weights, next to the same differences between two runs of the arm's own code.
    Under a CUDA graph all but the first batch are replayed on a batch other than the
    captured one. gSASRec draws negatives for every position under fixed shapes, so its
    reference keeps fixed shapes and the check covers only fused Adam and the CUDA graph."""
    cfg = dataclasses.replace(cfg, params=cfg.params | {"dropout": 0.0})
    ref_modes = {"fixed_shapes": modes["fixed_shapes"] and cfg.arm == "gsasrec"}
    if not any(modes[m] and not ref_modes.get(m) for m in MODES):
        raise ValueError("--check needs a mode that its reference does not use")
    (ref_losses, ref), (again_losses, again), (losses, got) = (
        _check_run(cfg, data_dir, device, m) for m in (ref_modes, ref_modes, modes)
    )
    return {
        "check_reference": [k for k, v in ref_modes.items() if v],
        "check_losses": [round(x, 6) for x in losses.tolist()],
        "check_loss_diff": (losses - ref_losses).abs().max().item(),
        "check_weight_diff": largest_diff(got, ref),
        "check_noise_loss_diff": (again_losses - ref_losses).abs().max().item(),
        "check_noise_weight_diff": largest_diff(again, ref),
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
    events = result.get("gpu_events_per_step")
    if events is not None and events <= 0:
        out.append(
            "the profiler reported no GPU events for the added steps; gpu_ms_per_step and "
            "gpu_busy_share are not measurements"
        )
    if result.get("gpu_trace_truncated"):
        out.append(
            "the profiler stopped collecting GPU events before the steps ended; GPU time and "
            "events per step are too low. Repeat with fewer steps"
        )
    if result.get("syncs_one_step_epoch") == 0:
        out.append(
            "no synchronisation was counted in an epoch that reads its loss, so the count does "
            "not work here; syncs_per_step is not a measurement"
        )
    if "cuda_graph" in result.get("modes", ()):
        out.append(
            "GPU time counts the kernels the profiler reports; if gpu_events_per_step is far "
            "below the run without the graph, replayed kernels are missing from it"
        )
    return out


def config_from_run(run_dir: Path) -> RunConfig:
    """The configuration of an existing run, as a training-only profiling run."""
    c = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return RunConfig(
        "profile", c["dataset"], c["split"], c["arm"], c["seed"], "tune",
        shuffle=c.get("shuffle", "none"), params=c["params"], max_len=c.get("max_len"),
    )  # fmt: skip


def _print_timing(timing: dict) -> None:
    print(f"timing {json.dumps(timing)}", flush=True)


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
    if args.warmup < 0:
        parser.error("--warmup must be 0 or more")
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
    try:
        if args.check:
            print(json.dumps(check(cfg, args.data_dir, device, modes)))
            return 0
        result = measure(cfg, args.data_dir, device, args.steps, args.warmup, modes=modes,
                         sync_debug=args.sync_debug, log=_print_timing)  # fmt: skip
    except ValueError as error:
        parser.error(str(error))
    print(f"steps {result['steps']}")
    for note in notes(result):
        print(f"note: {note}")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
