"""Environment feasibility check.

Probes the local PyTorch and GPU stack and writes a JSON report plus a text summary
to results/env/. A failing probe is recorded as data; the run itself only fails on
a bug in this script.

    uv run python scripts/check_env.py            # full check on the GPU (about 10 minutes)
    uv run python scripts/check_env.py --quick    # tiny shapes, also runs on CPU

The report holds hardware, software and timing data only. User names, host names
and file paths are removed before anything is written.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import functools
import getpass
import importlib.metadata
import json
import math
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "results" / "env"

FULL = {
    "quick": False,
    "matmul_n": 4096,
    "matmul_reps": 20,
    "sdpa_batch": 128,
    "sdpa_heads": 2,
    "sdpa_head_dims": [32, 64, 128, 256],
    "sdpa_lengths": [50, 200, 800],
    "sdpa_reps": 10,
    "model_batch": 128,
    "model_len": 200,
    "model_dim": 64,
    "model_items": [3_500, 26_000, 100_000],
    "model_negatives": 256,
    "model_reps": 20,
    "score_users": 4096,
    "score_chunk": 1024,
    "thermal_seconds": 120.0,
    "thermal_matmul_n": 8192,
    "compile_timeout_s": 600,
}
QUICK = {
    "quick": True,
    "matmul_n": 128,
    "matmul_reps": 2,
    "sdpa_batch": 4,
    "sdpa_heads": 2,
    "sdpa_head_dims": [16],
    "sdpa_lengths": [16],
    "sdpa_reps": 2,
    "model_batch": 4,
    "model_len": 16,
    "model_dim": 16,
    "model_items": [200],
    "model_negatives": 8,
    "model_reps": 2,
    "score_users": 64,
    "score_chunk": 32,
    "thermal_seconds": 1.0,
    "thermal_matmul_n": 128,
    "compile_timeout_s": 600,
}

BACKENDS = {
    "math": SDPBackend.MATH,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}
# none: bidirectional, no mask; causal: is_causal=True; padding: bidirectional with left
# padding (BERT4Rec); causal_padding: causal plus left padding as a boolean mask (SASRec).
MASK_MODES = ["none", "causal", "padding", "causal_padding"]
# Relative max error against the fp32 math backend (outputs and q, k, v gradients).
REL_TOL = {torch.float32: 1e-3, torch.bfloat16: 5e-2, torch.float16: 1e-2}
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# ---------------------------------------------------------------- utilities


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_ms(fn, device: torch.device, warmup: int, reps: int) -> list[float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(reps):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
    return times


def summarize(times: list[float]) -> dict:
    return {
        "median_ms": round(statistics.median(times), 4),
        "min_ms": round(min(times), 4),
        "max_ms": round(max(times), 4),
        "reps": len(times),
    }


def error_text(exc: BaseException) -> str:
    """First and last non-empty lines; compiler errors put the cause at the end."""
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    detail = " | ".join(dict.fromkeys([lines[0], lines[-1]])) if lines else ""
    return f"{type(exc).__name__}: {detail}"[:400]


def _try(fn):
    try:
        value = fn()
    except Exception as exc:  # probes record failures instead of raising
        return {"error": error_text(exc)}
    return value.decode() if isinstance(value, bytes) else value


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@contextlib.contextmanager
def tf32(enabled: bool):
    """Set TF32 for fp32 matmuls and restore the previous state afterwards."""
    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):
        previous = matmul.fp32_precision
        matmul.fp32_precision = "tf32" if enabled else "ieee"
    else:
        previous = matmul.allow_tf32
        matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        if hasattr(matmul, "fp32_precision"):
            matmul.fp32_precision = previous
        else:
            matmul.allow_tf32 = previous


def _free(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ------------------------------------------------------------ privacy filter

# Paths run to the next quote or line end, so names with spaces are covered too.
_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:[\\/][^\"'<>|\r\n]*"),
    re.compile(r"\\\\[^\"'<>|\r\n]*"),
    re.compile(r"/(?:home|Users|root|tmp|var|usr|opt|mnt|private|etc)/[^\"'<>|\r\n]*"),
]
PATH_MARK, NAME_MARK = "<p>", "<u>"  # cannot contain a user or host name
_ROOT_VARS = ("USERPROFILE", "LOCALAPPDATA", "APPDATA", "TEMP", "TMP")
_ROOT_VARS += ("ONEDRIVE", "ONEDRIVECOMMERCIAL", "ONEDRIVECONSUMER")
_NAME_VARS = ("USERNAME", "USER", "LOGNAME", "COMPUTERNAME", "USERDOMAIN")


def sensitive_roots() -> list[str]:
    roots = {str(REPO_ROOT), str(Path.home()), sys.prefix, sys.base_prefix, tempfile.gettempdir()}
    roots |= {os.environ[k] for k in _ROOT_VARS if os.environ.get(k)}
    variants = set()
    for root in roots:
        if len(root) > 3:
            variants |= {root, root.replace("\\", "/"), root.replace("/", "\\")}
    return sorted(variants, key=len, reverse=True)


def sensitive_tokens() -> list[str]:
    candidates = [socket.gethostname(), platform.node(), Path.home().name]
    candidates += [os.environ.get(k) for k in _NAME_VARS]
    with contextlib.suppress(Exception):
        candidates.append(getpass.getuser())
    return sorted({c for c in candidates if c and len(c) >= 3}, key=len, reverse=True)


def scrub_text(text: str, roots: list[str], tokens: list[str]) -> str:
    for root in roots:
        text = re.sub(re.escape(root), PATH_MARK, text, flags=re.IGNORECASE)
    for pattern in _PATH_PATTERNS:
        text = pattern.sub(PATH_MARK, text)
    for token in tokens:
        text = re.sub(re.escape(token), NAME_MARK, text, flags=re.IGNORECASE)
    return text


def scrub(obj, roots: list[str], tokens: list[str]):
    """Scrub string values and drop non-finite floats. Keys are fixed by this script."""
    if isinstance(obj, str):
        return scrub_text(obj, roots, tokens)
    if isinstance(obj, dict):
        return {k: scrub(v, roots, tokens) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [scrub(v, roots, tokens) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# --------------------------------------------------------------------- NVML


def _nvml():
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as exc:
        return None, error_text(exc)


def _event_reasons(nv, handle) -> int:
    if hasattr(nv, "nvmlDeviceGetCurrentClocksEventReasons"):
        return nv.nvmlDeviceGetCurrentClocksEventReasons(handle)
    return nv.nvmlDeviceGetCurrentClocksThrottleReasons(handle)


def _temperature(nv, handle) -> int:
    try:
        return nv.nvmlDeviceGetTemperatureV(handle, nv.NVML_TEMPERATURE_GPU)
    except Exception:  # older drivers lack the versioned call
        return nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)


REASON_BITS = {
    "sw_power_cap": "nvmlClocksEventReasonSwPowerCap",
    "hw_slowdown": "nvmlClocksEventReasonHwSlowdown",
    "sw_thermal": "nvmlClocksEventReasonSwThermalSlowdown",
    "hw_thermal": "nvmlClocksEventReasonHwThermalSlowdown",
    "hw_power_brake": "nvmlClocksEventReasonHwPowerBrakeSlowdown",
}


def nvml_static() -> dict:
    nv, handle = _nvml()
    if nv is None:
        return {"available": False, "error": handle}
    mem_clocks = _try(lambda: nv.nvmlDeviceGetSupportedMemoryClocks(handle))
    return {
        "available": True,
        "driver_version": _try(nv.nvmlSystemGetDriverVersion),
        "cuda_driver_version": _try(nv.nvmlSystemGetCudaDriverVersion),
        "name": _try(lambda: nv.nvmlDeviceGetName(handle)),
        "power_limit_w": _try(lambda: nv.nvmlDeviceGetPowerManagementLimit(handle) / 1000),
        "power_limit_default_w": _try(
            lambda: nv.nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000
        ),
        "enforced_power_limit_w": _try(lambda: nv.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000),
        "max_sm_clock_mhz": _try(lambda: nv.nvmlDeviceGetMaxClockInfo(handle, nv.NVML_CLOCK_SM)),
        "max_mem_clock_mhz": _try(lambda: nv.nvmlDeviceGetMaxClockInfo(handle, nv.NVML_CLOCK_MEM)),
        "slowdown_temperature_c": _try(
            lambda: nv.nvmlDeviceGetTemperatureThreshold(
                handle, nv.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN
            )
        ),
        "idle_temperature_c": _try(lambda: _temperature(nv, handle)),
        "idle_pstate": _try(lambda: nv.nvmlDeviceGetPerformanceState(handle)),
        # Read-only hints on clock control; locking itself is never attempted here.
        "supported_memory_clocks_mhz": mem_clocks,
        "supported_graphics_clock_count": _try(
            lambda: len(nv.nvmlDeviceGetSupportedGraphicsClocks(handle, mem_clocks[0]))
        ),
        "app_clock_changes_need_admin": _try(
            lambda: bool(
                nv.nvmlDeviceGetAPIRestriction(
                    handle, nv.NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS
                )
            )
        ),
    }


class NvmlSampler(threading.Thread):
    """Samples clocks, temperature, power and clock event reasons at a fixed interval."""

    def __init__(self, interval_s: float = 0.1):
        super().__init__(daemon=True)
        self.nv, self.handle = _nvml()
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop_event = threading.Event()

    @property
    def available(self) -> bool:
        return self.nv is not None

    def run(self) -> None:
        nv, h = self.nv, self.handle
        t0 = time.perf_counter()
        while not self._stop_event.is_set():
            self.samples.append(
                {
                    "t_s": round(time.perf_counter() - t0, 3),
                    "sm_mhz": _try(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
                    "mem_mhz": _try(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)),
                    "temp_c": _try(lambda: _temperature(nv, h)),
                    "power_w": _try(lambda: round(nv.nvmlDeviceGetPowerUsage(h) / 1000, 2)),
                    "pstate": _try(lambda: nv.nvmlDeviceGetPerformanceState(h)),
                    "reasons": _try(lambda: _event_reasons(nv, h)),
                }
            )
            self._stop_event.wait(self.interval_s)

    def stop(self) -> list[dict]:
        self._stop_event.set()
        self.join()
        return self.samples


# ------------------------------------------------------------------- probes


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def windows_settings() -> dict:
    """Power scheme, AC status and GPU scheduling mode, read without changing anything."""
    if platform.system() != "Windows":
        return {}
    import ctypes
    import winreg

    def reg(path: str, name: str):
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                return winreg.QueryValueEx(key, name)[0]
        except FileNotFoundError:
            return "not set"  # value absent: Windows uses its default

    class PowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_byte),
            ("BatteryFlag", ctypes.c_byte),
            ("BatteryLifePercent", ctypes.c_byte),
            ("SystemStatusFlag", ctypes.c_byte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    def ac_power():
        status = PowerStatus()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
        return {0: False, 1: True}.get(status.ACLineStatus)

    def active_scheme_guid():
        # Only the GUID: the plan name is localised and can be user-defined.
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, timeout=10)
        match = re.search(
            r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
            out.stdout.decode(errors="replace"),
        )
        return match.group(0).lower() if match else None

    schemes = r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes"
    return {
        "on_ac_power": _try(ac_power),
        "power_scheme_guid": _try(active_scheme_guid),
        "power_mode_overlay_ac": _try(lambda: reg(schemes, "ActiveOverlayAcPowerScheme")),
        "hw_gpu_scheduling_mode": _try(
            lambda: reg(r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers", "HwSchMode")
        ),
    }


def _cuda_init_error() -> str | None:
    try:
        torch.cuda.init()
    except Exception as exc:
        return error_text(exc)
    return None


def probe_environment(device: torch.device, cfg: dict) -> dict:
    cuda = torch.cuda.is_available()
    info = {
        "python": platform.python_version(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "wsl": _is_wsl(),
        },
        "packages": {
            name: _package_version(name)
            for name in ("torch", "numpy", "nvidia-ml-py", "triton", "triton-windows")
        },
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": cuda,
        "cuda_init_error": None if cuda else _cuda_init_error(),
        "arch_list": torch.cuda.get_arch_list() if cuda else [],
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        "windows": windows_settings(),
        "nvml": nvml_static(),  # also when CUDA is unusable: the driver version explains why
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        info["gpu"] = {
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "arch_in_wheel": f"sm_{props.major}{props.minor}" in info["arch_list"],
            "total_memory_gib": round(total / 2**30, 2),
            "free_memory_gib": round(free / 2**30, 2),
            "multiprocessors": props.multi_processor_count,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    return info


def probe_matmul(device: torch.device, cfg: dict) -> list[dict]:
    n = cfg["matmul_n"]
    modes = [("fp32", torch.float32, False), ("bf16", torch.bfloat16, False)]
    if device.type == "cuda":
        modes += [("tf32", torch.float32, True), ("fp16", torch.float16, False)]
    rows = []
    for name, dtype, use_tf32 in modes:
        row = {"mode": name, "n": n}
        try:
            with tf32(use_tf32):
                a = torch.randn(n, n, device=device, dtype=dtype)
                b = torch.randn(n, n, device=device, dtype=dtype)
                stats = summarize(time_ms(lambda: a @ b, device, 3, cfg["matmul_reps"]))  # noqa: B023
            row |= stats | {"tflops": round(2 * n**3 / (stats["median_ms"] / 1e3) / 1e12, 2)}
        except Exception as exc:
            row["error"] = error_text(exc)
        rows.append(row)
        _free(device)
    return rows


def attention_mask(mode: str, batch: int, length: int, device: torch.device):
    """Return (attn_mask, is_causal). Padding modes use left padding of random size."""
    if mode == "none":
        return None, False
    if mode == "causal":
        return None, True
    pad = torch.randint(0, length // 2 + 1, (batch,), device=device)
    pos = torch.arange(length, device=device)
    key_real = pos[None, :] >= pad[:, None]
    allowed = key_real[:, None, None, :].expand(batch, 1, length, length).clone()
    if mode == "causal_padding":
        allowed &= torch.ones(length, length, dtype=torch.bool, device=device).tril()
    # Each query may attend to itself, so no row is fully masked.
    allowed |= torch.eye(length, dtype=torch.bool, device=device)
    return allowed, False


def dispatched_backend(q, k, v, mask, causal) -> str:
    """Backend that SDPA picks by default, from the dispatcher PyTorch itself uses."""
    choice = torch._fused_sdp_choice(q, k, v, mask, 0.0, causal)
    return SDPBackend(choice).name.lower()


def _rel_err(value: torch.Tensor, reference: torch.Tensor) -> float:
    scale = reference.abs().max().item() or 1.0
    return (value.float() - reference).abs().max().item() / scale


def _reference(base, grad, mask, causal):
    """fp32 math-backend output and q, k, v gradients for one shape and mask."""
    q, k, v = (t.clone().requires_grad_() for t in base)
    with sdpa_kernel([SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)
    out.backward(grad)
    return out.detach(), [t.grad for t in (q, k, v)]


def probe_sdpa(device: torch.device, cfg: dict) -> list[dict]:
    rows = []
    batch, heads = cfg["sdpa_batch"], cfg["sdpa_heads"]
    for length in cfg["sdpa_lengths"]:
        for head_dim in cfg["sdpa_head_dims"]:
            base = torch.randn(3, batch, heads, length, head_dim, device=device)
            grad = torch.randn(batch, heads, length, head_dim, device=device)
            for mode in MASK_MODES:
                mask, causal = attention_mask(mode, batch, length, device)
                ref_out, ref_grads = _reference(base, grad, mask, causal)
                for dtype_name, dtype in DTYPES.items():
                    if device.type == "cpu" and dtype is torch.float16:
                        continue
                    q, k, v = (t.to(dtype).requires_grad_() for t in base)
                    g = grad.to(dtype)
                    common = {"length": length, "head_dim": head_dim, "mask": mode}
                    common["dtype"] = dtype_name

                    def call(q=q, k=k, v=v, m=mask, causal=causal):
                        return F.scaled_dot_product_attention(
                            q, k, v, attn_mask=m, is_causal=causal
                        )

                    default = _try(functools.partial(dispatched_backend, q, k, v, mask, causal))
                    for backend_name, backend in BACKENDS.items():
                        row = dict(common, backend=backend_name, default_choice=default)
                        try:
                            with sdpa_kernel([backend]):
                                out = call()
                                out.backward(g)
                                out_err = _rel_err(out, ref_out)
                                grad_err = max(
                                    _rel_err(t.grad, r)
                                    for t, r in zip((q, k, v), ref_grads, strict=True)
                                )
                                times = time_ms(
                                    lambda call=call, g=g: call().backward(g),
                                    device,
                                    2,
                                    cfg["sdpa_reps"],
                                )
                            row |= {
                                "ok": True,
                                "out_rel_err": round(out_err, 6),
                                "grad_rel_err": round(grad_err, 6),
                                "matches_reference": max(out_err, grad_err) <= REL_TOL[dtype],
                                "fwd_bwd": summarize(times),
                            }
                        except Exception as exc:
                            row |= {"ok": False, "error": error_text(exc)}
                        for t in (q, k, v):
                            t.grad = None
                        rows.append(row)
                    _free(device)
    return rows


class Block(nn.Module):
    """Pre-LN Transformer block with causal self-attention through SDPA."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attn.transpose(1, 2).reshape(b, n, d))
        return x + self.ffn(self.norm2(x))


class TinySASRec(nn.Module):
    """Synthetic-workload stand-in: SASRec-shaped encoder with tied item embeddings."""

    def __init__(self, num_items: int, dim: int, length: int, blocks: int = 2, heads: int = 2):
        super().__init__()
        self.items = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.pos = nn.Embedding(length, dim)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(blocks))
        self.norm = nn.LayerNorm(dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.items(seq) + self.pos.weight[: seq.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def loss_fn(name: str, model: TinySASRec, hidden, targets, negatives: int):
    emb = model.items.weight
    num_items = emb.shape[0] - 1
    if name == "full_ce":
        logits = hidden @ emb.T
        return F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
    pos = (hidden * emb[targets]).sum(-1)
    if name == "sampled_softmax":
        neg_ids = torch.randint(1, num_items + 1, (negatives,), device=hidden.device)
        logits = torch.cat([pos.unsqueeze(-1), hidden @ emb[neg_ids].T], dim=-1)
        labels = torch.zeros(logits.shape[:-1], dtype=torch.long, device=hidden.device)
        return F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten())
    if name == "bce":
        neg_ids = torch.randint(1, num_items + 1, targets.shape, device=hidden.device)
        neg = (hidden * emb[neg_ids]).sum(-1)
        return (F.softplus(-pos.float()) + F.softplus(neg.float())).mean()
    raise ValueError(name)


def train_step_fn(model, loss_name, seq, targets, negatives, precision, device):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_bf16 = precision == "bf16"

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            hidden = model(seq)
            loss = loss_fn(loss_name, model, hidden, targets, negatives)
        loss.backward()
        opt.step()

    return step


def _exceeds_free_memory(loss_name: str, logit_elements: int, device: torch.device) -> str | None:
    """Skip full cross-entropy when its logits alone cannot fit in free VRAM.

    Rough cost per logit: fp32 logits, fp32 log-softmax saved for backward, fp32 gradient.
    """
    if loss_name != "full_ce" or device.type != "cuda":
        return None
    needed = logit_elements * 12
    free, _ = torch.cuda.mem_get_info(device)
    if needed <= 0.9 * free:
        return None
    return f"estimated {needed / 2**30:.1f} GiB for logits exceeds {free / 2**30:.1f} GiB free"


def probe_model(device: torch.device, cfg: dict) -> dict:
    batch, length, dim = cfg["model_batch"], cfg["model_len"], cfg["model_dim"]
    precisions = ["fp32", "tf32", "bf16"] if device.type == "cuda" else ["fp32", "bf16"]
    train_rows, score_rows = [], []
    for num_items in cfg["model_items"]:
        seq = torch.randint(1, num_items + 1, (batch, length), device=device)
        targets = torch.randint(1, num_items + 1, (batch, length), device=device)
        for loss_name in ("full_ce", "sampled_softmax", "bce"):
            for precision in precisions:
                row = {"items": num_items, "loss": loss_name, "precision": precision}
                row |= {"batch": batch, "length": length, "dim": dim}
                too_big = _exceeds_free_memory(loss_name, batch * length * (num_items + 1), device)
                if too_big:
                    train_rows.append(row | {"ok": False, "skipped": too_big})
                    continue
                model = None
                try:
                    torch.manual_seed(0)
                    model = TinySASRec(num_items, dim, length).to(device)
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    with tf32(precision == "tf32"):
                        step = train_step_fn(
                            model,
                            loss_name,
                            seq,
                            targets,
                            cfg["model_negatives"],
                            precision,
                            device,
                        )
                        stats = summarize(time_ms(step, device, 3, cfg["model_reps"]))
                    row |= {"ok": True, "step": stats}
                    row["tokens_per_s"] = round(batch * length / (stats["median_ms"] / 1e3))
                    if device.type == "cuda":
                        peak = torch.cuda.max_memory_allocated(device)
                        total = torch.cuda.get_device_properties(device).total_memory
                        row["peak_memory_gib"] = round(peak / 2**30, 3)
                        # Windows may page into shared system memory instead of raising OOM.
                        row["spilled_to_system_memory"] = peak > total
                except torch.OutOfMemoryError:
                    row |= {"ok": False, "error": "out of memory"}
                except Exception as exc:
                    row |= {"ok": False, "error": error_text(exc)}
                del model
                _free(device)
                train_rows.append(row)
        score_rows.append(probe_scoring(num_items, device, cfg))
    return {"train_step": train_rows, "full_ranking_scoring": score_rows}


def probe_scoring(num_items: int, device: torch.device, cfg: dict) -> dict:
    """Full-catalogue scoring plus top-10, chunked over users, fp32."""
    users, chunk, dim = cfg["score_users"], cfg["score_chunk"], cfg["model_dim"]
    row = {"items": num_items, "users": users, "chunk": chunk}
    try:
        reps = torch.randn(users, dim, device=device)
        emb = torch.randn(num_items + 1, dim, device=device)

        def score_all():
            for start in range(0, users, chunk):
                scores = reps[start : start + chunk] @ emb.T
                scores[:, 0] = float("-inf")
                scores.topk(10, dim=1)

        with tf32(False):
            stats = summarize(time_ms(score_all, device, 1, 5))
        row |= {"ok": True, "per_1k_users_ms": round(stats["median_ms"] * 1000 / users, 3)}
    except Exception as exc:
        row |= {"ok": False, "error": error_text(exc)}
    _free(device)
    return row


def probe_thermal(device: torch.device, cfg: dict) -> dict:
    """Sustained bf16 matmul load with NVML sampling; shows warm-up time and throttling."""
    n, seconds = cfg["thermal_matmul_n"], cfg["thermal_seconds"]
    dtype = torch.bfloat16
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(10):  # library handles and kernel loading stay out of the first window
        a @ b
    _sync(device)
    sampler = NvmlSampler()
    if sampler.available:
        sampler.start()
    window_s = max(seconds / 24, 0.25)
    windows, t_end = [], time.perf_counter() + seconds
    try:
        while time.perf_counter() < t_end:
            t0, count = time.perf_counter(), 0
            while time.perf_counter() - t0 < window_s:
                a @ b
                count += 1
                if count % 8 == 0:
                    _sync(device)
            _sync(device)
            elapsed = time.perf_counter() - t0
            windows.append(round(2 * n**3 * count / elapsed / 1e12, 3))
    finally:
        samples = sampler.stop() if sampler.available else []
    early = statistics.median(windows[1:4] or windows)
    result = {
        "seconds": seconds,
        "matmul_n": n,
        "tflops_per_window": windows,
        "last_over_max_window": round(windows[-1] / max(windows), 3),
        "last_over_early_windows": round(windows[-1] / early, 3) if early else None,
        "nvml_available": sampler.available,
    }
    if samples:
        result["summary"] = _thermal_summary(sampler.nv, samples)
        result["samples"] = samples
    return result


def _thermal_summary(nv, samples: list[dict]) -> dict:
    def series(key):
        return [s[key] for s in samples if isinstance(s[key], int | float)]

    def describe(values):
        if not values:
            return None
        q = statistics.quantiles(values, n=20) if len(values) >= 2 else [values[0]] * 19
        return {
            "min": min(values),
            "p5": round(q[0], 1),
            "median": statistics.median(values),
            "p95": round(q[-1], 1),
            "max": max(values),
        }

    reasons = series("reasons")
    shares = {}
    for label, attr in REASON_BITS.items():
        bit = getattr(nv, attr, None)
        if bit is not None and reasons:
            shares[label] = round(sum(1 for r in reasons if r & bit) / len(reasons), 3)
    temps = series("temp_c")
    return {
        "samples": len(samples),
        "sm_mhz": describe(series("sm_mhz")),
        "mem_mhz": describe(series("mem_mhz")),
        "temp_c": describe(temps) | {"start": temps[0], "end": temps[-1]} if temps else None,
        "power_w": describe(series("power_w")),
        "share_of_samples_with_reason": shares,
    }


def probe_compile(device: torch.device, cfg: dict) -> dict:
    """Run torch.compile in a child process so a hang or crash cannot stop the check."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--compile-worker",
        "--device",
        device.type,
    ]
    if cfg["quick"]:
        cmd.append("--quick")
    timeout = cfg["compile_timeout_s"]
    # Output goes to files: pipes can block when a timed-out process tree is killed.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out_path, err_path = Path(tmp, "stdout.txt"), Path(tmp, "stderr.txt")
        with out_path.open("wb") as out, err_path.open("wb") as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err, cwd=REPO_ROOT)
            try:
                returncode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                return {"ok": False, "error": f"timeout after {timeout} s"}
        stdout = out_path.read_text(encoding="utf-8", errors="replace")
        stderr = err_path.read_text(encoding="utf-8", errors="replace")
    for line in stdout.splitlines():
        if line.startswith("COMPILE_RESULT "):
            return json.loads(line.removeprefix("COMPILE_RESULT "))
    tail = [line for line in stderr.strip().splitlines() if line.strip()][-15:]
    return {"ok": False, "returncode": returncode, "stderr_tail": tail}


def _kill_tree(proc: subprocess.Popen) -> None:
    if platform.system() == "Windows":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=30)


def compile_worker(device: torch.device, cfg: dict) -> int:
    torch.manual_seed(0)
    batch, length, dim = cfg["model_batch"], cfg["model_len"], cfg["model_dim"]
    num_items = cfg["model_items"][0]
    seq = torch.randint(1, num_items + 1, (batch, length), device=device)
    targets = torch.randint(1, num_items + 1, (batch, length), device=device)
    precision = "bf16" if device.type == "cuda" else "fp32"
    result = {"precision": precision, "items": num_items, "batch": batch, "length": length}
    try:
        model = TinySASRec(num_items, dim, length).to(device)
        eager = train_step_fn(model, "full_ce", seq, targets, 0, precision, device)
        eager_stats = summarize(time_ms(eager, device, 3, cfg["model_reps"]))
        compiled = train_step_fn(
            torch.compile(model), "full_ce", seq, targets, 0, precision, device
        )
        t0 = time.perf_counter()
        compiled()
        _sync(device)
        compile_s = time.perf_counter() - t0
        compiled_stats = summarize(time_ms(compiled, device, 3, cfg["model_reps"]))
        result |= {
            "ok": True,
            "compile_s": round(compile_s, 1),
            "eager": eager_stats,
            "compiled": compiled_stats,
            "speedup": round(eager_stats["median_ms"] / compiled_stats["median_ms"], 3),
        }
    except Exception as exc:
        result |= {"ok": False, "error": error_text(exc)}
    print("COMPILE_RESULT " + json.dumps(result), flush=True)
    return 0


# ------------------------------------------------------------------ report


def _section(report: dict, name: str, kind: type):
    value = report.get(name)
    return value if isinstance(value, kind) else None


def build_summary(report: dict) -> str:
    env = _section(report, "environment", dict) or {}
    nvml = env.get("nvml") or {}
    lines = [
        f"seqrec-lab environment check ({report['mode']}, {report['created_utc']})",
        f"python {env.get('python')} | torch {(env.get('packages') or {}).get('torch')}"
        f" | cuda {env.get('torch_cuda')} | cudnn {env.get('cudnn')}"
        f" | driver {nvml.get('driver_version')}",
    ]
    gpu = env.get("gpu")
    if gpu:
        lines.append(
            f"gpu {gpu['name']} | capability {gpu['capability']}"
            f" | in wheel arch list: {gpu['arch_in_wheel']} | {gpu['total_memory_gib']} GiB"
            f" | bf16 {gpu['bf16_supported']}"
        )
    else:
        lines.append(f"device {report['device']}; CUDA error: {env.get('cuda_init_error')}")
    if env.get("windows"):
        lines.append(f"windows settings {env['windows']}")

    matmul = _section(report, "matmul", list)
    if matmul:
        rates = ", ".join(f"{r['mode']} {r.get('tflops', 'error')}" for r in matmul)
        lines.append(f"\nmatmul TFLOPS: {rates}")

    sdpa = _section(report, "sdpa", list)
    if sdpa:
        lines.append("\nSDPA backends (ok / tried over dtypes, masks and shapes):")
        for backend in BACKENDS:
            rows = [r for r in sdpa if r["backend"] == backend]
            ok = [r for r in rows if r.get("ok")]
            bad = [r for r in ok if not r.get("matches_reference")]
            lines.append(f"  {backend:9s} {len(ok)}/{len(rows)}  outside tolerance: {len(bad)}")
        first = (sdpa[0]["length"], sdpa[0]["head_dim"])
        lines.append(f"Default choice by mask and dtype (length {first[0]}, head dim {first[1]}):")
        for r in sdpa:
            if (r["length"], r["head_dim"]) == first and r["backend"] == "math":
                lines.append(f"  {r['mask']:15s} {r['dtype']}: {r['default_choice']}")

    model = _section(report, "model", dict)
    if model and model.get("train_step"):
        lines.append("\nSynthetic SASRec train step (ms median | tokens/s | peak GiB):")
        for r in model["train_step"]:
            head = f"  items {r['items']:>7} {r['loss']:15s} {r['precision']:4s}"
            if r.get("ok"):
                spilled = "  SPILLED" if r.get("spilled_to_system_memory") else ""
                lines.append(
                    f"{head} {r['step']['median_ms']:9.2f} | {r['tokens_per_s']:>9} |"
                    f" {r.get('peak_memory_gib', '-')}{spilled}"
                )
            else:
                lines.append(f"{head} {r.get('skipped') or r.get('error')}")
        scoring = ", ".join(
            f"{r['items']} items {r.get('per_1k_users_ms', 'error')}"
            for r in model["full_ranking_scoring"]
        )
        lines.append(f"Full-ranking scoring, ms per 1k users: {scoring}")

    comp = _section(report, "compile", dict)
    if comp and comp.get("ok"):
        lines.append(
            f"\ntorch.compile: ok, compile {comp['compile_s']} s, speedup {comp['speedup']}x"
        )
    elif comp:
        detail = comp.get("error") or comp.get("skipped") or (comp.get("stderr_tail") or [""])[-1]
        lines.append(f"\ntorch.compile: not ok ({detail})")

    thermal = _section(report, "thermal", dict)
    if thermal and thermal.get("tflops_per_window"):
        lines.append(
            f"\nThermal: {thermal['seconds']} s bf16 load, TFLOPS per window"
            f" {thermal['tflops_per_window']}"
        )
        lines.append(
            f"  last/max {thermal['last_over_max_window']}"
            f" | last/early {thermal['last_over_early_windows']}"
        )
        summary = thermal.get("summary")
        if summary:
            lines.append(f"  SM MHz {summary['sm_mhz']}")
            lines.append(f"  temp C {summary['temp_c']}")
            lines.append(f"  power W {summary['power_w']}")
            shares = summary["share_of_samples_with_reason"]
            lines.append(f"  share of samples per clock event reason {shares}")
        else:
            lines.append("  NVML not available: no clock or temperature data")

    failed = [
        k for k in report["durations_s"] if isinstance(report.get(k), dict) and "error" in report[k]
    ]
    if failed:
        lines.append(f"\nProbes that crashed: {failed}")
    lines.append("\nProbe durations (s): " + json.dumps(report["durations_s"]))
    return "\n".join(lines) + "\n"


def write_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    """Scrub, then write JSON and text summary. Nothing unscrubbed reaches disk."""
    roots, tokens = sensitive_roots(), sensitive_tokens()
    summary = scrub_text(build_summary(report), roots, tokens)
    payload = json.dumps(scrub(report, roots, tokens), indent=1, allow_nan=False)
    gpu = (report["environment"].get("gpu") or {}).get("name", report["device"])
    slug = re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-")
    stamp = report["created_utc"].replace("-", "").replace(":", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"check_env_{slug}_{stamp}.json"
    text_path = json_path.with_suffix(".txt")
    json_path.write_text(payload + "\n", encoding="utf-8")
    text_path.write_text(summary, encoding="utf-8")
    return json_path, text_path


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="tiny shapes, a few seconds")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--thermal-seconds", type=float, default=None)
    parser.add_argument("--compile-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = QUICK if args.quick else FULL
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.compile_worker:
        return compile_worker(device, cfg)
    if args.thermal_seconds is not None:
        cfg = dict(cfg, thermal_seconds=args.thermal_seconds)
    # A full run on CPU would take hours; without a usable GPU only the environment is recorded.
    gpu_missing = not args.quick and device.type == "cpu" and args.device is None

    torch.manual_seed(0)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "quick" if args.quick else "full",
        "device": device.type,
        "config": cfg,
        "durations_s": {},
    }
    probes = [
        ("environment", probe_environment),
        ("thermal", probe_thermal),
        ("matmul", probe_matmul),
        ("sdpa", probe_sdpa),
        ("model", probe_model),
        ("compile", probe_compile),
    ]
    for name, probe in probes:
        if name != "environment" and gpu_missing:
            break
        if name == "compile" and args.skip_compile:
            report[name] = {"ok": False, "skipped": "--skip-compile"}
            continue
        print(f"[check_env] {name} ...", flush=True)
        t0 = time.perf_counter()
        try:
            report[name] = probe(device, cfg)
        except Exception:
            report[name] = {"error": traceback.format_exc(limit=4)}
        report["durations_s"][name] = round(time.perf_counter() - t0, 1)
        _free(device)

    json_path, text_path = write_report(report, args.out)
    print(text_path.read_text(encoding="utf-8"))
    for path in (json_path, text_path):
        try:
            shown = path.resolve().relative_to(REPO_ROOT)
        except ValueError:
            shown = path.name
        print(f"[check_env] wrote {shown}")
    if gpu_missing:
        print(
            "[check_env] no usable CUDA GPU: only the environment was checked (see cuda_init_error)"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
