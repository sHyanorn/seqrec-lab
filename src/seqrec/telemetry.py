"""GPU telemetry through NVML: static facts, a background sampler and a warm-up load.

Every query is wrapped so that a missing NVML or an unsupported call is recorded as
data instead of raising.
"""

from __future__ import annotations

import statistics
import threading
import time

import torch


def error_text(exc: BaseException) -> str:
    """First and last non-empty lines; compiler and driver errors put the cause at the end."""
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    detail = " | ".join(dict.fromkeys([lines[0], lines[-1]])) if lines else ""
    return f"{type(exc).__name__}: {detail}"[:400]


def try_call(fn):
    try:
        value = fn()
    except Exception as exc:  # telemetry records failures instead of raising
        return {"error": error_text(exc)}
    return value.decode() if isinstance(value, bytes) else value


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
    mem_clocks = try_call(lambda: nv.nvmlDeviceGetSupportedMemoryClocks(handle))
    return {
        "available": True,
        "driver_version": try_call(nv.nvmlSystemGetDriverVersion),
        "cuda_driver_version": try_call(nv.nvmlSystemGetCudaDriverVersion),
        "name": try_call(lambda: nv.nvmlDeviceGetName(handle)),
        "power_limit_w": try_call(lambda: nv.nvmlDeviceGetPowerManagementLimit(handle) / 1000),
        "power_limit_default_w": try_call(
            lambda: nv.nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000
        ),
        "enforced_power_limit_w": try_call(
            lambda: nv.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
        ),
        "max_sm_clock_mhz": try_call(
            lambda: nv.nvmlDeviceGetMaxClockInfo(handle, nv.NVML_CLOCK_SM)
        ),
        "max_mem_clock_mhz": try_call(
            lambda: nv.nvmlDeviceGetMaxClockInfo(handle, nv.NVML_CLOCK_MEM)
        ),
        "slowdown_temperature_c": try_call(
            lambda: nv.nvmlDeviceGetTemperatureThreshold(
                handle, nv.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN
            )
        ),
        "idle_temperature_c": try_call(lambda: _temperature(nv, handle)),
        "idle_pstate": try_call(lambda: nv.nvmlDeviceGetPerformanceState(handle)),
        # Read-only hints on clock control; locking itself is never attempted here.
        "supported_memory_clocks_mhz": mem_clocks,
        "supported_graphics_clock_count": try_call(
            lambda: len(nv.nvmlDeviceGetSupportedGraphicsClocks(handle, mem_clocks[0]))
        ),
        "app_clock_changes_need_admin": try_call(
            lambda: bool(
                nv.nvmlDeviceGetAPIRestriction(
                    handle, nv.NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS
                )
            )
        ),
    }


class NvmlSampler(threading.Thread):
    """Samples clocks, temperature, power, clock event reasons and device memory in use at a
    fixed interval."""

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
                    "sm_mhz": try_call(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
                    "mem_mhz": try_call(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)),
                    "temp_c": try_call(lambda: _temperature(nv, h)),
                    "power_w": try_call(lambda: round(nv.nvmlDeviceGetPowerUsage(h) / 1000, 2)),
                    "pstate": try_call(lambda: nv.nvmlDeviceGetPerformanceState(h)),
                    "reasons": try_call(lambda: _event_reasons(nv, h)),
                    # The whole device, other processes and the display included
                    "mem_used_mib": try_call(
                        lambda: round(nv.nvmlDeviceGetMemoryInfo(h).used / 2**20)
                    ),
                }
            )
            self._stop_event.wait(self.interval_s)

    def stop(self) -> list[dict]:
        self._stop_event.set()
        self.join()
        return self.samples


def summarize_samples(nv, samples: list[dict]) -> dict:
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
        "mem_used_mib": describe(series("mem_used_mib")),
        "share_of_samples_with_reason": shares,
    }


def warmup_gpu(seconds: float, device: torch.device, n: int = 8192) -> None:
    """Continuous bf16 matmul load so that timed work starts in the thermal steady state."""
    if device.type != "cuda" or seconds <= 0:
        return
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        for _ in range(8):
            a @ a
        torch.cuda.synchronize(device)
