import time
from types import SimpleNamespace

from seqrec import telemetry


class FakeNvml:
    """The NVML calls the sampler makes, with device memory rising by 1 GiB per sample."""

    NVML_CLOCK_SM, NVML_CLOCK_MEM, NVML_TEMPERATURE_GPU = 0, 1, 0

    def __init__(self):
        self.used = 0

    def nvmlDeviceGetClockInfo(self, handle, clock):
        return 2000

    def nvmlDeviceGetTemperatureV(self, handle, sensor):
        return 60

    def nvmlDeviceGetPowerUsage(self, handle):
        return 80_000

    def nvmlDeviceGetPerformanceState(self, handle):
        return 0

    def nvmlDeviceGetCurrentClocksEventReasons(self, handle):
        return 0

    def nvmlDeviceGetMemoryInfo(self, handle):
        self.used += 2**30
        return SimpleNamespace(used=self.used, total=16 * 2**30)


def test_the_sampler_records_device_memory_and_its_peak(monkeypatch):
    nv = FakeNvml()
    monkeypatch.setattr(telemetry, "_nvml", lambda: (nv, "handle"))
    sampler = telemetry.NvmlSampler(interval_s=0.01)
    sampler.start()
    time.sleep(0.1)
    samples = sampler.stop()
    assert [s["mem_used_mib"] for s in samples[:3]] == [1024, 2048, 3072]
    summary = telemetry.summarize_samples(nv, samples)
    assert summary["mem_used_mib"]["max"] == 1024 * len(samples)


def test_samples_from_before_memory_sampling_still_summarize():
    """nvml.jsonl collects every session of a run, so a resumed run mixes old and new lines."""
    old = {"t_s": 0.0, "sm_mhz": 2000, "mem_mhz": 1, "temp_c": 60, "power_w": 80.0,
           "pstate": 0, "reasons": 0}  # fmt: skip
    summary = telemetry.summarize_samples(FakeNvml(), [old, old | {"mem_used_mib": 900}])
    assert summary["mem_used_mib"]["max"] == 900 and summary["samples"] == 2
