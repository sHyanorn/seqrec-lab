"""The first steps of scripts/cloud/pod_bootstrap.sh, up to the started watchdog, against
fakes. The script stops later, at checks that only pass on the rented machine."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

CLOUD = Path(__file__).resolve().parents[1] / "scripts" / "cloud"

pytestmark = pytest.mark.skipif(shutil.which("flock") is None, reason="needs bash and flock")


def fake(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def machine(tmp_path):
    """Fakes: /workspace is a volume, `curl -o FILE` delivers the repository's pod_start.sh,
    and there is no nvidia-smi."""
    bin_dir, home = tmp_path / "bin", tmp_path / "home"
    bin_dir.mkdir()
    fake(bin_dir, "mountpoint", "true")
    fake(bin_dir, "runpodctl", "true")
    deliver = f'[ "$1" = -o ] && cp {CLOUD}/pod_start.sh "$2"'
    fake(bin_dir, "curl", f"while [ $# -gt 1 ]; do {deliver}; shift; done")
    assert shutil.which("nvidia-smi") is None
    env = os.environ | {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "SEQREC_HOME": str(home), "RUNNER_DIR": str(tmp_path / "runner"),
        "POD_START_LOCK": str(tmp_path / "lock"), "CHECK_SECONDS": "1", "RUNPOD_POD_ID": "pod1",
    }  # fmt: skip
    watchdog = f"^bash {home}/pod_start\\.sh$"

    def bootstrap(**more):
        return subprocess.run(
            ["bash", str(CLOUD / "pod_bootstrap.sh")], env=env | more, capture_output=True,
            text=True, timeout=60, stdin=subprocess.DEVNULL,
        )  # fmt: skip

    yield tmp_path, bootstrap, watchdog
    subprocess.run(["pkill", "-f", watchdog], check=False)


def running(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def test_the_watchdog_runs_before_anything_else_can_fail(machine):
    tmp, bootstrap, watchdog = machine
    result = bootstrap()
    assert result.returncode == 1
    assert "Watchdog started" in result.stdout
    assert "nvidia-smi is missing" in result.stderr  # a later check; the watchdog is up by then
    assert running(watchdog)
    held = subprocess.run(["flock", "-n", "-E", "75", str(tmp / "lock"), "true"])
    assert held.returncode == 75


def test_a_lock_held_by_something_else_is_not_reported_as_a_started_watchdog(machine):
    tmp, bootstrap, watchdog = machine
    holder = subprocess.Popen(["flock", str(tmp / "lock"), "sleep", "30"])
    try:
        result = bootstrap()
    finally:
        holder.kill()
    assert result.returncode == 1
    assert "Watchdog started" not in result.stdout
    assert "Stop this machine by hand" in result.stderr


def test_a_lock_that_cannot_be_checked_is_not_reported_as_a_started_watchdog(machine):
    tmp, bootstrap, watchdog = machine
    result = bootstrap(POD_START_LOCK=str(tmp / "missing" / "lock"))
    assert result.returncode == 1
    assert "Watchdog started" not in result.stdout
    assert "Stop this machine by hand" in result.stderr


def test_a_machine_that_could_not_stop_itself_is_not_reported_as_watched(machine):
    tmp, bootstrap, watchdog = machine
    result = bootstrap(RUNPOD_POD_ID="")
    assert result.returncode == 1
    assert "Watchdog started" not in result.stdout
    assert "Stop this machine by hand" in result.stderr
