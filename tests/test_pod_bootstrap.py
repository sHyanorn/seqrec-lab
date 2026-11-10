"""The first steps of scripts/cloud/pod_bootstrap.sh, up to the started watchdog, against
fakes. The script stops later, at checks that only pass on the rented machine."""

from __future__ import annotations

import fcntl
import os
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

CLOUD = Path(__file__).resolve().parents[1] / "scripts" / "cloud"

pytestmark = pytest.mark.skipif(shutil.which("flock") is None, reason="needs bash and flock")


def fake(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


# Everything the two scripts call before the bootstrap's later checks. The fixture's PATH holds
# these and the fakes only, so the tests see the same machine wherever they run.
TOOLS = (
    "bash env flock pgrep pkill timeout sleep nohup mkdir chmod mv cp rm tr sed date head cat tar"
)


@pytest.fixture
def machine(tmp_path):
    """Fakes: /workspace is a volume, `curl -o FILE` delivers the repository's pod_start.sh,
    and there is no nvidia-smi."""
    bin_dir, home = tmp_path / "bin", tmp_path / "home"
    bin_dir.mkdir()
    for tool in TOOLS.split():
        (bin_dir / tool).symlink_to(shutil.which(tool))
    fake(bin_dir, "mountpoint", "true")
    fake(bin_dir, "runpodctl", "true")
    deliver = f'[ "$1" = -o ] && cp {CLOUD}/pod_start.sh "$2"'
    fake(bin_dir, "curl", f"while [ $# -gt 1 ]; do {deliver}; shift; done")
    env = {
        "PATH": str(bin_dir), "SEQREC_HOME": str(home), "RUNNER_DIR": str(tmp_path / "runner"),
        "POD_START_LOCK": str(tmp_path / "lock"), "CHECK_SECONDS": "1", "RUNPOD_POD_ID": "pod1",
    }  # fmt: skip
    watchdog = f"^bash {home}/pod_start\\.sh$"

    def bootstrap(**more):
        return subprocess.run(
            [shutil.which("bash"), str(CLOUD / "pod_bootstrap.sh")], env=env | more,
            capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
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
    with (tmp / "lock").open("w") as holder:  # held here: no process is left behind
        fcntl.flock(holder, fcntl.LOCK_EX)
        result = bootstrap()
    assert result.returncode == 1
    assert "Watchdog started" not in result.stdout
    assert "not by the watchdog" in result.stderr


def test_a_lock_that_cannot_be_checked_is_not_reported_as_a_started_watchdog(machine):
    tmp, bootstrap, watchdog = machine
    result = bootstrap(POD_START_LOCK=str(tmp / "missing" / "lock"))
    assert result.returncode == 1
    assert "Watchdog started" not in result.stdout
    assert "cannot be checked" in result.stderr


def test_a_machine_that_could_not_stop_itself_is_not_reported_as_watched(machine):
    tmp, bootstrap, watchdog = machine
    try:
        known = b"RUNPOD_POD_ID=" in Path("/proc/1/environ").read_bytes()
    except OSError:
        known = False
    if known:
        pytest.skip("this machine's id is in its init environment")
    result = bootstrap(RUNPOD_POD_ID="")
    assert result.returncode == 1
    assert "Watchdog started" not in result.stdout
    assert "RUNPOD_POD_ID is not set" in result.stderr


def test_the_start_loop_must_run_the_copy_this_script_installed(machine):
    """A start loop for another path would start no watchdog after the next machine start."""
    tmp, bootstrap, watchdog = machine
    fake(tmp / "bin", "nvidia-smi", "echo 580.00")
    loop = "while true; do [ -x /workspace/seqrec/pod_start.sh ] && true; sleep 60; done"
    decoy = subprocess.Popen(["bash", "-c", loop], start_new_session=True)
    try:
        result = bootstrap()
    finally:
        os.killpg(decoy.pid, signal.SIGKILL)
    assert result.returncode == 1
    assert "Watchdog started" in result.stdout
    assert "start command is not set as documented" in result.stderr
