"""The first steps of scripts/cloud/pod_bootstrap.sh, up to the registration token, against
fakes."""

from __future__ import annotations

import fcntl
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

CLOUD = Path(__file__).resolve().parents[1] / "scripts" / "cloud"

pytestmark = pytest.mark.skipif(shutil.which("flock") is None, reason="needs bash and flock")


def fake(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


# Everything the script calls up to the registration token. The fixture's PATH holds these and
# the fakes only, so the tests see the same machine wherever they run.
TOOLS = "bash env flock pgrep timeout sleep mkdir tr sed curl tar"


@pytest.fixture
def machine(tmp_path):
    """Fakes: /workspace is a volume, and `watchdog()` starts a process that holds the lock
    under the watchdog's command line, as the container start command does."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in TOOLS.split():
        (bin_dir / tool).symlink_to(shutil.which(tool))
    fake(bin_dir, "mountpoint", "true")
    fake(tmp_path, "pod_start.sh", f"exec 9> {tmp_path}/lock; flock 9; sleep 30 9>&-")
    env = {
        "PATH": str(bin_dir), "SEQREC_HOME": str(tmp_path / "home"),
        "RUNNER_DIR": str(tmp_path / "runner"), "POD_START_LOCK": str(tmp_path / "lock"),
        "POD_START": str(tmp_path / "pod_start.sh"), "RUNPOD_POD_ID": "pod1",
        "RUNPOD_API_KEY": "k3y", "RUNNER_TOKEN": "not a token",
    }  # fmt: skip
    procs = []

    def bootstrap(**more):
        return subprocess.run(
            [shutil.which("bash"), str(CLOUD / "pod_bootstrap.sh")], env=env | more,
            capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
        )  # fmt: skip

    def watchdog():
        command = ["bash", str(tmp_path / "pod_start.sh")]
        procs.append(subprocess.Popen(command, env=env, start_new_session=True))
        time.sleep(0.3)

    yield tmp_path, bootstrap, watchdog
    for p in procs:
        os.killpg(p.pid, signal.SIGKILL)  # with its sleep: no process is left behind
        p.wait()


def test_a_watched_machine_gets_as_far_as_the_registration_token(machine):
    """The first registration runs on whatever Pod is to be had; the job checks the GPU."""
    tmp, bootstrap, watchdog = machine
    watchdog()
    result = bootstrap()
    assert result.returncode == 1
    assert "Watchdog running" in result.stdout
    assert "does not look like a registration token" in result.stderr


def test_a_machine_without_a_running_watchdog_is_refused(machine):
    """The start command runs the watchdog; this script starts none."""
    tmp, bootstrap, watchdog = machine
    result = bootstrap()
    assert result.returncode == 1
    assert "Watchdog running" not in result.stdout
    assert "the watchdog is not running" in result.stderr


def test_a_lock_held_by_something_else_is_not_reported_as_a_running_watchdog(machine):
    tmp, bootstrap, watchdog = machine
    with (tmp / "lock").open("w") as holder:  # held here: no process is left behind
        fcntl.flock(holder, fcntl.LOCK_EX)
        result = bootstrap()
    assert result.returncode == 1
    assert "Watchdog running" not in result.stdout
    assert "the watchdog is not running" in result.stderr


@pytest.mark.parametrize("missing", ["RUNPOD_POD_ID", "RUNPOD_API_KEY"])
def test_a_machine_that_could_not_delete_itself_is_not_reported_as_watched(machine, missing):
    tmp, bootstrap, watchdog = machine
    try:
        known = f"{missing}=".encode() in Path("/proc/1/environ").read_bytes()
    except OSError:
        known = False
    if known:
        pytest.skip("this machine's init environment has the variable")
    watchdog()
    result = bootstrap(**{missing: ""})
    assert result.returncode == 1
    assert "Watchdog running" not in result.stdout
    assert f"{missing} is not set" in result.stderr
