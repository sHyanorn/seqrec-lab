"""The machine watchdog (scripts/cloud/pod_start.sh), run against fakes with 0.1 s checks."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cloud" / "pod_start.sh"

pytestmark = pytest.mark.skipif(shutil.which("flock") is None, reason="needs bash and flock")


def fake(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def machine(tmp_path):
    """Fakes: `runpodctl` records its arguments in `stopped`; a job is running while the file
    `job` exists, as the process whose id the file holds; the runner is registered and its
    run.sh records its environment."""
    bin_dir, home, runner = tmp_path / "bin", tmp_path / "home", tmp_path / "runner"
    for d in (bin_dir, home, runner / "bin"):
        d.mkdir(parents=True)
    fake(bin_dir, "runpodctl", f'echo "$@" >> {tmp_path}/stopped; exit "${{STOP_EXIT:-0}}"')
    fake(bin_dir, "curl", f'{{ echo "$@"; cat; }} >> {tmp_path}/curled')
    fake(bin_dir, "pgrep", f'case "$*" in *Worker*) cat {tmp_path}/job 2> /dev/null ;; '
                           f'*Listener*) test -e {tmp_path}/listening ;; esac')  # fmt: skip
    (runner / ".runner").write_text("{}", encoding="utf-8")
    fake(runner / "bin", "installdependencies.sh", "true")
    fake(runner, "run.sh", f"env > {tmp_path}/runner-env; touch {tmp_path}/listening")
    env = os.environ | {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "SEQREC_HOME": str(home), "RUNNER_DIR": str(runner), "CHECK_SECONDS": "0.1",
        "IDLE_MINUTES": "10", "RUNPOD_POD_ID": "pod1", "RUNPOD_API_KEY": "k3y",
        "STOP_REQUEST": str(tmp_path / "stop-request"), "POD_START_LOCK": str(tmp_path / "lock"),
        "STOP_WAIT_SECONDS": "0",
    }  # fmt: skip
    procs = []

    def start(**more):
        proc = subprocess.Popen(
            ["bash", str(SCRIPT)], env=env | more, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )  # fmt: skip
        procs.append(proc)
        return proc

    yield tmp_path, start
    for p in procs:
        p.kill()


def wait_for(path: Path, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if path.exists():
            return True
        time.sleep(0.02)
    return False


def test_the_runner_starts_with_uv_on_path_and_without_the_machine_key(machine):
    tmp, start = machine
    start()
    assert wait_for(tmp / "runner-env", 3)
    time.sleep(0.1)
    env = (tmp / "runner-env").read_text(encoding="utf-8")
    assert f"PATH={tmp}/home/bin:" in env
    assert "RUNNER_ALLOW_RUNASROOT=1" in env
    assert "k3y" not in env


def test_an_idle_machine_stops_after_the_idle_limit_and_not_before(machine):
    tmp, start = machine
    start()
    assert not wait_for(tmp / "stopped", 0.6)
    assert wait_for(tmp / "stopped", 3)
    assert (tmp / "stopped").read_text(encoding="utf-8").splitlines()[0] == "pod stop pod1"


def test_a_running_job_is_never_interrupted_and_a_stop_request_waits_for_it(machine):
    tmp, start = machine
    (tmp / "job").write_text("11", encoding="utf-8")
    start()
    time.sleep(0.5)
    (tmp / "stop-request").touch()
    assert not wait_for(tmp / "stopped", 1.5)  # past the idle limit, still running
    (tmp / "job").unlink()
    assert wait_for(tmp / "stopped", 1)


def test_a_stop_request_is_honoured_after_two_checks_without_a_job(machine):
    tmp, start = machine
    (tmp / "stop-request").touch()
    start()
    assert wait_for(tmp / "stopped", 0.6)  # well before the idle limit


@pytest.mark.parametrize("stop_exit", ["0", "1"])
def test_every_stop_uses_both_ways_and_is_repeated_while_the_machine_runs(machine, stop_exit):
    tmp, start = machine
    start(STOP_EXIT=stop_exit, IDLE_MINUTES="2")
    assert wait_for(tmp / "curled", 3)
    time.sleep(0.5)
    assert len((tmp / "stopped").read_text(encoding="utf-8").splitlines()) >= 2
    curled = (tmp / "curled").read_text(encoding="utf-8")
    assert "pods/pod1/action" in curled.splitlines()[0]
    assert "k3y" not in curled.splitlines()[0]  # the key goes in on stdin
    assert "Authorization: Bearer k3y" in curled


def test_one_job_running_past_the_busy_limit_stops_the_machine(machine):
    tmp, start = machine
    (tmp / "job").write_text("11", encoding="utf-8")
    start(BUSY_MINUTES="5")
    assert wait_for(tmp / "stopped", 3)


def test_jobs_that_follow_each_other_are_not_counted_as_one(machine):
    tmp, start = machine
    (tmp / "job").write_text("11", encoding="utf-8")
    start(BUSY_MINUTES="8")
    for pid in ("12", "13", "14"):
        time.sleep(0.4)
        (tmp / "job").write_text(pid, encoding="utf-8")
    assert not wait_for(tmp / "stopped", 0.4)


def test_a_second_instance_exits_at_once(machine):
    tmp, start = machine
    start()
    time.sleep(0.3)
    assert start().wait(timeout=3) == 0


def test_a_new_instance_takes_over_as_soon_as_the_old_one_is_killed(machine):
    """The lock must die with the watchdog, not live on in the sleep it was waiting for."""
    tmp, start = machine
    old = start(CHECK_SECONDS="5")
    time.sleep(0.5)
    old.terminate()
    old.wait(timeout=3)
    new = start(CHECK_SECONDS="5")
    time.sleep(0.5)
    assert new.poll() is None


def test_a_stop_in_progress_does_not_keep_the_lock_after_the_watchdog_dies(machine):
    tmp, start = machine
    old = start(IDLE_MINUTES="1", STOP_WAIT_SECONDS="5")
    assert wait_for(tmp / "stopped", 3)
    time.sleep(0.2)  # now in the wait between the two stop attempts
    old.kill()
    old.wait(timeout=3)
    new = start(CHECK_SECONDS="5")
    time.sleep(0.5)
    assert new.poll() is None
