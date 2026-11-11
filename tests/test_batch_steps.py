"""The batch workflow's shell steps (scripts/cloud/batch_steps.sh), run against fakes."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "cloud" / "batch_steps.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def steps(*args, cwd=REPO, env=None, path_prefix=None):
    job = (
        "GITHUB_",
        "GIT_",
        "GH_",
        "PUSH_",
        "SEQREC_",
        "TASK",
        "PLAN",
        "HOURS",
        "PROFILE_",
        "REPEATS",
        "RUNPOD_",
        "CPU",
        "STOP_",
    )
    full = {k: v for k, v in os.environ.items() if not k.startswith(job)}
    full |= {"GITHUB_REF": "refs/heads/main", "TASK": "run", "HOURS": "22"} | (env or {})
    if path_prefix:
        full["PATH"] = f"{path_prefix}{os.pathsep}{full['PATH']}"
    return subprocess.run(
        ["bash", str(SCRIPT), *args], cwd=cwd, env=full, capture_output=True, text=True, timeout=60
    )


def fake(bin_dir: Path, name: str, body: str) -> None:
    bin_dir.mkdir(exist_ok=True)
    path = bin_dir / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


# --- check-inputs -------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["run", "tune", "test", "export", "bootstrap"])
def test_a_plan_in_the_repository_is_accepted(task):
    assert steps("check-inputs", env={"TASK": task, "PLAN": "m2-anchor.toml"}).returncode == 0


@pytest.mark.parametrize(
    "plan",
    ["", "../x.toml", "a;b.toml", "m2-anchor", "m2-anchor.toml x", "$(id).toml", "nope.toml"],
)
def test_other_plan_names_are_rejected(plan):
    result = steps("check-inputs", env={"PLAN": plan})
    assert result.returncode == 1
    assert "plan" in result.stderr


def test_a_plan_whose_batch_differs_from_its_file_name_is_rejected(tmp_path):
    plans = tmp_path / "configs" / "plans"
    plans.mkdir(parents=True)
    (plans / "a_b-1.toml").write_text('batch = "a_b-1"\n', encoding="utf-8")
    (plans / "other.toml").write_text('batch = "m2-anchor"\n', encoding="utf-8")
    assert steps("check-inputs", cwd=tmp_path, env={"PLAN": "a_b-1.toml"}).returncode == 0
    result = steps("check-inputs", cwd=tmp_path, env={"PLAN": "other.toml"})
    assert result.returncode == 1
    assert "batch" in result.stderr


def test_only_main_is_accepted():
    result = steps("check-inputs", env={"PLAN": "m2-anchor.toml", "GITHUB_REF": "refs/heads/x"})
    assert result.returncode == 1
    assert "main" in result.stderr


@pytest.mark.parametrize("task", ["", "deploy", "run; id"])
def test_unknown_tasks_are_rejected(task):
    assert steps("check-inputs", env={"TASK": task, "PLAN": "m2-anchor.toml"}).returncode == 1


@pytest.mark.parametrize("hours", ["0", "23", "1.5", "", "2h"])
def test_hours_outside_1_to_22_are_rejected(hours):
    result = steps("check-inputs", env={"PLAN": "m2-anchor.toml", "HOURS": hours})
    assert result.returncode == 1
    assert "hours" in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        "--dataset steam --arm sasrec-bce",
        "--dataset ml-1m --arm gsasrec --steps 400 --warmup 0",
        "--dataset ml-1m --arm gsasrec --steps 47 --epochs 400",
        "--dataset steam --arm sasrec-ce --fixed-shapes --fused-adam --cuda-graph --sync-debug",
        "  --dataset   ml-20m --arm bert4rec --check ",
        '--dataset steam --arm gsasrec --params {"dim":128,"dropout":0.5,"shared_output":false}',
    ],
)
def test_profile_arguments_the_script_knows_are_accepted(args):
    env = {"TASK": "profile", "PROFILE_ARGS": args, "REPEATS": "5"}
    assert steps("check-inputs", env=env).returncode == 0


@pytest.mark.parametrize(
    "args",
    [
        "",
        "--arm sasrec-bce",  # no dataset
        "--dataset steam",  # no arm
        "--dataset nope --arm sasrec-bce",
        "--dataset steam --arm sasrec-bce --steps many",
        "--dataset steam --arm sasrec-bce --steps",
        "--dataset steam --arm sasrec-bce --epochs many",
        "--dataset steam --arm sasrec-bce --device cpu",
        "--dataset steam --arm sasrec-bce --data-dir /etc",
        "--dataset steam --arm sasrec-bce --params {}",
        '--dataset steam --arm sasrec-bce --params {"dim":"$(id)"}',
        '--dataset steam --arm sasrec-bce --params {"dim":64};id',
        "--dataset steam --arm sasrec-bce --params [64]",
        "--dataset steam --arm sasrec-bce --from-run /tmp/x",
        "--dataset steam --arm $(id)",
        "--dataset steam --arm a;id",
        "--dataset steam --arm sasrec-bce extra",
    ],
)
def test_other_profile_arguments_are_rejected(args):
    env = {"TASK": "profile", "PROFILE_ARGS": args, "REPEATS": "5"}
    result = steps("check-inputs", env=env)
    assert result.returncode == 1
    assert "profile_args" in result.stderr


@pytest.mark.parametrize("repeats", ["0", "21", "", "3;id", "2.0"])
def test_repeats_outside_1_to_20_are_rejected(repeats):
    env = {"TASK": "profile", "PROFILE_ARGS": "--dataset steam --arm gsasrec", "REPEATS": repeats}
    result = steps("check-inputs", env=env)
    assert result.returncode == 1
    assert "repeats" in result.stderr


SCHEDULES = sorted((REPO / "configs" / "profiles").glob("*.txt"))


def schedule(tmp_path: Path, text: str, name: str = "s.txt") -> dict:
    """A schedule file under tmp_path, and the inputs of a profile job that names it."""
    profiles = tmp_path / "configs" / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / name).write_text(text, encoding="utf-8")
    return {"TASK": "profile", "PLAN": name, "REPEATS": "5"}


def test_the_repository_has_schedules():
    assert SCHEDULES


@pytest.mark.parametrize("path", SCHEDULES, ids=lambda p: p.name)
def test_a_schedule_in_the_repository_is_accepted(path):
    env = {"TASK": "profile", "PLAN": path.name, "PROFILE_ARGS": " ", "REPEATS": "5"}
    result = steps("check-inputs", env=env)
    assert result.returncode == 0, result.stderr


D048 = [p for p in SCHEDULES if p.name.startswith("d048-")]
D048_MODES = (
    "",
    " --fused-adam",
    " --fixed-shapes --fused-adam",
    " --fixed-shapes --fused-adam --cuda-graph",
)
D048_EPOCHS = (100, 100, 100, 100, 58, 25, 17, 30)  # of 200 steps, configurations 1 to 8


def test_d048_has_a_schedule_for_each_of_its_eight_configurations():
    assert [p.name.split("-")[1] for p in D048] == [str(n) for n in range(1, 9)]


@pytest.mark.parametrize("path", D048, ids=lambda p: p.name)
def test_a_d048_schedule_holds_the_processes_of_its_design_in_order(path):
    """Three checks, then five rounds that time each mode once from a moving start, then in
    configurations 1 to 4 one count of synchronisations per mode."""
    n = int(path.name.split("-")[1])
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.startswith("--")]
    base = lines[0].removesuffix(" --check --fused-adam")
    timed = f"{base} --steps 200 --epochs {D048_EPOCHS[n - 1]}"
    want = [f"{base} --check{mode}" for mode in D048_MODES[1:]]
    want += [timed + D048_MODES[(r + k) % 4] for r in range(5) for k in range(4)]
    if n <= 4:
        want += [f"{base} --sync-debug{mode}" for mode in D048_MODES]
    assert lines == want


@pytest.mark.parametrize("path", D048, ids=lambda p: p.name)
def test_a_d048_schedule_names_only_hyperparameters_its_arm_takes(path):
    from seqrec.models.sasrec import OBJECTIVES
    from seqrec.models.trainer import SHARED_DEFAULTS, resolve

    for line in path.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if "--params" in words:
            arm = words[words.index("--arm") + 1]
            params = json.loads(words[words.index("--params") + 1])
            resolve(params, SHARED_DEFAULTS | {"shared_output": True} | OBJECTIVES[arm])


@pytest.mark.parametrize("name", ["../x.txt", "a;b.txt", "s", "s.txt x", "$(id).txt", "s.toml"])
def test_other_schedule_names_are_rejected(tmp_path, name):
    env = schedule(tmp_path, "--dataset steam --arm sasrec-bce\n") | {"PLAN": name}
    result = steps("check-inputs", cwd=tmp_path, env=env)
    assert result.returncode == 1
    assert "plan: not a schedule file name" in result.stderr


def test_a_schedule_that_does_not_exist_is_rejected(tmp_path):
    env = schedule(tmp_path, "--dataset steam --arm sasrec-bce\n") | {"PLAN": "nope.txt"}
    result = steps("check-inputs", cwd=tmp_path, env=env)
    assert result.returncode == 1
    assert "does not exist" in result.stderr


@pytest.mark.parametrize(
    "text",
    [
        "",
        "# nothing but a comment\n",
        "--dataset steam --arm sasrec-bce\n--dataset steam --arm sasrec-bce --device cpu\n",
        "--dataset steam --arm sasrec-bce\n--arm sasrec-bce\n",
    ],
)
def test_a_schedule_with_no_line_or_a_refused_line_is_rejected(tmp_path, text):
    result = steps("check-inputs", cwd=tmp_path, env=schedule(tmp_path, text))
    assert result.returncode == 1
    assert "profile_args" in result.stderr or "plan" in result.stderr


def test_a_schedule_takes_the_place_of_profile_arguments(tmp_path):
    env = schedule(tmp_path, "--dataset steam --arm sasrec-bce\n")
    assert steps("check-inputs", cwd=tmp_path, env=env).returncode == 0
    env["PROFILE_ARGS"] = "--dataset steam --arm sasrec-bce"
    result = steps("check-inputs", cwd=tmp_path, env=env)
    assert result.returncode == 1
    assert "profile_args" in result.stderr


def test_the_test_task_has_no_hour_limit_to_check():
    env = {"TASK": "test", "PLAN": "m2-anchor.toml", "HOURS": ""}
    assert steps("check-inputs", env=env).returncode == 0


def test_the_export_task_needs_a_plan_and_no_hour_limit():
    env = {"TASK": "export", "PLAN": "m2-anchor.toml", "HOURS": ""}
    assert steps("check-inputs", env=env).returncode == 0
    result = steps("check-inputs", env={"TASK": "export", "PLAN": ""})
    assert result.returncode == 1
    assert "plan" in result.stderr


def test_the_bootstrap_task_needs_a_plan_and_no_hour_limit():
    env = {"TASK": "bootstrap", "PLAN": "m2-anchor.toml", "HOURS": ""}
    assert steps("check-inputs", env=env).returncode == 0
    result = steps("check-inputs", env={"TASK": "bootstrap", "PLAN": "../m2-anchor.toml"})
    assert result.returncode == 1
    assert "plan" in result.stderr


def test_setup_needs_no_other_input():
    assert steps("check-inputs", env={"TASK": "setup"}).returncode == 0


# --- status -------------------------------------------------------------------------------

STATUS_ENV = {
    "GH_TOKEN": "t0ken",
    "GITHUB_API_URL": "https://api.example",
    "GITHUB_REPOSITORY": "o/r",
    "GITHUB_SHA": "abc123",
    "GITHUB_RUN_ID": "77",
    "GITHUB_SERVER_URL": "https://github.example",
}
RECORD_POSTS = 'printf "%s\\n" "$@" | grep "^{{" >> {}/posts; cat > /dev/null'


def test_status_posts_one_commit_status(tmp_path):
    fake(
        tmp_path / "bin", "curl", f'printf "%s\\n" "$@" > {tmp_path}/curl-args; cat > {tmp_path}/in'
    )
    result = steps("status", "pending", 'say "hi"\n' + "x" * 200, env=STATUS_ENV,
                   path_prefix=tmp_path / "bin")  # fmt: skip
    assert result.returncode == 0
    args = (tmp_path / "curl-args").read_text(encoding="utf-8")
    assert "https://api.example/repos/o/r/statuses/abc123" in args
    assert "t0ken" not in args  # the token goes in on stdin, not on the command line
    assert (tmp_path / "in").read_text(encoding="utf-8").strip() == "Authorization: Bearer t0ken"
    body = next(line for line in args.splitlines() if line.startswith("{"))
    sent = json.loads(body)
    assert sent["state"] == "pending"
    assert sent["context"] == "batch/77"  # one context per job: jobs do not overwrite each other
    assert sent["target_url"] == "https://github.example/o/r/actions/runs/77"
    assert sent["description"].startswith("say hi")
    assert len(sent["description"]) <= 140
    assert "t0ken" not in result.stdout + result.stderr


def test_a_failing_status_call_does_not_fail_the_step(tmp_path):
    fake(tmp_path / "bin", "curl", "exit 22")
    result = steps("status", "pending", "x", env=STATUS_ENV, path_prefix=tmp_path / "bin")
    assert result.returncode == 0


# --- plan ---------------------------------------------------------------------------------


def plan_env(tmp_path, **more):
    return STATUS_ENV | {
        "TASK": "run",
        "PLAN": "m2-anchor.toml",
        "SEQREC_HOME": str(tmp_path / "home"),
        "PROGRESS_SECONDS": "1",
    } | more  # fmt: skip


def test_plan_runs_the_task_and_reports_start_and_success(tmp_path):
    fake(tmp_path / "bin", "curl", RECORD_POSTS.format(tmp_path))
    fake(tmp_path / "bin", "uv", f'echo "$@" > {tmp_path}/uv-args; env > {tmp_path}/uv-env')
    runs = tmp_path / "home" / "runs" / "m2-anchor"
    for name in ("r1", "r2"):
        (runs / name).mkdir(parents=True)
    (runs / "r1" / "summary.json").write_text("{}", encoding="utf-8")
    result = steps("plan", env=plan_env(tmp_path), path_prefix=tmp_path / "bin")
    assert result.returncode == 0, result.stderr
    home = tmp_path / "home"
    assert (tmp_path / "uv-args").read_text(encoding="utf-8").split() == [
        "run", "--no-sync", "seqrec", "run", "configs/plans/m2-anchor.toml",
        "--runs-dir", f"{home}/runs", "--data-dir", f"{home}/data",
    ]  # fmt: skip
    posts = (tmp_path / "posts").read_text(encoding="utf-8").splitlines()
    assert '"state":"pending"' in posts[0]
    assert '"state":"pending"' in posts[-1]  # success is reported only after the commit step
    assert "1 runs trained" in posts[-1]
    assert "plan finished" in posts[-1]
    assert "last" not in posts[-1]  # a summary without the three fields is not described
    assert "t0ken" not in (tmp_path / "uv-env").read_text(encoding="utf-8")


def summary(run_dir: Path, stop: str, epochs: int, wall_s: float, mtime: int) -> None:
    run_dir.mkdir(parents=True)
    path = run_dir / "summary.json"
    nested = {"status": "x", "stop_reason": "nested", "epochs": 1, "wall_s": 1.0}
    path.write_text(
        json.dumps(
            {"run_id": run_dir.name, "stop_reason": stop, "epochs": epochs, "wall_s": wall_s,
             "nvml": nested},
            indent=1,
        ),
        encoding="utf-8",
    )  # fmt: skip
    for written in (path, run_dir):  # the directory too: a file system may stamp both alike
        os.utime(written, (mtime, mtime))


def test_progress_describes_the_run_that_finished_last(tmp_path):
    fake(tmp_path / "bin", "curl", RECORD_POSTS.format(tmp_path))
    fake(tmp_path / "bin", "uv", "true")
    runs = tmp_path / "home" / "runs" / "m2-anchor"
    summary(runs / "older", "max_epochs", 200, 100.4, 1_000)
    summary(runs / "later", "patience", 384, 3868.2, 2_000)
    (runs / "running").mkdir()
    os.utime(runs / "running", (3_000, 3_000))
    result = steps("plan", env=plan_env(tmp_path), path_prefix=tmp_path / "bin")
    assert result.returncode == 0, result.stderr
    sent = json.loads((tmp_path / "posts").read_text(encoding="utf-8").splitlines()[0])
    text = sent["description"]
    assert "2 runs trained" in text
    assert "last later: patience, 384 epochs, 64 min" in text
    assert "newest running" in text


def test_the_end_of_a_plan_is_reported_before_the_run_names(tmp_path):
    """The description is cut at 140 characters; how the plan ended must survive that."""
    fake(tmp_path / "bin", "curl", RECORD_POSTS.format(tmp_path))
    fake(tmp_path / "bin", "uv", "exit 3")
    runs = tmp_path / "home" / "runs" / "m2-anchor"
    summary(runs / ("r" * 120), "patience", 384, 3868.2, 2_000)
    result = steps("plan", env=plan_env(tmp_path), path_prefix=tmp_path / "bin")
    assert result.returncode == 3
    sent = json.loads((tmp_path / "posts").read_text(encoding="utf-8").splitlines()[-1])
    assert "exit code 3" in sent["description"]


def test_plan_passes_a_failure_on_and_reports_it(tmp_path):
    fake(tmp_path / "bin", "curl", RECORD_POSTS.format(tmp_path))
    fake(tmp_path / "bin", "uv", "exit 3")
    result = steps("plan", env=plan_env(tmp_path), path_prefix=tmp_path / "bin")
    assert result.returncode == 3
    assert '"state":"failure"' in (tmp_path / "posts").read_text(encoding="utf-8").splitlines()[-1]


def test_plan_reports_while_running_and_stops_at_the_hour_limit(tmp_path):
    fake(tmp_path / "bin", "curl", RECORD_POSTS.format(tmp_path))
    fake(tmp_path / "bin", "uv", "sleep 30")
    env = plan_env(tmp_path, HOURS="1", HOUR_SECONDS="3")  # an hour of 3 seconds
    result = steps("plan", env=env, path_prefix=tmp_path / "bin")
    assert result.returncode == 124  # timeout(1)'s code for a command it stopped
    posts = (tmp_path / "posts").read_text(encoding="utf-8").splitlines()
    assert sum('"state":"pending"' in p for p in posts) >= 2  # the start and a periodic one
    assert '"state":"failure"' in posts[-1]
    assert "hour limit" in posts[-1]
    leftover = subprocess.run(["pgrep", "-f", "sleep 30"], capture_output=True, text=True)
    assert leftover.stdout == ""


def test_the_test_task_is_never_cut_short(tmp_path):
    """A test read that is interrupted counts as used and cannot be repeated."""
    fake(tmp_path / "bin", "curl", "cat > /dev/null")
    fake(tmp_path / "bin", "uv", "sleep 2")
    env = plan_env(tmp_path, TASK="test", HOURS="1", HOUR_SECONDS="1")
    assert steps("plan", env=env, path_prefix=tmp_path / "bin").returncode == 0


# --- profile ------------------------------------------------------------------------------


def test_profile_runs_one_process_per_repeat(tmp_path):
    fake(tmp_path / "bin", "uv", f'echo "$@" >> {tmp_path}/uv-args')
    env = {
        "TASK": "profile",
        "PROFILE_ARGS": "--dataset steam --arm sasrec-bce --steps 50",
        "REPEATS": "3",
        "SEQREC_HOME": str(tmp_path / "home"),
    }
    result = steps("profile", env=env, path_prefix=tmp_path / "bin")
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "uv-args").read_text(encoding="utf-8").splitlines()
    one = (
        "run --no-sync python scripts/profile_step.py "
        f"--data-dir {tmp_path}/home/data --dataset steam --arm sasrec-bce --steps 50"
    )
    assert calls == [one] * 3


def test_profile_runs_every_line_of_a_schedule_once_in_order(tmp_path):
    fake(tmp_path / "bin", "uv", f'echo "$@" >> {tmp_path}/uv-args')
    text = (
        "# round 1\n"
        "--dataset steam --arm sasrec-ce --epochs 100\n"
        "\n"
        "--dataset steam --arm sasrec-ce --epochs 100 --fused-adam\n"
        '--dataset steam --arm gsasrec --params {"dim":128,"shared_output":false} --check\n'
    )
    env = schedule(tmp_path, text) | {"SEQREC_HOME": str(tmp_path / "home")}
    result = steps("profile", cwd=tmp_path, env=env, path_prefix=tmp_path / "bin")
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "uv-args").read_text(encoding="utf-8").splitlines()
    head = f"run --no-sync python scripts/profile_step.py --data-dir {tmp_path}/home/data "
    lines = [line for line in text.splitlines() if line.startswith("--")]
    assert calls == [head + line for line in lines]
    # Each process is announced by its line, so a log can be split by process.
    assert [x for x in result.stdout.splitlines() if x.startswith("profile: ")] == [
        f"profile: {line}" for line in lines
    ]


def test_profile_goes_on_after_a_failed_process_and_fails_at_the_end(tmp_path):
    fake(
        tmp_path / "bin",
        "uv",
        f'echo "$@" >> {tmp_path}/uv-args; case "$*" in *--fused-adam*) exit 1 ;; esac',
    )
    text = (
        "--dataset steam --arm sasrec-ce\n"
        "--dataset steam --arm sasrec-ce --fused-adam\n"
        "--dataset steam --arm sasrec-ce --fixed-shapes\n"
    )
    env = schedule(tmp_path, text) | {"SEQREC_HOME": str(tmp_path / "home")}
    result = steps("profile", cwd=tmp_path, env=env, path_prefix=tmp_path / "bin")
    assert result.returncode == 1
    assert len((tmp_path / "uv-args").read_text(encoding="utf-8").splitlines()) == 3
    assert "failed: --dataset steam --arm sasrec-ce --fused-adam" in result.stderr


# --- data-unchanged and commit ------------------------------------------------------------


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def clone(tmp_path):
    """A clone of a bare `origin` with one commit by A U Thor on main, and a second clone."""
    origin = tmp_path / "origin.git"
    git("init", "--bare", "-b", "main", str(origin), cwd=tmp_path)
    work, other = tmp_path / "work", tmp_path / "other"
    git("clone", str(origin), str(work), cwd=tmp_path)
    ident = ["-c", "user.name=A U Thor", "-c", "user.email=author@example.com"]
    for rel in ("results/data/stats.csv", "results/runs/b1.csv", "results/runs/b2.csv"):
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text("run_id,x\nr1,v1\n", encoding="utf-8")
    git("add", ".", cwd=work)
    git(*ident, "commit", "-m", "feat: first", cwd=work)
    git("push", "origin", "HEAD:main", cwd=work)
    git("clone", str(origin), str(other), cwd=tmp_path)
    return work, other, origin


def log(repo, fmt="%an <%ae> %s"):
    out = subprocess.run(
        ["git", "log", "-1", f"--format={fmt}", "main"], cwd=repo, capture_output=True, text=True
    )
    return out.stdout.strip()


def test_data_unchanged_passes_on_a_clean_tree_and_fails_on_any_difference(clone):
    work, _, _ = clone
    assert steps("data-unchanged", cwd=work).returncode == 0
    (work / "results/data/stats.csv").write_text("run_id,x\nr1,v2\n", encoding="utf-8")
    result = steps("data-unchanged", cwd=work)
    assert result.returncode == 1
    assert "+r1,v2" in result.stdout
    git("checkout", "results/data/stats.csv", cwd=work)
    (work / "results/data/new.csv").write_text("x\n", encoding="utf-8")
    result = steps("data-unchanged", cwd=work)
    assert result.returncode == 1
    assert "new.csv" in result.stdout


OWNER = "Aaron Yang <yang341133@gmail.com>"


def push(repo, rel, text, subject):
    (repo / rel).write_text(text, encoding="utf-8")
    git("add", ".", cwd=repo)
    git("-c", "user.name=X", "-c", "user.email=x@example.com", "commit", "-m", subject, cwd=repo)
    git("push", "origin", "HEAD:main", cwd=repo)


def test_commit_pushes_only_the_named_paths_under_a_fixed_identity(clone):
    work, _, origin = clone
    (work / "results/runs/b1.csv").write_text("run_id,x\nr1,v2\n", encoding="utf-8")
    (work / "results/runs/b2.csv").write_text("not this job\n", encoding="utf-8")
    result = steps("commit", "chore(results): add run b1 results", "results/runs/b1.csv",
                   cwd=work, env={"GITHUB_RUN_ID": "77", "GITHUB_SHA": "abc123"})  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert log(origin) == f"{OWNER} chore(results): add run b1 results"
    assert log(origin, "%cn <%ce>") == OWNER
    assert log(origin, "%b") == "Run 77 at abc123"
    names = subprocess.run(
        ["git", "show", "--name-only", "--format=", "main"],
        cwd=origin, capture_output=True, text=True,
    )  # fmt: skip
    assert names.stdout.split() == ["results/runs/b1.csv"]
    assert (work / "results/runs/b2.csv").read_text(encoding="utf-8") == "not this job\n"


def test_commit_refuses_a_path_that_does_not_exist(clone):
    work, _, origin = clone
    result = steps("commit", "chore(results): x", "results/runs/missing.csv", cwd=work)
    assert result.returncode == 1
    assert "does not exist" in result.stderr
    assert log(origin).endswith("feat: first")


def test_commit_does_nothing_when_the_paths_are_unchanged(clone):
    work, _, origin = clone
    (work / "results/runs/b2.csv").write_text("not this job\n", encoding="utf-8")
    result = steps("commit", "chore(results): x", "results/runs/b1.csv", cwd=work)
    assert result.returncode == 0
    assert "nothing to commit" in result.stdout
    assert log(origin).endswith("feat: first")


def test_commit_lands_on_top_of_commits_pushed_meanwhile(clone):
    work, other, origin = clone
    push(other, "README.md", "docs\n", "docs: x")
    (work / "results/runs/b1.csv").write_text("run_id,x\nr1,v2\n", encoding="utf-8")
    result = steps("commit", "chore(results): add b1", "results/runs/b1.csv", cwd=work)
    assert result.returncode == 0, result.stderr
    assert log(origin, "%s") == "chore(results): add b1"
    parent = subprocess.run(
        ["git", "log", "-1", "--format=%s", "main~1"], cwd=origin, capture_output=True, text=True
    )
    assert parent.stdout.strip() == "docs: x"
    files = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "main"], cwd=origin, capture_output=True, text=True
    )
    assert "README.md" in files.stdout.split()


def test_commit_replaces_the_file_on_main_with_the_new_export(clone):
    work, other, origin = clone
    push(other, "results/runs/b1.csv", "run_id,x\nr1,old\n", "chore(results): earlier export")
    (work / "results/runs/b1.csv").write_text("run_id,x\nr1,new\nr2,new\n", encoding="utf-8")
    result = steps("commit", "chore(results): add b1", "results/runs/b1.csv", cwd=work)
    assert result.returncode == 0, result.stderr
    shown = subprocess.run(
        ["git", "show", "main:results/runs/b1.csv"], cwd=origin, capture_output=True, text=True
    )
    assert shown.stdout == "run_id,x\nr1,new\nr2,new\n"


def test_commit_refuses_an_export_that_lacks_runs_main_has(clone):
    """The same batch run on another machine: its rows must not disappear from main."""
    work, other, origin = clone
    push(other, "results/runs/b1.csv", "run_id,x\nr1,a\nr9,a\n", "chore(results): other machine")
    (work / "results/runs/b1.csv").write_text("run_id,x\nr1,b\nr2,b\n", encoding="utf-8")
    result = steps("commit", "chore(results): add b1", "results/runs/b1.csv", cwd=work)
    assert result.returncode == 1
    assert "r9" in result.stderr
    assert log(origin, "%s") == "chore(results): other machine"


def test_commit_adds_new_files_of_a_directory_and_keeps_those_on_main(clone):
    work, other, origin = clone
    push(other, "results/data/later.csv", "x\n", "chore: later")
    (work / "results/data/report.txt").write_text("r\n", encoding="utf-8")
    result = steps("commit", "chore(env): add a report", "results/data", cwd=work)
    assert result.returncode == 0, result.stderr
    files = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "main"], cwd=origin, capture_output=True, text=True
    ).stdout.split()
    assert {"results/data/later.csv", "results/data/report.txt", "results/data/stats.csv"} <= set(
        files
    )


def test_commit_with_push_dry_run_leaves_the_remote_alone(clone):
    work, _, origin = clone
    (work / "results/runs/b1.csv").write_text("run_id,x\nr1,v2\n", encoding="utf-8")
    result = steps("commit", "chore(results): add b1", "results/runs/b1.csv", cwd=work,
                   env={"PUSH_DRY_RUN": "1"})  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert log(origin).endswith("feat: first")


# --- results: the batch file and the summaries --------------------------------------------

# Stands in for `seqrec export --summaries-only --out-dir DIR`: the "summary" lists the batch
# files it found in DIR, which shows where and from what it was computed.
SUMMARIZE = (
    'echo "$@" >> {0}/uv-args; dir="${{@: -1}}"; '
    'if [[ " $* " == *" --check-reads "* ]]; then cp "$dir/b1.csv" {0}/main-b1; '
    'exit "${{CHECK_FAIL:-0}}"; fi; [ -z "${{UV_FAIL:-}}" ] || exit 1; '
    '(cd "$dir" && ls | grep -v overview > overview.csv)'
)


def results_env(work):
    plans = work / "configs" / "plans"
    plans.mkdir(parents=True)
    (plans / "b1.toml").write_text('batch = "b1"\n', encoding="utf-8")
    (work / "results/runs/b1.csv").write_text("run_id,x\nr1,v2\n", encoding="utf-8")
    return {"TASK": "run", "PLAN": "b1.toml", "SEQREC_HOME": str(work.parent / "home")}


def test_results_commits_the_batch_file_and_summaries_of_all_batch_files_on_main(clone, tmp_path):
    work, other, origin = clone
    fake(tmp_path / "bin", "uv", SUMMARIZE.format(tmp_path))
    push(other, "results/runs/b3.csv", "run_id,x\nr1,v1\n", "chore(results): another batch")
    (work / "results/runs/overview.csv").write_text("from this checkout\n", encoding="utf-8")
    result = steps("results", cwd=work, env=results_env(work), path_prefix=tmp_path / "bin")
    assert result.returncode == 0, result.stderr
    assert log(origin, "%s") == "chore(results): add run b1 results"
    names = subprocess.run(
        ["git", "show", "--name-only", "--format=", "main"],
        cwd=origin, capture_output=True, text=True,
    )  # fmt: skip
    assert names.stdout.split() == ["results/runs/b1.csv", "results/runs/overview.csv"]
    shown = subprocess.run(
        ["git", "show", "main:results/runs/overview.csv"],
        cwd=origin,
        capture_output=True,
        text=True,
    )
    assert shown.stdout.split() == ["b1.csv", "b2.csv", "b3.csv"]  # b3 is not in this checkout
    check, args = (a.split() for a in (tmp_path / "uv-args").read_text("utf-8").splitlines())
    assert check[:7] == ["run", "--no-sync", "seqrec", "export", "--check-reads", "--runs-dir",
                         f"{tmp_path}/home/runs"]  # fmt: skip
    assert check[7:] == ["--out-dir", args[6]]
    assert args[:6] == ["run", "--no-sync", "seqrec", "export", "--summaries-only", "--out-dir"]
    assert args[6].endswith("/main/results/runs") and not args[6].startswith(str(work))


def test_results_pushes_nothing_when_main_records_reads_this_machine_lacks(clone, tmp_path):
    """Main may have moved since the checkout: its batch files are checked as they are now,
    before any of them is replaced."""
    work, other, origin = clone
    fake(tmp_path / "bin", "uv", SUMMARIZE.format(tmp_path))
    push(other, "results/runs/b1.csv", "run_id,x\nr1,elsewhere\n", "chore(results): other volume")
    env = results_env(work) | {"CHECK_FAIL": "1"}
    result = steps("results", cwd=work, env=env, path_prefix=tmp_path / "bin")
    assert result.returncode == 1
    assert "test reads" in result.stderr
    assert log(origin, "%s") == "chore(results): other volume"
    assert (tmp_path / "main-b1").read_text(encoding="utf-8") == "run_id,x\nr1,elsewhere\n"


def test_results_pushes_nothing_when_the_summaries_fail(clone, tmp_path):
    """`seqrec export` exits 1 on a test-access problem, e.g. too many anchor reads."""
    work, _, origin = clone
    fake(tmp_path / "bin", "uv", SUMMARIZE.format(tmp_path))
    env = results_env(work) | {"UV_FAIL": "1"}
    result = steps("results", cwd=work, env=env, path_prefix=tmp_path / "bin")
    assert result.returncode == 1
    assert "summaries" in result.stderr
    assert log(origin).endswith("feat: first")


def test_results_needs_the_batch_file_of_the_plan(clone, tmp_path):
    work, _, origin = clone
    fake(tmp_path / "bin", "uv", SUMMARIZE.format(tmp_path))
    env = results_env(work)
    (work / "results/runs/b1.csv").unlink()
    result = steps("results", cwd=work, env=env, path_prefix=tmp_path / "bin")
    assert result.returncode == 1
    assert "does not exist" in result.stderr
    assert steps("results", cwd=work, env=env | {"PLAN": "../b1.toml"}).returncode == 1
    assert log(origin).endswith("feat: first")


BOOTSTRAP = (
    'echo "$@" > {0}/uv-args; env > {0}/uv-env; out="${{@: -1}}"; '
    'mkdir -p "$(dirname "$out")"; printf "dataset,low\\nml-1m,0.1\\n" > "$out"'
)


def test_bootstrap_computes_its_table_without_the_token_and_commits_it_alone(clone, tmp_path):
    """The intervals of METHODS 7.6 come from the plan's batch on the volume and are pushed
    as results/analysis/bootstrap-<batch>.csv (D056)."""
    work, _, origin = clone
    fake(tmp_path / "bin", "uv", BOOTSTRAP.format(tmp_path))
    env = results_env(work) | {"TASK": "bootstrap", "GH_TOKEN": "s3cret"}
    result = steps("bootstrap", cwd=work, env=env, path_prefix=tmp_path / "bin")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "uv-args").read_text(encoding="utf-8").split() == [
        "run", "--no-sync", "seqrec", "bootstrap", f"{tmp_path}/home/runs/b1",
        "--out", "results/analysis/bootstrap-b1.csv",
    ]  # fmt: skip
    seen = (tmp_path / "uv-env").read_text(encoding="utf-8")
    assert "s3cret" not in seen and "AUTHORIZATION" not in seen
    assert log(origin, "%s") == "chore(results): add the bootstrap of b1"
    names = subprocess.run(
        ["git", "show", "--name-only", "--format=", "main"],
        cwd=origin, capture_output=True, text=True,
    )  # fmt: skip
    assert names.stdout.split() == ["results/analysis/bootstrap-b1.csv"]
    assert steps("bootstrap", cwd=work, env=env | {"PLAN": "../b1.toml"}).returncode == 1


# --- watchdog-running and export ----------------------------------------------------------


def test_a_job_refuses_a_machine_whose_watchdog_is_not_running(tmp_path):
    """Without the watchdog nothing stops the machine; it holds this lock while it runs."""
    import fcntl
    import time

    env = {"POD_START_LOCK": str(tmp_path / "lock"), "POD_START": f"{tmp_path}/pod_start.sh"}
    result = steps("watchdog-running", env=env)
    assert result.returncode == 1
    assert "watchdog is not running" in result.stderr
    with (tmp_path / "lock").open("w") as other:  # held, but not by the watchdog
        fcntl.flock(other, fcntl.LOCK_EX)
        result = steps("watchdog-running", env=env)
    assert result.returncode == 1
    assert "not by the watchdog" in result.stderr
    fake(tmp_path, "pod_start.sh", f"exec 9> {tmp_path}/lock; flock 9; sleep 5 9>&-")
    watchdog = subprocess.Popen([str(tmp_path / "pod_start.sh")], start_new_session=True)
    try:
        time.sleep(0.3)
        assert steps("watchdog-running", env=env).returncode == 0
    finally:
        os.killpg(watchdog.pid, signal.SIGKILL)  # with its sleep: no process is left behind
        watchdog.wait()


# Stands in for `seqrec export`: lists $WROTE in the file named after --written.
EXPORT = (
    'echo "$@" > {0}/uv-args; ls results/runs > {0}/seen; while [ $# -gt 1 ]; do '
    '[ "$1" != --written ] || printf "%s" "${{WROTE:-}}" > "$2"; shift; done; '
    'exit "${{UV_EXIT:-0}}"'
)


def test_export_passes_only_a_batch_file_this_export_wrote(clone, tmp_path):
    """A file that came with the checkout must not pass for this machine's export. It stays
    where it is all the same: the export refuses to replace one whose reads it lacks."""
    work, _, _ = clone
    env = results_env(work)
    fake(tmp_path / "bin", "uv", EXPORT.format(tmp_path))

    def export_step(**more):
        return steps("export", cwd=work, env=env | more, path_prefix=tmp_path / "bin")

    result = export_step()  # wrote nothing
    assert result.returncode == 1
    assert "no run of b1" in result.stderr
    assert "b1.csv" in (tmp_path / "seen").read_text(encoding="utf-8").split()
    args = (tmp_path / "uv-args").read_text(encoding="utf-8").split()
    assert args[:9] == [
        "run", "--no-sync", "seqrec", "export", "--runs-dir", f"{tmp_path}/home/runs",
        "--plan", "configs/plans/b1.toml", "--written",
    ]  # fmt: skip
    assert len(args) == 10 and not Path(args[9]).exists()  # the list is removed afterwards
    assert export_step(WROTE="b1.csv\n").returncode == 0
    assert export_step(WROTE="b1.csv\n", UV_EXIT="1").returncode == 1
    assert export_step(WROTE="b2.csv\nxb1.csv\n").returncode == 1  # other batches only


def test_a_lock_that_cannot_be_checked_does_not_pass_for_a_running_watchdog(tmp_path):
    result = steps("watchdog-running", env={"POD_START_LOCK": str(tmp_path / "missing" / "lock")})
    assert result.returncode == 1
    assert "cannot check" in result.stderr


def test_the_summaries_are_computed_without_the_push_token(clone, tmp_path):
    """The summaries run repository code; the token is for the push alone."""
    work, _, origin = clone
    fake(tmp_path / "bin", "uv", f"env >> {tmp_path}/uv-env; " + SUMMARIZE.format(tmp_path))
    env = results_env(work) | {"GH_TOKEN": "s3cret"}
    result = steps("results", cwd=work, env=env, path_prefix=tmp_path / "bin")
    assert result.returncode == 0, result.stderr
    assert log(origin, "%s") == "chore(results): add run b1 results"
    seen = (tmp_path / "uv-env").read_text(encoding="utf-8")
    assert "s3cret" not in seen and "AUTHORIZATION" not in seen


# --- machine ------------------------------------------------------------------------------


def machine_step(tmp_path, driver, pod="pod1", cpu=""):
    """`batch_steps.sh machine` on a fake machine whose GPU driver is `driver`, for a job that
    requires the CPU model `cpu`."""
    bin_dir = tmp_path / "bin"
    table = f'echo "RTX 5090, {driver}, 32607 MiB"'
    fake(bin_dir, "nvidia-smi", f'[[ "$*" == *=driver_version* ]] && echo {driver} || {table}')
    fake(bin_dir, "lscpu", 'echo "Model name: Fake CPU"')
    fake(bin_dir, "nproc", "echo 16")
    fake(bin_dir, "uname", "echo 6.8.0-fake")
    fake(bin_dir, "uv", '[ "$1" = --version ] && echo "uv 0.11.32" || echo "2.9.0+cu132 RTX 5090"')
    (tmp_path / "runner").mkdir()
    fake(tmp_path / "runner" / "bin", "Runner.Listener", "echo 2.330.0")
    (tmp_path / "runner.log").write_text("starting the runner\n", encoding="utf-8")
    env = {"RUNNER_DIR": str(tmp_path / "runner"), "SEQREC_HOME": str(tmp_path)}
    env |= {"RUNPOD_POD_ID": pod, "CPU": cpu, "STOP_REQUEST": str(tmp_path / "stop-request")}
    return steps("machine", env=env, path_prefix=bin_dir)


def test_the_machine_step_refuses_a_driver_the_cuda_13_wheels_do_not_run_on(tmp_path):
    result = machine_step(tmp_path, "570.86.16")
    assert result.returncode == 1
    assert "driver branch below 580" in result.stderr


def test_the_machine_step_describes_a_machine_with_a_new_enough_driver(tmp_path):
    result = machine_step(tmp_path, "580.95.05")
    assert result.returncode == 0, result.stderr
    for text in ("RTX 5090, 580.95.05, 32607 MiB", "Model name: Fake CPU", "16", "6.8.0-fake",
                 "uv 0.11.32", "runner 2.330.0", "2.9.0+cu132 RTX 5090", "pod pod1"):  # fmt: skip
        assert text in result.stdout.splitlines()
    assert result.stdout.endswith("starting the runner\n")  # the end of the watchdog's log


def test_the_machine_step_refuses_another_cpu_and_asks_for_the_machine_to_be_deleted(tmp_path):
    """METHODS 9.4: every batch runs on one kind of host. The job that finds another one
    stops before any work, and the machine goes whatever the job's `stop` input says."""
    result = machine_step(tmp_path, "580.95.05", cpu="Ryzen 9 7950X")
    assert result.returncode == 1
    assert "cpu: this machine has Fake CPU, not Ryzen 9 7950X" in result.stderr
    assert (tmp_path / "stop-request").exists()
    assert "pod pod1" in result.stdout.splitlines()  # described all the same


@pytest.mark.parametrize("cpu", ["Fake CPU", "Fake", ""])
def test_the_machine_step_accepts_the_required_cpu_or_any_when_none_is_required(tmp_path, cpu):
    result = machine_step(tmp_path, "580.95.05", cpu=cpu)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "stop-request").exists()


def test_the_required_cpu_matches_whole_words_of_the_model_name(tmp_path):
    """`Fake CP` is not the model `Fake CPU`, as a 7950X is not a 7950X3D."""
    result = machine_step(tmp_path, "580.95.05", cpu="Fake CP")
    assert result.returncode == 1
    assert "cpu: this machine has Fake CPU, not Fake CP" in result.stderr


@pytest.mark.parametrize("cpu", ["x;y", "$(id)", "a*", "é", "x" * 41])
def test_other_cpu_names_are_rejected(cpu):
    result = steps("check-inputs", env={"PLAN": "m2-anchor.toml", "CPU": cpu})
    assert result.returncode == 1
    assert "cpu" in result.stderr


def test_a_cpu_name_of_letters_digits_and_spaces_is_accepted():
    env = {"PLAN": "m2-anchor.toml", "CPU": "Ryzen 9 7950X"}
    assert steps("check-inputs", env=env).returncode == 0


def test_the_machine_step_refuses_a_machine_without_a_pod_id(tmp_path):
    """Run times are compared only within one Pod (D046); runs that record no Pod would pass
    for runs of the same one."""
    result = machine_step(tmp_path, "580.95.05", pod="")
    assert result.returncode == 1
    assert "RUNPOD_POD_ID" in result.stderr
