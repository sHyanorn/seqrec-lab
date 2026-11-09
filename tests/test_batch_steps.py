"""The batch workflow's shell steps (scripts/cloud/batch_steps.sh), run against fakes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "cloud" / "batch_steps.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def steps(*args, cwd=REPO, env=None, path_prefix=None):
    full = {k: v for k, v in os.environ.items() if not k.startswith(("GITHUB_", "GIT_"))}
    full |= {"GITHUB_REF": "refs/heads/main", "TASK": "run", "HOURS": "23"} | (env or {})
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


@pytest.mark.parametrize("task", ["run", "tune", "test"])
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


@pytest.mark.parametrize("hours", ["0", "24", "1.5", "", "2h"])
def test_hours_outside_1_to_23_are_rejected(hours):
    result = steps("check-inputs", env={"PLAN": "m2-anchor.toml", "HOURS": hours})
    assert result.returncode == 1
    assert "hours" in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        "--dataset steam --arm sasrec-bce",
        "--dataset ml-1m --arm gsasrec --steps 400 --warmup 0",
        "--dataset steam --arm sasrec-ce --fixed-shapes --fused-adam --cuda-graph --sync-debug",
        "  --dataset   ml-20m --arm bert4rec --check ",
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
        "--dataset steam --arm sasrec-bce --device cpu",
        "--dataset steam --arm sasrec-bce --data-dir /etc",
        "--dataset steam --arm sasrec-bce --params {}",
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


def test_setup_needs_no_other_input():
    assert steps("check-inputs", env={"TASK": "setup"}).returncode == 0


# --- status -------------------------------------------------------------------------------

STATUS_ENV = {
    "GH_TOKEN": "t0ken",
    "GITHUB_API_URL": "https://api.example",
    "GITHUB_REPOSITORY": "o/r",
    "GITHUB_SHA": "abc123",
}


def test_status_posts_one_commit_status(tmp_path):
    fake(tmp_path / "bin", "curl", f'printf "%s\\n" "$@" > {tmp_path}/curl-args')
    result = steps("status", "pending", 'say "hi"\n' + "x" * 200, env=STATUS_ENV,
                   path_prefix=tmp_path / "bin")  # fmt: skip
    assert result.returncode == 0
    args = (tmp_path / "curl-args").read_text(encoding="utf-8")
    assert "https://api.example/repos/o/r/statuses/abc123" in args
    assert "Authorization: Bearer t0ken" in args
    body = next(line for line in args.splitlines() if line.startswith("{"))
    sent = json.loads(body)
    assert sent["state"] == "pending"
    assert sent["context"] == "batch/progress"
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
    fake(tmp_path / "bin", "curl", f'printf "%s\\n" "$@" | grep "^{{" >> {tmp_path}/posts')
    fake(tmp_path / "bin", "uv", f'echo "$@" > {tmp_path}/uv-args')
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
    assert '"state":"success"' in posts[-1]
    assert "1 runs trained" in posts[-1]


def test_plan_passes_a_failure_on_and_reports_it(tmp_path):
    fake(tmp_path / "bin", "curl", f'printf "%s\\n" "$@" | grep "^{{" >> {tmp_path}/posts')
    fake(tmp_path / "bin", "uv", "exit 3")
    result = steps("plan", env=plan_env(tmp_path), path_prefix=tmp_path / "bin")
    assert result.returncode == 3
    assert '"state":"failure"' in (tmp_path / "posts").read_text(encoding="utf-8").splitlines()[-1]


def test_plan_reports_while_running_and_stops_at_the_hour_limit(tmp_path):
    fake(tmp_path / "bin", "curl", f'printf "%s\\n" "$@" | grep "^{{" >> {tmp_path}/posts')
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
        (work / rel).write_text("v1\n", encoding="utf-8")
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
    (work / "results/data/stats.csv").write_text("v2\n", encoding="utf-8")
    result = steps("data-unchanged", cwd=work)
    assert result.returncode == 1
    assert "+v2" in result.stdout
    git("checkout", "results/data/stats.csv", cwd=work)
    (work / "results/data/new.csv").write_text("x\n", encoding="utf-8")
    result = steps("data-unchanged", cwd=work)
    assert result.returncode == 1
    assert "new.csv" in result.stdout


def test_commit_adds_only_the_named_paths_and_pushes_as_the_last_author(clone):
    work, _, origin = clone
    (work / "results/runs/b1.csv").write_text("v2\n", encoding="utf-8")
    (work / "results/runs/b2.csv").write_text("not this job\n", encoding="utf-8")
    result = steps(
        "commit", "chore(results): add run b1.toml results",
        "results/runs/b1.csv", "results/runs/tuning_best.csv", cwd=work,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert log(origin) == "A U Thor <author@example.com> chore(results): add run b1.toml results"
    names = subprocess.run(
        ["git", "show", "--name-only", "--format=", "main"],
        cwd=origin, capture_output=True, text=True,
    )  # fmt: skip
    assert names.stdout.split() == ["results/runs/b1.csv"]


def test_commit_does_nothing_when_the_paths_are_unchanged(clone):
    work, _, origin = clone
    (work / "results/runs/b2.csv").write_text("not this job\n", encoding="utf-8")
    result = steps("commit", "chore(results): x", "results/runs/b1.csv", cwd=work)
    assert result.returncode == 0
    assert log(origin).endswith("feat: first")


def test_commit_lands_on_top_of_commits_pushed_meanwhile(clone):
    work, other, origin = clone
    (other / "README.md").write_text("docs\n", encoding="utf-8")
    git("add", ".", cwd=other)
    git("-c", "user.name=X", "-c", "user.email=x@example.com", "commit", "-m", "docs: x", cwd=other)
    git("push", "origin", "HEAD:main", cwd=other)
    (work / "results/runs/b1.csv").write_text("v2\n", encoding="utf-8")
    (work / "results/runs/b2.csv").write_text("not this job\n", encoding="utf-8")
    result = steps("commit", "chore(results): add b1", "results/runs/b1.csv", cwd=work)
    assert result.returncode == 0, result.stderr
    assert log(origin, "%s") == "chore(results): add b1"
    assert log(origin, "%P") != ""
    parent = subprocess.run(
        ["git", "log", "-1", "--format=%s", "main~1"], cwd=origin, capture_output=True, text=True
    )
    assert parent.stdout.strip() == "docs: x"


def test_commit_fails_without_pushing_when_the_same_file_changed_meanwhile(clone):
    work, other, origin = clone
    (other / "results/runs/b1.csv").write_text("theirs\n", encoding="utf-8")
    git("add", ".", cwd=other)
    git("-c", "user.name=X", "-c", "user.email=x@example.com", "commit", "-m", "fix: x", cwd=other)
    git("push", "origin", "HEAD:main", cwd=other)
    (work / "results/runs/b1.csv").write_text("ours\n", encoding="utf-8")
    result = steps("commit", "chore(results): add b1", "results/runs/b1.csv", cwd=work)
    assert result.returncode == 1
    assert "conflict" in result.stderr
    assert log(origin, "%s") == "fix: x"


def test_commit_with_push_dry_run_leaves_the_remote_alone(clone):
    work, _, origin = clone
    (work / "results/runs/b1.csv").write_text("v2\n", encoding="utf-8")
    result = steps("commit", "chore(results): add b1", "results/runs/b1.csv", cwd=work,
                   env={"PUSH_DRY_RUN": "1"})  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert log(origin).endswith("feat: first")
