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


@pytest.mark.parametrize("task", ["run", "tune", "test", "export"])
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


def test_the_test_task_has_no_hour_limit_to_check():
    env = {"TASK": "test", "PLAN": "m2-anchor.toml", "HOURS": ""}
    assert steps("check-inputs", env=env).returncode == 0


def test_the_export_task_needs_a_plan_and_no_hour_limit():
    env = {"TASK": "export", "PLAN": "m2-anchor.toml", "HOURS": ""}
    assert steps("check-inputs", env=env).returncode == 0
    result = steps("check-inputs", env={"TASK": "export", "PLAN": ""})
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
    os.utime(path, (mtime, mtime))


def test_progress_describes_the_run_that_finished_last(tmp_path):
    fake(tmp_path / "bin", "curl", RECORD_POSTS.format(tmp_path))
    fake(tmp_path / "bin", "uv", "true")
    runs = tmp_path / "home" / "runs" / "m2-anchor"
    summary(runs / "older", "max_epochs", 200, 100.4, 1_000)
    summary(runs / "later", "patience", 384, 3868.2, 2_000)
    (runs / "running").mkdir()
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
    'echo "$@" >> {0}/uv-args; [ -z "${{UV_FAIL:-}}" ] || exit 1; '
    'dir="${{@: -1}}"; (cd "$dir" && ls | grep -v overview > overview.csv)'
)


def results_env(work):
    plans = work / "configs" / "plans"
    plans.mkdir(parents=True)
    (plans / "b1.toml").write_text('batch = "b1"\n', encoding="utf-8")
    (work / "results/runs/b1.csv").write_text("run_id,x\nr1,v2\n", encoding="utf-8")
    return {"TASK": "run", "PLAN": "b1.toml"}


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
    args = (tmp_path / "uv-args").read_text(encoding="utf-8").split()
    assert args[:6] == ["run", "--no-sync", "seqrec", "export", "--summaries-only", "--out-dir"]
    assert args[6].endswith("/main/results/runs") and not args[6].startswith(str(work))


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


# --- watchdog-running and export ----------------------------------------------------------


def test_a_job_refuses_a_machine_whose_watchdog_is_not_running(tmp_path):
    """Without the watchdog nothing stops the machine; it holds this lock while it runs."""
    import fcntl
    import time

    env = {"POD_START_LOCK": str(tmp_path / "lock"), "SEQREC_HOME": str(tmp_path)}
    result = steps("watchdog-running", env=env)
    assert result.returncode == 1
    assert "watchdog is not running" in result.stderr
    with (tmp_path / "lock").open("w") as other:  # held, but not by the watchdog
        fcntl.flock(other, fcntl.LOCK_EX)
        result = steps("watchdog-running", env=env)
    assert result.returncode == 1
    assert "not by the watchdog" in result.stderr
    fake(tmp_path, "pod_start.sh", f"exec 9> {tmp_path}/lock; flock 9; sleep 5 9>&-")
    watchdog = subprocess.Popen([str(tmp_path / "pod_start.sh")])
    try:
        time.sleep(0.3)
        assert steps("watchdog-running", env=env).returncode == 0
    finally:
        watchdog.kill()


def test_export_leaves_only_a_batch_file_written_from_this_machine(clone, tmp_path):
    """A file that came with the checkout must not pass for this machine's export."""
    work, _, _ = clone
    env = results_env(work) | {"SEQREC_HOME": str(tmp_path / "home")}
    fake(tmp_path / "bin", "uv", f'echo "$@" > {tmp_path}/uv-args')  # writes no batch file
    result = steps("export", cwd=work, env=env, path_prefix=tmp_path / "bin")
    assert result.returncode == 1
    assert "no run of b1" in result.stderr
    assert (tmp_path / "uv-args").read_text(encoding="utf-8").split() == [
        "run", "--no-sync", "seqrec", "export", "--runs-dir", f"{tmp_path}/home/runs",
        "--plan", "configs/plans/b1.toml",
    ]  # fmt: skip
    fake(tmp_path / "bin", "uv", "echo run_id > results/runs/b1.csv")
    assert steps("export", cwd=work, env=env, path_prefix=tmp_path / "bin").returncode == 0
    fake(tmp_path / "bin", "uv", "echo run_id > results/runs/b1.csv; exit 1")
    assert steps("export", cwd=work, env=env, path_prefix=tmp_path / "bin").returncode == 1


def test_a_lock_that_cannot_be_checked_does_not_pass_for_a_running_watchdog(tmp_path):
    result = steps("watchdog-running", env={"POD_START_LOCK": str(tmp_path / "missing" / "lock")})
    assert result.returncode == 1
    assert "cannot check" in result.stderr


def test_the_summaries_are_computed_without_the_push_token(clone, tmp_path):
    """The summaries run repository code; the token is for the push alone."""
    work, _, origin = clone
    fake(tmp_path / "bin", "uv", f"env > {tmp_path}/uv-env; " + SUMMARIZE.format(tmp_path))
    env = results_env(work) | {"GH_TOKEN": "s3cret"}
    result = steps("results", cwd=work, env=env, path_prefix=tmp_path / "bin")
    assert result.returncode == 0, result.stderr
    assert log(origin, "%s") == "chore(results): add run b1 results"
    seen = (tmp_path / "uv-env").read_text(encoding="utf-8")
    assert "s3cret" not in seen and "AUTHORIZATION" not in seen
