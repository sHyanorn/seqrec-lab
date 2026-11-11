"""The batch workflow file: who can start it and how its inputs reach the shell."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
BATCH = yaml.safe_load((WORKFLOWS / "batch.yml").read_text(encoding="utf-8"))
JOB = BATCH["jobs"]["batch"]
STEPS = {s["name"]: s for s in JOB["steps"] if "name" in s}


HOSTED = re.compile(r"(ubuntu|windows|macos)-[a-z0-9.-]+")


def test_a_workflow_that_can_reach_the_machine_starts_by_hand_only():
    """A push or pull-request trigger would let a stranger's code run as root on the machine.
    Any job not on a GitHub-hosted image counts as able to reach it."""
    for path in WORKFLOWS.glob("*.y*ml"):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        hosted = all(
            isinstance(job.get("runs-on"), str) and HOSTED.fullmatch(job["runs-on"])
            for job in workflow["jobs"].values()
        )
        if not hosted:  # YAML reads the key `on` as True
            assert list(workflow[True]) == ["workflow_dispatch"], path.name


def test_the_job_runs_on_the_one_machine_and_from_main_only():
    assert JOB["runs-on"] == ["self-hosted", "linux", "x64", "gpu"]
    assert JOB["if"] == "github.ref == 'refs/heads/main'"
    assert "label" not in BATCH[True]["workflow_dispatch"]["inputs"]


def test_no_input_is_pasted_into_a_script():
    """Inputs reach the shell as environment variables, which batch_steps.sh checks."""
    for step in JOB["steps"]:
        assert "${{" not in step.get("run", ""), step.get("name")
    assert all(re.fullmatch(r"\$\{\{ inputs\.\w+ \}\}", BATCH["env"][k])
               for k in ("TASK", "PLAN", "HOURS", "PROFILE_ARGS", "REPEATS", "CPU"))  # fmt: skip


def test_every_job_requires_the_registered_host_unless_told_otherwise():
    """METHODS 9.4: every batch runs on a Ryzen 9 7950X. The machine step checks the model
    before any work, so a job queued behind the first needs no separate probe."""
    assert BATCH[True]["workflow_dispatch"]["inputs"]["cpu"]["default"] == "Ryzen 9 7950X"
    names = [s.get("name") for s in JOB["steps"]]
    assert STEPS["Machine"]["run"].strip() == "bash scripts/cloud/batch_steps.sh machine"
    assert all(names.index("Machine") < names.index(n) for n in ("Setup", "Profile", "Plan"))


def test_actions_are_pinned_to_a_commit():
    for step in JOB["steps"]:
        if "uses" in step:
            assert re.fullmatch(r"[\w-]+/[\w-]+@[0-9a-f]{40}", step["uses"]), step["uses"]


def test_the_token_is_given_to_reporting_and_commit_steps_only():
    with_token = {s["name"] for s in JOB["steps"] if "GH_TOKEN" in s.get("env", {})}
    assert with_token == {
        "Report start", "Commit the environment report", "Plan", "Commit results", "Bootstrap",
        "Report end",
    }  # fmt: skip
    checkout = next(s for s in JOB["steps"] if "uses" in s)
    assert checkout["with"]["persist-credentials"] is False


def test_the_export_task_exports_and_commits_without_running_the_plan():
    assert "export" in BATCH[True]["workflow_dispatch"]["inputs"]["task"]["options"]
    assert "export" not in STEPS["Plan"]["if"]
    for name in ("Export", "Commit results"):
        assert STEPS[name]["if"] == "steps.plan.outcome == 'success' || inputs.task == 'export'"


def test_the_bootstrap_task_runs_its_own_step_and_no_plan():
    """METHODS 7.6 from the ranks a tested batch keeps on the volume (D056)."""
    assert "bootstrap" in BATCH[True]["workflow_dispatch"]["inputs"]["task"]["options"]
    assert STEPS["Bootstrap"]["if"] == "inputs.task == 'bootstrap'"
    assert STEPS["Bootstrap"]["run"].strip() == "bash scripts/cloud/batch_steps.sh bootstrap"
    for name in ("Plan", "Export", "Commit results"):
        assert "bootstrap" not in STEPS[name]["if"]
    assert "bootstrap" in STEPS["Report start"]["if"]


def test_the_machine_is_asked_to_stop_whatever_happened():
    assert STEPS["Request stop"]["if"] == "always() && inputs.stop"
    assert JOB["steps"][-1]["name"] == "Request stop"


def test_results_are_committed_through_the_tested_step():
    assert STEPS["Commit results"]["run"].strip() == "bash scripts/cloud/batch_steps.sh results"


def test_the_watchdog_is_checked_and_the_export_is_fresh():
    names = [s.get("name") for s in JOB["steps"]]
    assert STEPS["Watchdog"]["run"].strip() == "bash scripts/cloud/batch_steps.sh watchdog-running"
    assert names.index("Watchdog") < names.index("Sync")
    assert STEPS["Export"]["run"].strip() == "bash scripts/cloud/batch_steps.sh export"


def test_recorded_test_reads_are_checked_before_any_gpu_work():
    """Batch files on main that this volume did not write stop the job before it trains."""
    name = "Recorded test reads (on failure, move results/runs/* to results/<name>/)"
    step = STEPS[name]
    assert step["if"] == 'contains(fromJSON(\'["run", "tune", "test", "export"]\'), inputs.task)'
    assert step["run"].strip() == (
        'uv run --no-sync seqrec export --check-reads --runs-dir "$SEQREC_HOME/runs"'
    )
    names = [s.get("name") for s in JOB["steps"]]
    assert names.index(name) == names.index("Sync") + 1
