"""The batch workflow file: who can start it and how its inputs reach the shell."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
BATCH = yaml.safe_load((WORKFLOWS / "batch.yml").read_text(encoding="utf-8"))
JOB = BATCH["jobs"]["batch"]
STEPS = {s["name"]: s for s in JOB["steps"] if "name" in s}


def test_a_workflow_that_uses_the_machine_starts_by_hand_only():
    """A push or pull-request trigger would let a stranger's code run as root on the machine."""
    for path in WORKFLOWS.glob("*.y*ml"):
        text = path.read_text(encoding="utf-8")
        if "self-hosted" in text:
            triggers = yaml.safe_load(text)[True]  # YAML reads the key `on` as True
            assert list(triggers) == ["workflow_dispatch"], path.name


def test_the_job_runs_on_the_one_machine_and_from_main_only():
    assert JOB["runs-on"] == ["self-hosted", "linux", "x64", "gpu"]
    assert JOB["if"] == "github.ref == 'refs/heads/main'"
    assert "label" not in BATCH[True]["workflow_dispatch"]["inputs"]


def test_no_input_is_pasted_into_a_script():
    """Inputs reach the shell as environment variables, which batch_steps.sh checks."""
    for step in JOB["steps"]:
        assert "${{" not in step.get("run", ""), step.get("name")
    assert all(re.fullmatch(r"\$\{\{ inputs\.\w+ \}\}", BATCH["env"][k])
               for k in ("TASK", "PLAN", "HOURS", "PROFILE_ARGS", "REPEATS"))  # fmt: skip


def test_actions_are_pinned_to_a_commit():
    for step in JOB["steps"]:
        if "uses" in step:
            assert re.fullmatch(r"[\w-]+/[\w-]+@[0-9a-f]{40}", step["uses"]), step["uses"]


def test_the_token_is_given_to_reporting_and_commit_steps_only():
    with_token = {s["name"] for s in JOB["steps"] if "GH_TOKEN" in s.get("env", {})}
    assert with_token == {
        "Report start", "Commit the environment report", "Plan", "Commit results", "Report end",
    }  # fmt: skip
    checkout = next(s for s in JOB["steps"] if "uses" in s)
    assert checkout["with"]["persist-credentials"] is False


def test_the_export_task_exports_and_commits_without_running_the_plan():
    assert "export" in BATCH[True]["workflow_dispatch"]["inputs"]["task"]["options"]
    assert "export" not in STEPS["Plan"]["if"]
    for name in ("Export", "Commit results"):
        assert STEPS[name]["if"] == "steps.plan.outcome == 'success' || inputs.task == 'export'"


def test_the_machine_is_asked_to_stop_whatever_happened():
    assert STEPS["Request stop"]["if"] == "always() && inputs.stop"
    assert JOB["steps"][-1]["name"] == "Request stop"
