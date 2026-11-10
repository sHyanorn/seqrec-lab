"""configs/cloud/pod.json, the request body a batch's Pod is created from."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "cloud" / "pod.json"


def pod() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_the_start_command_is_the_loop_the_bootstrap_checks_for():
    """pod_bootstrap.sh looks for this text in the command line of a running process."""
    command_line = " ".join(pod()["cmd"])
    assert "while true; do [ -x /workspace/seqrec/pod_start.sh" in command_line


def test_the_network_volume_is_mounted_where_the_scripts_and_the_workflow_expect_it():
    assert [mount["path"] for mount in pod()["mounts"]["network"]] == ["/workspace"]


def test_no_environment_variable_is_set():
    """The platform gives the Pod its API key; one written here would be a committed secret."""
    assert "env" not in pod()
