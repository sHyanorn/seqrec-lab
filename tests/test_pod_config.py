"""configs/cloud/pod.json, the request body a batch's Pod is created from."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "cloud" / "pod.json"


def pod() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_the_start_command_runs_the_watchdog_of_main_and_keeps_no_copy_on_the_volume():
    """A copy on the volume would be run by every later Pod, however old it had become. The
    jobs and the bootstrap look for the process this command starts."""
    command_line = " ".join(pod()["cmd"])
    fetched = "/run/seqrec-pod-start.sh"
    main = "https://github.com/sHyanorn/seqrec-lab/raw/main/scripts/cloud/pod_start.sh"
    assert f"curl -fsSL {main} -o {fetched} && bash {fetched} " in command_line
    assert "/workspace/seqrec/pod_start.sh" not in command_line
    for script in ("batch_steps.sh", "pod_bootstrap.sh"):
        text = (CONFIG.parents[2] / "scripts" / "cloud" / script).read_text(encoding="utf-8")
        assert f'"^bash ${{POD_START:-{fetched}}}\\$"' in text


def test_the_network_volume_is_mounted_where_the_scripts_and_the_workflow_expect_it():
    assert [mount["path"] for mount in pod()["mounts"]["network"]] == ["/workspace"]


def test_no_environment_variable_is_set():
    """The platform gives the Pod its API key; one written here would be a committed secret."""
    assert "env" not in pod()
