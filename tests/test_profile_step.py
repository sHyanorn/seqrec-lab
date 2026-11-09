import sys
from pathlib import Path

from conftest import random_frame

from seqrec.data.preprocess import build_log
from seqrec.data.store import load_splits, log_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import profile_step  # noqa: E402


def test_profile_step_reports_time_per_step(tmp_path, capsys):
    data = tmp_path / "data"
    build_log(random_frame(users=300)).save(log_path(data, "ml-1m"))
    load_splits.cache_clear()
    args = ["--dataset", "ml-1m", "--arm", "sasrec-bce", "--steps", "2", "--warmup", "1"]
    assert profile_step.main([*args, "--data-dir", str(data), "--device", "cpu"]) == 0
    load_splits.cache_clear()
    out = capsys.readouterr().out
    for field in ("wall_ms_per_step", "cpu_ms_per_step", "gpu_kernel_ms_per_step", "steps 2"):
        assert field in out
    assert not list(tmp_path.glob("runs*"))  # writes nothing
