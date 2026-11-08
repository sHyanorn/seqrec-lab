import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_env  # noqa: E402

CPU = torch.device("cpu")


def test_quick_run_writes_scrubbed_report(tmp_path):
    args = ["--quick", "--skip-compile", "--device", "cpu", "--out", str(tmp_path)]
    assert check_env.main(args) == 0

    (json_path,) = tmp_path.glob("check_env_*.json")
    assert json_path.with_suffix(".txt").exists()
    text = json_path.read_text(encoding="utf-8")
    report = json.loads(text)

    assert report["environment"]["packages"]["torch"].startswith("2.15")
    math_rows = [r for r in report["sdpa"] if r["backend"] == "math"]
    assert math_rows and all(r["ok"] and r["matches_reference"] for r in math_rows)
    assert {r["mask"] for r in report["sdpa"]} == set(check_env.MASK_MODES)
    train = report["model"]["train_step"]
    assert train and all(r["ok"] and r["tokens_per_s"] > 0 for r in train)
    assert report["thermal"]["tflops_per_window"]
    for token in check_env.sensitive_tokens():
        assert token.lower() not in text.lower()


def test_scrub_removes_paths_with_spaces_and_names():
    roots = check_env.sensitive_roots()
    tokens = ["LAPTOP-AB12", "John Smith"]
    raw = {
        "error": r"failed: C:\Users\John Smith\OneDrive - Contoso Ltd\seqrec-lab\x.py line 3",
        "stderr_tail": ["/home/jsmith/My Projects/run.log", "built on laptop-ab12"],
        "nan": float("nan"),
    }
    cleaned = check_env.scrub(raw, roots, tokens)
    text = json.dumps(cleaned, allow_nan=False).lower()
    for fragment in ("smith", "contoso", "onedrive", "projects", "laptop-ab12"):
        assert fragment not in text
    assert cleaned["nan"] is None


def test_short_user_name_leaves_keys_and_summary_intact(monkeypatch, tmp_path):
    monkeypatch.setenv("USER", "ian")
    monkeypatch.setenv("USERNAME", "ian")
    args = ["--quick", "--skip-compile", "--device", "cpu", "--out", str(tmp_path)]
    assert check_env.main(args) == 0
    (json_path,) = tmp_path.glob("check_env_*.json")
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert "median_ms" in report["matmul"][0]


def test_summary_survives_crashed_probes():
    report = {
        "mode": "full",
        "created_utc": "2026-01-01T00:00:00Z",
        "device": "cuda",
        "durations_s": {"environment": 0.1, "matmul": 0.1, "sdpa": 0.1},
        "environment": {"error": "Traceback ..."},
        "matmul": {"error": "Traceback ..."},
        "sdpa": {"error": "Traceback ..."},
        "model": {"error": "Traceback ..."},
        "compile": {"error": "Traceback ..."},
        "thermal": {"error": "Traceback ..."},
    }
    summary = check_env.build_summary(report)
    assert "Probes that crashed" in summary


def test_padding_masks_never_leave_a_query_without_keys():
    for mode in ("padding", "causal_padding"):
        mask, causal = check_env.attention_mask(mode, batch=8, length=12, device=CPU)
        assert not causal
        assert mask.shape == (8, 1, 12, 12)
        assert mask.any(dim=-1).all()
    mask, _ = check_env.attention_mask("causal_padding", batch=8, length=12, device=CPU)
    assert not torch.triu(mask[0, 0], diagonal=1).any()


@pytest.mark.skipif(sys.platform == "win32", reason="CPU inductor needs MSVC on Windows")
def test_compile_probe_runs_in_child_process():
    result = check_env.probe_compile(CPU, check_env.QUICK)
    assert result["ok"], result
    assert result["speedup"] > 0


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
def test_full_mode_without_gpu_stops_after_environment(tmp_path):
    assert check_env.main(["--out", str(tmp_path)]) == 1
    (json_path,) = tmp_path.glob("check_env_*.json")
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["environment"]["cuda_init_error"]
    assert "sdpa" not in report
