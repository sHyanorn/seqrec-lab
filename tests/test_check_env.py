import json
import sys
from pathlib import Path

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


def test_scrub_removes_names_and_paths():
    tokens = ["LAPTOP-AB12", "alice"]
    raw = {
        "error": r"failed in C:\Users\Alice\AppData\Local\Temp\x.py on laptop-ab12",
        "nested": ["/home/alice/run.log", "alice"],
        "alice": 1,
    }
    cleaned = json.dumps(check_env.scrub(raw, tokens)).lower()
    assert "alice" not in cleaned
    assert "laptop-ab12" not in cleaned
    assert "<path>" in cleaned


def test_padding_masks_never_leave_a_query_without_keys():
    for mode in ("padding", "causal_padding"):
        mask, causal = check_env.attention_mask(mode, batch=8, length=12, device=CPU)
        assert not causal
        assert mask.shape == (8, 1, 12, 12)
        assert mask.any(dim=-1).all()
    mask, _ = check_env.attention_mask("causal_padding", batch=8, length=12, device=CPU)
    assert not torch.triu(mask[0, 0], diagonal=1).any()


def test_compile_probe_runs_in_child_process():
    result = check_env.probe_compile(torch.device("cpu"), check_env.QUICK)
    assert result["ok"], result
    assert result["speedup"] > 0
