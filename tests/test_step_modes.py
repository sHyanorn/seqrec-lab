import json
import sys
from pathlib import Path

import pytest
import torch
from conftest import random_frame

from seqrec.data.preprocess import build_log
from seqrec.data.store import load_splits, log_path
from seqrec.models import batches, losses
from seqrec.models.trainer import BATCH_SIZE
from seqrec.runner.config import RunConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import profile_step  # noqa: E402
import step_modes  # noqa: E402

SMALL = {"dim": 16, "blocks": 1, "heads": 1}


@pytest.fixture
def data(tmp_path):
    path = tmp_path / "data"
    build_log(random_frame(users=300)).save(log_path(path, "ml-1m"))
    load_splits.cache_clear()
    yield path
    load_splits.cache_clear()


def model_and_batch(data, arm, params=None):
    cfg = RunConfig("profile", "ml-1m", "loo", arm, 0, "tune", params=SMALL | (params or {}))
    model, _ = profile_step.build(cfg, data, torch.device("cpu"), {})
    model.net.eval()  # no dropout: both losses see the same network output
    return model, model.make_batch(model.data.rows[:BATCH_SIZE], model.gen)


@pytest.mark.parametrize("arm", ["sasrec-bce", "sasrec-ce", "sasrec-ss"])
def test_fixed_shapes_give_the_arms_own_loss(data, arm):
    """The masked loss over all positions equals the loss over labelled positions, drawing
    the same negatives from the same generator state."""
    model, batch = model_and_batch(data, arm)
    state = model.gen.get_state()
    want = model.loss(batch)
    model.gen.set_state(state)
    got = step_modes.fixed_shape_loss(model, batch)
    assert got.item() == pytest.approx(want.item(), rel=1e-6)


def test_fixed_shapes_give_gbce_over_the_labelled_positions(data):
    model, batch = model_and_batch(data, "gsasrec", {"negatives": 16})
    state = model.gen.get_state()
    got = step_modes.fixed_shape_loss(model, batch)
    model.gen.set_state(state)
    labels = batch.labels.flatten()
    neg = batches.uniform_negatives(model.n_items, (len(labels), 16), model.gen, model.device)
    labelled = labels > 0
    h = model.net(batch.inputs).flatten(0, 1)[labelled]
    want = losses.gbce(h, model.net.output_table(), labels[labelled], neg[labelled], 0.75,
                       model.n_items)  # fmt: skip
    assert got.item() == pytest.approx(want.item(), rel=1e-6)


def test_modes_are_refused_where_they_do_not_apply(data):
    model, _ = model_and_batch(data, "sasrec-ce")
    with pytest.raises(ValueError, match="CUDA device"):
        step_modes.Stepper(model, fixed_shapes=True, cuda_graph=True)
    cfg = RunConfig("profile", "ml-1m", "loo", "bert4rec", 0, "tune", params=SMALL)
    bert, _ = profile_step.build(cfg, data, torch.device("cpu"), {})
    with pytest.raises(ValueError, match="cover"):
        step_modes.Stepper(bert, fixed_shapes=True)
    with pytest.raises(SystemExit):
        profile_step.main(["--dataset", "ml-1m", "--arm", "gsasrec", "--cuda-graph"])


def test_profile_step_times_the_modes_and_checks_them(data, capsys):
    base = ["--dataset", "ml-1m", "--arm", "sasrec-bce", "--params", json.dumps(SMALL)]
    run = [*base, "--data-dir", str(data), "--device", "cpu", "--fixed-shapes", "--fused-adam"]
    assert profile_step.main([*run, "--steps", "2", "--warmup", "1"]) == 0
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["modes"] == ["fixed_shapes", "fused_adam"]
    assert profile_step.main([*run, "--check"]) == 0
    checked = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert checked["check_reference"] == []
    assert checked["check_loss_diff"] < 1e-5 and checked["check_weight_diff"] < 1e-5
