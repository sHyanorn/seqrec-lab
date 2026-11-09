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
    build_log(random_frame(users=600)).save(log_path(path, "ml-1m"))
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
    args = ["--dataset", "ml-1m", "--arm", "bert4rec", "--params", json.dumps(SMALL)]
    with pytest.raises(SystemExit):  # reported through the parser, not as a traceback
        profile_step.main([*args, "--data-dir", str(data), "--device", "cpu", "--fixed-shapes"])
    chunked, _ = model_and_batch(data, "sasrec-ce", {"ce_chunk": 64})
    with pytest.raises(ValueError, match="ce_chunk"):  # the masked loss is not chunked
        step_modes.Stepper(chunked, fixed_shapes=True)


def test_fixed_shape_gbce_scores_the_negatives_in_one_pass(data, monkeypatch):
    """Sliced, recomputed negative logits (D042) would add a second pass that the arm's own
    loss does not make below the slice bound, and would stash generator state inside a
    captured graph."""
    model, batch = model_and_batch(data, "gsasrec", {"negatives": 16})
    monkeypatch.setattr(losses, "GBCE_CHUNK_ELEMENTS", 64)  # every batch is beyond the bound

    def recomputed(*args, **kwargs):
        raise AssertionError("fixed shapes must not recompute the negative logits")

    monkeypatch.setattr(losses, "checkpoint", recomputed)
    assert torch.isfinite(step_modes.fixed_shape_loss(model, batch))


def modes_model(data, arm, modes, params=None):
    cfg = RunConfig("profile", "ml-1m", "loo", arm, 0, "tune", params=SMALL | (params or {}))
    return profile_step.build(cfg, data, torch.device("cpu"), modes)


def test_the_modes_reach_the_timed_epoch(data):
    model, _ = modes_model(data, "sasrec-bce", {"fixed_shapes": True, "fused_adam": True})
    stepper = model.train_epoch.__self__
    assert isinstance(stepper, step_modes.Stepper)
    assert stepper.loss_fn is step_modes.fixed_shape_loss
    assert model.opt.defaults["fused"] is True
    plain, _ = modes_model(data, "sasrec-bce", {})
    assert plain.train_epoch.__self__ is plain and not plain.opt.defaults["fused"]


@pytest.mark.parametrize("arm", step_modes.ARMS)
def test_the_stepper_without_modes_trains_as_the_arm_does(data, arm):
    """Two epochs with dropout and a partial last batch: the same losses and weights."""
    own, splits = modes_model(data, arm, {}, {"negatives": 16} if arm == "gsasrec" else None)
    other, _ = modes_model(data, arm, {}, {"negatives": 16} if arm == "gsasrec" else None)
    stepper = step_modes.Stepper(other)
    assert len(own.data.rows) % BATCH_SIZE
    results = []
    for train_epoch in (own.train_epoch, stepper.train_epoch):
        torch.manual_seed(0)  # dropout draws from the global generator
        results.append([train_epoch(splits.train, epoch) for epoch in range(2)])
    assert results[0] == results[1]
    for a, b in zip(own.net.parameters(), other.net.parameters(), strict=True):
        assert torch.equal(a, b)


def test_the_captured_batch_is_a_copy(data):
    """Later batches are copied into the captured one; a view of the model's rows there
    would overwrite the training-row index."""
    model, batch = model_and_batch(data, "sasrec-ce")
    static = step_modes.static_copy(batch)
    for name in ("rows", "inputs", "labels"):
        a, b = getattr(static, name), getattr(batch, name)
        assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
    before = model.data.rows.clone()
    static.rows.fill_(0)
    assert torch.equal(model.data.rows, before)


CHECK = ["--dataset", "ml-1m", "--arm", "sasrec-bce", "--params", json.dumps(SMALL), "--check"]


def run_check(data, capsys, *flags, arm="sasrec-bce"):
    args = [a if a != "sasrec-bce" else arm for a in CHECK]
    assert profile_step.main([*args, "--data-dir", str(data), "--device", "cpu", *flags]) == 0
    return json.loads(capsys.readouterr().out.splitlines()[-1])


def test_check_compares_several_batches_with_the_arms_own_code(data, capsys):
    checked = run_check(data, capsys, "--fixed-shapes", "--fused-adam")
    assert checked["check_reference"] == [] and len(checked["check_losses"]) == 3
    assert checked["check_loss_diff"] < 1e-5 and checked["check_weight_diff"] < 1e-5
    # the same reference twice: what two runs of one code path differ by
    assert checked["check_noise_loss_diff"] < 1e-5 and checked["check_noise_weight_diff"] < 1e-5


def test_check_shows_a_wrong_loss(data, capsys, monkeypatch):
    monkeypatch.setattr(step_modes, "masked_mean", lambda per_position, mask: per_position.mean())
    checked = run_check(data, capsys, "--fixed-shapes")
    assert checked["check_loss_diff"] > 1e-3 > checked["check_noise_loss_diff"]


def test_check_refuses_to_compare_a_reference_with_itself(data):
    run = ["--data-dir", str(data), "--device", "cpu"]
    with pytest.raises(SystemExit):  # no mode
        profile_step.main([*CHECK, *run])
    gsasrec = [a if a != "sasrec-bce" else "gsasrec" for a in CHECK]
    with pytest.raises(SystemExit):  # the gSASRec reference keeps fixed shapes
        profile_step.main([*gsasrec, *run, "--fixed-shapes"])


def test_check_reports_a_non_finite_weight():
    a = [torch.zeros(2), torch.zeros(2)]
    b = [torch.zeros(2), torch.tensor([0.0, float("nan")])]
    assert profile_step.largest_diff(a, a) == 0.0
    assert profile_step.largest_diff([torch.ones(2), *a], [torch.zeros(2), *b]) != pytest.approx(1)
    assert torch.isnan(torch.tensor(profile_step.largest_diff(a, b)))


def test_profile_step_times_the_modes(data, capsys):
    base = ["--dataset", "ml-1m", "--arm", "sasrec-bce", "--params", json.dumps(SMALL)]
    run = [*base, "--data-dir", str(data), "--device", "cpu", "--fixed-shapes", "--fused-adam"]
    assert profile_step.main([*run, "--steps", "2", "--warmup", "1"]) == 0
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["modes"] == ["fixed_shapes", "fused_adam"]
    assert result["loss"] > 0
