from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import random_frame
from reference.pmixer_sasrec import SASRec as PmixerSASRec

from seqrec.data.preprocess import build_log
from seqrec.data.store import build_splits
from seqrec.models import sasrec
from seqrec.runner.config import RunConfig
from seqrec.runner.run import ARMS

CPU = torch.device("cpu")
SASREC_ARMS = ("sasrec-bce", "sasrec-ce", "sasrec-ss", "gsasrec")


@pytest.fixture(scope="module")
def train():
    splits, _ = build_splits(build_log(random_frame()), "loo")
    return splits.train


def cfg(arm="sasrec-ce", **params):
    return RunConfig("b", "ml-1m", "loo", arm, 1, "tune", params=params, max_len=8)


def net(n_items=30, length=8, dim=16, blocks=2, heads=2, shared=True):
    torch.manual_seed(0)
    m = sasrec.SASRec(n_items, length, dim, blocks, heads, 0.0, shared)
    for p in m.parameters():  # LayerNorm defaults would hide a swapped norm
        torch.nn.init.normal_(p, std=0.3)
    return m.eval()


def assert_causal(model, ids: torch.Tensor, t: int) -> None:
    """Outputs at positions <= t do not depend on inputs after t."""
    changed = ids.clone()
    changed[:, t + 1 :] = torch.randint(1, 31, changed[:, t + 1 :].shape)
    with torch.no_grad():
        before, after = model(ids), model(changed)
    torch.testing.assert_close(before[:, : t + 1], after[:, : t + 1], atol=1e-6, rtol=0)
    assert not torch.allclose(before[:, t + 1 :], after[:, t + 1 :])


def test_outputs_are_causal():
    ids = torch.randint(1, 31, (4, 8))
    ids[1, :3] = 0  # left padding
    for t in (0, 3, 6):
        assert_causal(net(), ids, t)


def test_attention_matches_multihead_attention():
    torch.manual_seed(1)
    dim, heads, n = 16, 2, 6
    ours = sasrec.Attention(dim, heads, 0.0).eval()
    ref = torch.nn.MultiheadAttention(dim, heads, batch_first=True).eval()
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([ours.query.weight, ours.key_value.weight]))
        ref.in_proj_bias.copy_(torch.cat([ours.query.bias, ours.key_value.bias]))
        ref.out_proj.weight.copy_(ours.out.weight)
        ref.out_proj.bias.copy_(ours.out.bias)
    q, kv = torch.randn(3, n, dim), torch.randn(3, n, dim)
    real = torch.ones(3, n, dtype=torch.bool)
    allowed = sasrec.attention_mask(real)
    expected, _ = ref(q, kv, kv, attn_mask=~allowed[0, 0], need_weights=False)
    torch.testing.assert_close(ours(q, kv, allowed), expected, atol=1e-6, rtol=0)


def test_attention_mask_blocks_padding_keys_and_keeps_rows_non_empty():
    real = torch.tensor([[False, False, True, True]])
    expected = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]], dtype=torch.bool
    )
    assert torch.equal(sasrec.attention_mask(real)[0, 0], expected)
    attn = sasrec.Attention(8, 2, 0.0).eval()
    q, kv = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    other = kv.clone()
    other[:, :2] = torch.randn(1, 2, 8)  # padded keys change
    allowed = sasrec.attention_mask(real)
    out = attn(q, kv, allowed)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:, 2:], attn(q, other, allowed)[:, 2:])


def test_forward_matches_the_pmixer_reference_without_padding():
    ours = net(dim=16, heads=2)
    length = 8
    args = SimpleNamespace(
        device="cpu", hidden_units=16, maxlen=length, dropout_rate=0.0, num_heads=2, num_blocks=2
    )
    ref = PmixerSASRec(1, 30, args).eval()
    with torch.no_grad():
        ref.item_emb.weight.copy_(ours.item.weight)
        ref.pos_emb.weight[1:].copy_(ours.position.weight)  # pmixer counts positions from 1
        for i, block in enumerate(ours.blocks):
            ref.attention_layernorms[i].load_state_dict(block.attn_norm.state_dict())
            mha = ref.attention_layers[i]
            mha.in_proj_weight.copy_(
                torch.cat([block.attn.query.weight, block.attn.key_value.weight])
            )
            mha.in_proj_bias.copy_(torch.cat([block.attn.query.bias, block.attn.key_value.bias]))
            mha.out_proj.load_state_dict(block.attn.out.state_dict())
            ref.forward_layernorms[i].load_state_dict(block.ffn_norm.state_dict())
            ffn = ref.forward_layers[i]
            ffn.conv1.weight.copy_(block.ffn[0].weight[:, :, None])
            ffn.conv1.bias.copy_(block.ffn[0].bias)
            ffn.conv2.weight.copy_(block.ffn[3].weight[:, :, None])
            ffn.conv2.bias.copy_(block.ffn[3].bias)
        ref.last_layernorm.load_state_dict(ours.norm.state_dict())
    ids = np.random.default_rng(0).integers(1, 31, (5, length))
    with torch.no_grad():
        expected = ref.log2feats(ids)
        got = ours(torch.from_numpy(ids))
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("arm", SASREC_ARMS)
def test_each_arm_trains_and_scores_in_fp32(arm, train):
    params = {"negatives": 16} if arm in ("sasrec-ss", "gsasrec") else {}
    model = ARMS[arm](cfg(arm, dim=16, **params), train, CPU)
    stats = model.train_epoch(train, 0)
    assert np.isfinite(stats["loss"])
    scores = model.score(torch.randint(0, train.n_items + 1, (3, 8)))
    assert scores.shape == (3, train.n_items + 1) and scores.dtype == torch.float32


def test_chunked_ce_trains_like_plain_ce(train):
    weights = []
    for chunk in (0, 7):
        torch.manual_seed(0)  # the runner seeds before building a model
        model = ARMS["sasrec-ce"](cfg(dim=16, dropout=0.0, ce_chunk=chunk), train, CPU)
        model.train_epoch(train, 0)
        weights.append(model.net.state_dict())
    for k in weights[0]:
        torch.testing.assert_close(weights[0][k], weights[1][k], atol=1e-5, rtol=1e-4)


def test_state_dict_is_a_snapshot_and_resumes_the_same_stream(train):
    model = ARMS["sasrec-bce"](cfg("sasrec-bce", dim=16), train, CPU)
    model.train_epoch(train, 0)
    saved, rng = model.state_dict(), torch.get_rng_state()
    frozen = {k: v.clone() for k, v in saved["net"].items()}
    model.train_epoch(train, 1)
    assert all(torch.equal(frozen[k], saved["net"][k]) for k in frozen)
    resumed = ARMS["sasrec-bce"](cfg("sasrec-bce", dim=16), train, CPU)
    resumed.load_state_dict(saved)
    torch.set_rng_state(rng)
    resumed.train_epoch(train, 1)
    for k, v in model.net.state_dict().items():
        assert torch.equal(v, resumed.net.state_dict()[k]), k


def test_unknown_hyperparameters_are_rejected(train):
    with pytest.raises(ValueError, match="unknown hyperparameters"):
        ARMS["sasrec-ce"](cfg(negatives=5), train, CPU)
    with pytest.raises(ValueError, match="divisible"):
        ARMS["sasrec-ce"](cfg(dim=10, heads=3), train, CPU)


def test_left_padding_does_not_change_real_positions():
    model = net()
    items = torch.tensor([[4, 9, 2, 7, 5]])
    with torch.no_grad():
        long = model(torch.cat([torch.zeros(1, 3, dtype=torch.long), items], 1))[:, 3:]
        short = model(torch.cat([torch.zeros(1, 1, dtype=torch.long), items], 1))[:, 1:]
    torch.testing.assert_close(long, short, atol=1e-6, rtol=0)
