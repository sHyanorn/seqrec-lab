import numpy as np
import pytest
import torch
from conftest import random_frame

from seqrec.data.preprocess import build_log
from seqrec.data.store import build_splits
from seqrec.models import bert4rec
from seqrec.runner.config import RunConfig
from seqrec.runner.run import ARMS

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def train():
    splits, _ = build_splits(build_log(random_frame()), "loo")
    return splits.train


def model(train, **params):
    torch.manual_seed(0)
    cfg = RunConfig("b", "ml-1m", "loo", "bert4rec", 1, "tune", params=params, max_len=8)
    return ARMS["bert4rec"](cfg, train, CPU)


def test_block_matches_the_pytorch_post_ln_layer():
    torch.manual_seed(0)
    dim, heads = 16, 2
    ours = bert4rec.Block(dim, heads, 0.0).eval()
    ref = torch.nn.TransformerEncoderLayer(
        dim, heads, 4 * dim, 0.0, activation="gelu", layer_norm_eps=1e-12, batch_first=True
    ).eval()
    with torch.no_grad():
        for p in ours.parameters():
            torch.nn.init.normal_(p, std=0.3)
        ref.self_attn.in_proj_weight.copy_(
            torch.cat([ours.attn.query.weight, ours.attn.key_value.weight])
        )
        ref.self_attn.in_proj_bias.copy_(
            torch.cat([ours.attn.query.bias, ours.attn.key_value.bias])
        )
        ref.self_attn.out_proj.load_state_dict(ours.attn.out.state_dict())
        ref.norm1.load_state_dict(ours.attn_norm.state_dict())
        ref.linear1.load_state_dict(ours.ffn[0].state_dict())
        ref.linear2.load_state_dict(ours.ffn[2].state_dict())
        ref.norm2.load_state_dict(ours.ffn_norm.state_dict())
    x = torch.randn(3, 6, dim)
    real = torch.ones(3, 6, dtype=torch.bool)
    real[1, :2] = False
    ref.train()  # keep the reference off its inference fast path
    with torch.no_grad():
        expected = ref(x, src_key_padding_mask=~real)
        got = ours(x, real[:, None, None, :])
    torch.testing.assert_close(got[real], expected[real], atol=1e-5, rtol=1e-5)


def test_scores_read_an_appended_mask_token(train):
    m = model(train, dim=16, heads=2)
    inputs = torch.tensor([[0, 0, 0, 0, 3, 5, 7, 9], [1, 2, 3, 4, 5, 6, 7, 8]])
    shown = m.net.with_mask_token(inputs)
    assert shown.tolist() == [[0, 0, 0, 3, 5, 7, 9, 41], [2, 3, 4, 5, 6, 7, 8, 41]]
    scores = m.score(inputs)
    assert scores.shape == (2, train.n_items + 2) and scores.dtype == torch.float32
    with torch.no_grad():
        h = m.net(shown)[:, -1]
    torch.testing.assert_close(scores, h @ m.net.item.weight.T + m.net.bias)


def test_scores_do_not_depend_on_the_rest_of_the_batch(train):
    m = model(train, dim=16, heads=2)
    inputs = torch.randint(1, train.n_items + 1, (5, 8))
    inputs[2, :5] = 0
    alone = m.score(inputs[2:3])
    torch.testing.assert_close(m.score(inputs)[2:3], alone, atol=1e-5, rtol=1e-5)


def test_trains_with_finite_loss(train):
    m = model(train, dim=16, heads=2, mask_prob=0.3)
    assert np.isfinite(m.train_epoch(train, 0)["loss"])
    assert m.hparams["mask_prob"] == 0.3
