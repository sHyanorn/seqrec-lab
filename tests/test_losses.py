import math

import pytest
import torch
import torch.nn.functional as F

from seqrec.models import losses

torch.manual_seed(0)


def tensors(m=10, n=7, d=4, requires_grad=False):
    h = torch.randn(m, d, dtype=torch.float64, requires_grad=requires_grad)
    w = torch.randn(n + 2, d, dtype=torch.float64, requires_grad=requires_grad)  # 0, items, mask
    y = torch.randint(1, n + 1, (m,))
    return h, w, y


def check_full_ce_matches_naive_softmax_over_items_only():
    h, w, y = tensors()
    bias = torch.randn(9, dtype=torch.float64)
    w[0] = 50.0  # padding and mask rows must not enter the softmax
    w[-1] = -50.0
    for b in (None, bias):
        naive = []
        for i in range(len(y)):
            scores = [float(h[i] @ w[j] + (0 if b is None else b[j])) for j in range(1, 8)]
            top = max(scores)
            log_z = top + math.log(sum(math.exp(s - top) for s in scores))
            naive.append(log_z - scores[y[i] - 1])
        got = losses.full_ce(h, w, y, bias=b, n_items=7)
        assert got.item() == pytest.approx(sum(naive) / len(naive), abs=1e-6)


def test_full_ce_matches_naive_softmax_over_items_only():
    check_full_ce_matches_naive_softmax_over_items_only()


def test_chunked_ce_has_the_same_loss_and_gradients():
    h, w, y = tensors(m=11, requires_grad=True)
    bias = torch.randn(9, dtype=torch.float64, requires_grad=True)
    grads = []
    for chunk in (0, 3):
        loss = losses.full_ce(h, w, y, bias=bias, n_items=7, chunk=chunk)
        grads.append((loss.detach(), *torch.autograd.grad(loss, (h, w, bias))))
    for a, b in zip(*grads, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=0)


def check_sampled_softmax_drops_negatives_equal_to_the_positive():
    h, w, y = tensors(m=3)
    y = torch.tensor([2, 5, 2])
    neg = torch.tensor([2, 4, 6])  # equals the positive of rows 0 and 2
    got = losses.sampled_softmax(h, w, y, neg)
    naive = []
    for i in range(3):
        kept = [j for j in neg.tolist() if j != y[i]]
        logits = torch.stack([h[i] @ w[y[i]]] + [h[i] @ w[j] for j in kept])
        naive.append(-torch.log_softmax(logits, 0)[0])
    assert got.item() == pytest.approx(torch.stack(naive).mean().item(), abs=1e-9)


def test_sampled_softmax_drops_negatives_equal_to_the_positive():
    check_sampled_softmax_drops_negatives_equal_to_the_positive()


def test_bce_is_the_sasrec_code_formula():
    h, w, y = tensors()
    neg = torch.randint(1, 8, y.shape)
    pos, ng = (h * w[y]).sum(-1), (h * w[neg]).sum(-1)
    naive = -(torch.log(torch.sigmoid(pos)) + torch.log(1 - torch.sigmoid(ng))).mean()
    assert losses.bce(h, w, y, neg).item() == pytest.approx(naive.item(), abs=1e-9)


def test_gbce_beta():
    assert losses.gbce_beta(256, 3416, 0.75) == pytest.approx(1 - 0.75 * (1 - 256 / 3415))
    assert losses.gbce_beta(256, 3416, 0.75) == pytest.approx(0.30622, abs=1e-5)
    assert losses.gbce_beta(64, 1000, 0.0) == pytest.approx(1.0)


def reference_gbce(h, w, y, neg, t, n_items):
    """The transformed-logit form of gSASRec [4]: sigma(gamma) = sigma(s+)^beta, then binary
    cross-entropy averaged over the positive and K negatives (as in [33])."""
    k = neg.shape[1]
    beta = losses.gbce_beta(k, n_items, t)
    pos = (h * w[y]).sum(-1, keepdim=True)
    gamma = torch.log(1 / (torch.sigmoid(pos) ** (-beta) - 1))
    logits = torch.cat([gamma, torch.einsum("md,mkd->mk", h, w[neg])], 1)
    labels = torch.zeros_like(logits)
    labels[:, 0] = 1
    return F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(1).mean()


def test_gbce_matches_the_reference_form_without_collisions():
    h, w, y = tensors(m=20, n=50)
    y = torch.randint(1, 26, (20,))
    neg = torch.randint(26, 51, (20, 8))  # never equal to the positive
    for t in (0.0, 0.5, 0.75, 1.0):
        got = losses.gbce(h, w, y, neg, t=t, n_items=50)
        assert got.item() == pytest.approx(reference_gbce(h, w, y, neg, t, 50).item(), abs=1e-9)


def test_gbce_with_t_zero_and_one_negative_is_half_the_bce():
    h, w, y = tensors()
    neg = (y % 7) + 1  # differs from y
    got = losses.gbce(h, w, y, neg[:, None], t=0.0, n_items=7)
    assert got.item() == pytest.approx(losses.bce(h, w, y, neg).item() / 2, abs=1e-9)


def test_gbce_drops_negatives_equal_to_the_positive():
    h, w, y = tensors(m=2, n=10)
    y = torch.tensor([3, 4])
    neg = torch.tensor([[3, 5, 6], [7, 8, 9]])
    beta = losses.gbce_beta(3, 10, 0.5)
    pos = (h * w[y]).sum(-1)
    kept = [(5, 6), (7, 8, 9)]
    terms = [
        beta * F.softplus(-pos[i]) + sum(F.softplus(h[i] @ w[j]) for j in kept[i]) for i in (0, 1)
    ]
    expected = (torch.stack(terms) / 4).mean()
    assert losses.gbce(h, w, y, neg, t=0.5, n_items=10).item() == pytest.approx(expected.item())
