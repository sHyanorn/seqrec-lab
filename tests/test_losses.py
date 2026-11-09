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


def test_precision_under_bf16_autocast():
    torch.manual_seed(0)
    h, w = torch.randn(30, 16), torch.randn(52, 16)
    y = torch.randint(1, 51, (30,))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        shared = losses.sampled_logits(h, w, y, torch.randint(1, 51, (8,)))
        neg = torch.randint(1, 51, (30, 8))
        mixed = losses.gbce(h, w, y, neg, t=0.75, n_items=50)
    assert shared.dtype == torch.bfloat16  # positive and negatives at one precision
    exact = losses.gbce(h, w, y, neg, t=0.75, n_items=50)
    assert mixed.dtype == torch.float32 and mixed.item() == pytest.approx(exact.item(), abs=1e-6)


class _LargestTensor(torch.utils._python_dispatch.TorchDispatchMode):
    """Records the most elements of any new tensor an operation returns (views excluded)."""

    largest = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        inputs = {
            t.untyped_storage().data_ptr()
            for t in torch.utils._pytree.tree_leaves((args, kwargs))
            if isinstance(t, torch.Tensor)
        }
        for t in torch.utils._pytree.tree_leaves(out):
            if isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() not in inputs:
                self.largest = max(self.largest, t.numel())
        return out


def gbce_inputs(m, k, n, d):
    h, w, y = tensors(m=m, n=n, d=d, requires_grad=True)
    neg = torch.randint(1, n + 1, (m, k))
    neg[0, 0] = y[0]  # one collision
    return h, w, y, neg


@pytest.mark.parametrize("n", [50, 300])  # w.shape[0] at most K x d (gather), or above it
def test_gbce_never_holds_a_negative_embedding_per_position(n, monkeypatch):
    """D042: the (M, K, d) tensor of gathered negative embeddings, whose size changed with
    every batch, filled the device. Neither path creates it, in the forward or the backward
    pass."""
    m, k, d = 64, 8, 16
    monkeypatch.setattr(losses, "GBCE_CHUNK_ELEMENTS", 4 * k * d)
    h, w, y, neg = gbce_inputs(m, k, n, d)
    with _LargestTensor() as mode:
        losses.gbce(h, w, y, neg, t=0.75, n_items=n).backward()
    assert mode.largest < m * k * d


@pytest.mark.parametrize("n", [50, 300])
@pytest.mark.parametrize("budget", [2**26, 512])  # one slice, or several
def test_gbce_paths_match_the_gathered_form_in_value_and_gradient(n, budget, monkeypatch):
    m, k, d = 64, 8, 16
    monkeypatch.setattr(losses, "GBCE_CHUNK_ELEMENTS", budget)
    h, w, y, neg = gbce_inputs(m, k, n, d)
    beta = losses.gbce_beta(k, n, 0.75)
    pos = (h * w[y]).sum(-1)
    keep = (neg != y[:, None]).double()
    neg_logits = torch.einsum("md,mkd->mk", h, w[neg])
    want = ((beta * F.softplus(-pos) + (keep * F.softplus(neg_logits)).sum(1)) / (1 + k)).mean()
    want_grads = torch.autograd.grad(want, (h, w))
    got = losses.gbce(h, w, y, neg, t=0.75, n_items=n)
    got_grads = torch.autograd.grad(got, (h, w))
    assert got.item() == pytest.approx(want.item(), abs=1e-12)
    for g, e in zip(got_grads, want_grads, strict=True):
        torch.testing.assert_close(g, e, rtol=0, atol=1e-12)


@pytest.mark.parametrize("rows, alternative", [(50, 4 * 8 * 16), (300, 4 * 300)])
def test_negative_logits_take_the_smaller_intermediate(rows, alternative):
    """At most K x d rows: all item scores (4, rows); more: the embeddings (4, K, d). The other
    form would create a tensor of `alternative` elements."""
    h, w = torch.randn(4, 16), torch.randn(rows, 16)
    neg = torch.randint(1, rows, (4, 8))
    with _LargestTensor() as mode:
        losses.negative_logits(h, w, neg)
    assert mode.largest < alternative


@pytest.mark.parametrize("rows", [52, 300])
def test_negative_logits_slices_stay_within_the_budget(rows, monkeypatch):
    m, k, d, budget = 66, 8, 16, 512  # the last slice is partial
    monkeypatch.setattr(losses, "GBCE_CHUNK_ELEMENTS", budget)
    h, w = torch.randn(m, d), torch.randn(rows, d)
    neg = torch.randint(1, rows, (m, k))
    with _LargestTensor() as mode:
        got = losses.negative_logits(h, w, neg)
    assert mode.largest <= max(budget, m * k)  # the slices, then their concatenation
    torch.testing.assert_close(got, torch.einsum("md,mkd->mk", h, w[neg]))


def test_sliced_gbce_under_bf16_autocast_matches_fp32(monkeypatch):
    """The recomputed slices run with autocast off, as in the forward pass."""
    monkeypatch.setattr(losses, "GBCE_CHUNK_ELEMENTS", 512)
    torch.manual_seed(1)
    h, w = torch.randn(66, 16, requires_grad=True), torch.randn(302, 16, requires_grad=True)
    y, neg = torch.randint(1, 301, (66,)), torch.randint(1, 301, (66, 8))
    exact = losses.gbce(h, w, y, neg, t=0.75, n_items=300)
    want = torch.autograd.grad(exact, (h, w))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = losses.gbce(h, w, y, neg, t=0.75, n_items=300)
    got = torch.autograd.grad(mixed, (h, w))
    assert mixed.dtype == torch.float32 and mixed.item() == exact.item()
    for g, e in zip(got, want, strict=True):
        torch.testing.assert_close(g, e, rtol=0, atol=0)
