"""Training objectives (METHODS 4.3).

Every function takes the hidden states at labelled positions, h (M, d), the output table
indexed by item id, w (rows 0..N or more; row 0 is padding), and targets y (M,) in 1..N.
Scores may be computed under bf16 autocast; every loss is reduced in fp32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _wide(x: torch.Tensor) -> torch.Tensor:
    """At least fp32: bf16 scores are upcast, fp64 ones (tests) stay fp64."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def _ce_sum(h, w, bias, y):
    logits = h @ w.T
    if bias is not None:
        logits = logits + bias
    return F.cross_entropy(_wide(logits), y, reduction="sum")


def full_ce(h, w, y, bias=None, n_items: int | None = None, chunk: int = 0) -> torch.Tensor:
    """Softmax cross-entropy over items 1..N (padding and mask ids excluded).

    With `chunk` > 0, positions are processed in slices whose logits are recomputed in the
    backward pass, so at most `chunk` x N logits exist at once; loss and gradients are
    unchanged.
    """
    n = n_items if n_items is not None else w.shape[0] - 1
    w_items = w[1 : n + 1]
    b_items = bias[1 : n + 1] if bias is not None else None
    target = y - 1
    if chunk <= 0:
        return _ce_sum(h, w_items, b_items, target) / len(y)
    total = h.new_zeros((), dtype=torch.promote_types(h.dtype, torch.float32))
    for s in range(0, len(y), chunk):
        total = total + checkpoint(
            _ce_sum, h[s : s + chunk], w_items, b_items, target[s : s + chunk], use_reentrant=False
        )
    return total / len(y)


def mask_collisions(neg_logits: torch.Tensor, neg: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Set to -inf every negative that equals the position's positive item."""
    return neg_logits.masked_fill(neg == y[:, None], float("-inf"))


def sampled_logits(h, w, y, neg) -> torch.Tensor:
    """(M, 1 + K) logits of the positive and the shared negatives. Both come from matrix
    products, so under autocast they share one precision."""
    pos = torch.bmm(h[:, None, :], w[y][:, :, None]).squeeze(-1)
    neg_logits = mask_collisions(h @ w[neg].T, neg[None, :], y)
    return torch.cat([pos, neg_logits], dim=1)


def sampled_softmax(h, w, y, neg) -> torch.Tensor:
    """Softmax over the positive and K negatives shared by all positions (A3). The uniform
    proposal makes the log-Q correction constant, so none is applied."""
    return F.cross_entropy(_wide(sampled_logits(h, w, y, neg)), torch.zeros_like(y))


def bce(h, w, y, neg) -> torch.Tensor:
    """-log sigma(s+) - log(1 - sigma(s-)) with one negative per position, summed over the
    two terms and averaged over positions, as in the SASRec code [31] (A1)."""
    pos = _wide((h * w[y]).sum(-1))
    negs = _wide((h * w[neg]).sum(-1))
    return (F.softplus(-pos) + F.softplus(negs)).mean()


def gbce_beta(k: int, n_items: int, t: float) -> float:
    alpha = k / (n_items - 1)
    return alpha * (t * (1 - 1 / alpha) + 1 / alpha)


def gbce(h, w, y, neg, t: float, n_items: int) -> torch.Tensor:
    """gBCE with K uniform negatives per position (A4, [4]), computed in fp32 even under
    autocast (METHODS 4.3). Negatives equal to the positive are dropped. Each position's
    loss is averaged over its 1 + K terms, as in [33]."""
    k = neg.shape[1]
    beta = gbce_beta(k, n_items, t)
    with torch.autocast(h.device.type, enabled=False):
        h, w = _wide(h), _wide(w)
        pos = (h * w[y]).sum(-1)
        neg_logits = torch.einsum("md,mkd->mk", h, w[neg])
    keep = (neg != y[:, None]).to(neg_logits.dtype)
    per_position = beta * F.softplus(-pos) + (keep * F.softplus(neg_logits)).sum(1)
    return (per_position / (1 + k)).mean()
