"""SASRec backbone and arms A1 to A4 (METHODS 4.1, 4.3).

The block follows the SASRec code [31], which pmixer/SASRec.pytorch, [5] and [33] keep:
only the attention query is layer-normalised, keys and values read the un-normalised
stream, and both residuals add the normalised input. The feed-forward network is
d -> d with ReLU, and a final LayerNorm follows the last block. Inputs are left-padded;
the attention mask is causal and blocks padded keys, and a padded query attends to itself.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from seqrec.data.splits import TrainSplit
from seqrec.models import batches, losses
from seqrec.models.trainer import SHARED_DEFAULTS, NeuralModel, resolve
from seqrec.runner.config import RunConfig
from seqrec.runner.run import max_len_of, register_arm

LN_EPS = 1e-8  # as in [31]


def attention_mask(real: torch.Tensor) -> torch.Tensor:
    """(B, 1, L, L) boolean mask, True where attention is allowed."""
    length = real.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool, device=real.device).tril()
    eye = torch.eye(length, dtype=torch.bool, device=real.device)
    return ((causal & real[:, None, :]) | eye)[:, None]


class Attention(nn.Module):
    """Multi-head attention through SDPA, equivalent to nn.MultiheadAttention."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.query = nn.Linear(dim, dim)
        self.key_value = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor, allowed: torch.Tensor):
        b, n, d = query.shape
        h = self.heads
        q = self.query(query).view(b, n, h, d // h).transpose(1, 2)
        k, v = self.key_value(context).view(b, n, 2, h, d // h).permute(2, 0, 3, 1, 4)
        p = self.dropout if self.training else 0.0
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=p)
        return self.out(o.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=LN_EPS)
        self.attn = Attention(dim, heads, dropout)
        self.ffn_norm = nn.LayerNorm(dim, eps=LN_EPS)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        q = self.attn_norm(x)
        x = q + self.attn(q, x, allowed)
        x = self.ffn_norm(x)
        return x + self.ffn(x)


class SASRec(nn.Module):
    def __init__(self, n_items, max_len, dim, blocks, heads, dropout, shared_output):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.position = nn.Embedding(max_len, dim)
        self.output = None if shared_output else nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(dim, heads, dropout) for _ in range(blocks))
        self.norm = nn.LayerNorm(dim, eps=LN_EPS)

    def output_table(self) -> torch.Tensor:
        return (self.item if self.output is None else self.output).weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, L) left-padded ids -> (B, L, d); the most recent item has the last position."""
        real = (ids > 0)[..., None]
        x = self.item(ids) * self.item.embedding_dim**0.5 + self.position.weight[-ids.shape[1] :]
        x = self.drop(x) * real
        allowed = attention_mask(ids > 0)
        for block in self.blocks:
            x = block(x, allowed) * real
        return self.norm(x)


OBJECTIVES = {
    "sasrec-bce": {},
    "sasrec-ce": {"ce_chunk": 0},
    "sasrec-ss": {"negatives": 256},
    "gsasrec": {"negatives": 256, "t": 0.75},
}


class SASRecModel(NeuralModel):
    def __init__(self, cfg: RunConfig, train: TrainSplit, device: torch.device, objective: str):
        hp = resolve(cfg.params, SHARED_DEFAULTS | {"shared_output": True} | OBJECTIVES[objective])
        self.objective = objective
        net = SASRec(
            train.n_items, max_len_of(cfg), hp["dim"], hp["blocks"], hp["heads"],
            hp["dropout"], hp["shared_output"],
        )  # fmt: skip
        super().__init__(cfg, train, device, net, hp)
        self.sampler = (
            batches.HistoryExcludingSampler(train, device) if objective == "sasrec-bce" else None
        )

    def require(self, train: TrainSplit) -> None:
        super().require(train)
        if self.sampler is not None:
            self.sampler.require(train)

    def make_batch(self, rows, generator):
        return batches.sasrec_batch(self.data, rows, self.max_len)

    def check_batch(self, batch):
        batches.check_sasrec_batch(batch, self.data)

    def loss(self, batch):
        labelled = batch.labels > 0
        h = self.net(batch.inputs)[labelled]
        y = batch.labels[labelled]
        w = self.net.output_table()
        hp, n = self.hparams, self.n_items
        if self.objective == "sasrec-ce":
            return losses.full_ce(h, w, y, n_items=n, chunk=hp["ce_chunk"])
        if self.objective == "sasrec-ss":
            neg = batches.uniform_negatives(n, (hp["negatives"],), self.gen, self.device)
            return losses.sampled_softmax(h, w, y, neg)
        if self.objective == "sasrec-bce":
            neg = self.sampler.sample(batch.rows, batch.labels.shape[1:], self.gen)[labelled]
            return losses.bce(h, w, y, neg)
        neg = batches.uniform_negatives(n, (len(y), hp["negatives"]), self.gen, self.device)
        return losses.gbce(h, w, y, neg, t=hp["t"], n_items=n)

    def scores(self, inputs):
        return self.net(inputs)[:, -1] @ self.net.output_table().T


for _name in OBJECTIVES:
    register_arm(_name)(partial(SASRecModel, objective=_name))
