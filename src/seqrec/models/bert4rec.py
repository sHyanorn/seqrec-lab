"""BERT4Rec, arm A5 (METHODS 4.2).

As in [32] and the HuggingFace BertModel used by [5]: item and position embeddings are
summed, layer-normalised and dropped out; post-LN blocks with GELU and a 4d feed-forward
layer (no dropout after the GELU); an output head of dense, GELU and LayerNorm, then a dot
product with the item embeddings plus an item bias. Id 0 is padding and N+1 the mask token.
"""

from __future__ import annotations

import torch
from torch import nn

from seqrec.data.splits import TrainSplit
from seqrec.models import batches, losses
from seqrec.models.sasrec import Attention
from seqrec.models.trainer import SHARED_DEFAULTS, NeuralModel, resolve
from seqrec.runner.config import RunConfig
from seqrec.runner.run import max_len_of, register_arm

LN_EPS = 1e-12  # BertConfig default


class Block(nn.Module):
    """Post-LN Transformer block, as BertLayer."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.attn = Attention(dim, heads, dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(dim, eps=LN_EPS)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(dim, eps=LN_EPS)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        x = self.attn_norm(x + self.attn_drop(self.attn(x, x, allowed)))
        return self.ffn_norm(x + self.ffn(x))


class BERT4Rec(nn.Module):
    def __init__(self, n_items, max_len, dim, blocks, heads, dropout):
        super().__init__()
        self.mask_id = n_items + 1
        self.item = nn.Embedding(n_items + 2, dim, padding_idx=0)
        self.position = nn.Embedding(max_len, dim)
        self.emb_norm = nn.LayerNorm(dim, eps=LN_EPS)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(dim, heads, dropout) for _ in range(blocks))
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim, eps=LN_EPS))
        self.bias = nn.Parameter(torch.zeros(n_items + 2))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, L) left-padded ids -> (B, L, d) after the output head's transform."""
        x = self.item(ids) + self.position.weight[-ids.shape[1] :]
        x = self.drop(self.emb_norm(x))
        allowed = (ids > 0)[:, None, None, :]  # bidirectional; padded keys blocked
        for block in self.blocks:
            x = block(x, allowed)
        return self.head(x)

    def with_mask_token(self, inputs: torch.Tensor) -> torch.Tensor:
        """The last L-1 input items followed by the mask token, whose output is scored."""
        mask = torch.full_like(inputs[:, :1], self.mask_id)
        return torch.cat([inputs[:, 1:], mask], dim=1)


class BERT4RecModel(NeuralModel):
    def __init__(self, cfg: RunConfig, train: TrainSplit, device: torch.device):
        hp = resolve(cfg.params, SHARED_DEFAULTS | {"mask_prob": 0.2})
        net = BERT4Rec(
            train.n_items, max_len_of(cfg), hp["dim"], hp["blocks"], hp["heads"], hp["dropout"]
        )
        super().__init__(cfg, train, device, net, hp)

    def make_batch(self, rows, generator):
        return batches.bert4rec_batch(
            self.data, rows, self.max_len, self.hparams["mask_prob"], generator
        )

    def check_batch(self, batch):
        batches.check_bert4rec_batch(batch, self.data)

    def loss(self, batch):
        masked = batch.labels > 0
        h = self.net(batch.inputs)[masked]
        y = batch.labels[masked]
        return losses.full_ce(h, self.net.item.weight, y, bias=self.net.bias, n_items=self.n_items)

    def scores(self, inputs):
        h = self.net(self.net.with_mask_token(inputs))[:, -1]
        return h @ self.net.item.weight.T + self.net.bias


register_arm("bert4rec")(BERT4RecModel)
