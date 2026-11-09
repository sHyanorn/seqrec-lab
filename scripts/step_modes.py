"""Execution modes for the training step of the SASRec arms, for `profile_step.py` only.

- fixed shapes: the loss runs over every position of the batch with a mask, in place of
  selecting labelled positions by boolean indexing, whose output shape the CPU must read;
- fused Adam: `torch.optim.Adam(fused=True)`, the same update in one kernel per step;
- CUDA graph: forward, backward, clipping and the Adam step of a full batch are captured once
  and replayed; batches are built eagerly and copied into static buffers, and the last,
  partial batch of an epoch runs eagerly.

Training runs never use these modes; the losses follow METHODS 4.3 as `seqrec.models.losses`
does.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from seqrec.models import batches, losses
from seqrec.models.trainer import BATCH_SIZE, CLIP_NORM

ARMS = ("sasrec-bce", "sasrec-ce", "sasrec-ss", "gsasrec")


def masked_mean(per_position: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over labelled positions; the count stays on the device."""
    return (per_position * mask).sum() / mask.sum()


def fixed_shape_loss(model, batch) -> torch.Tensor:
    """The arm's loss over all B x L positions, unlabelled ones weighted 0. Equal to
    `model.loss` given the same negatives; gSASRec draws K negatives for every position."""
    h = model.net(batch.inputs).flatten(0, 1)
    labels = batch.labels.flatten()
    mask = (labels > 0).to(torch.float32)
    y = labels.clamp(min=1)  # any item id where no loss applies; its terms are masked out
    w, n, hp = model.net.output_table(), model.n_items, model.hparams
    if model.objective == "sasrec-ce":
        logits = losses._wide(h @ w[1 : n + 1].T)
        return masked_mean(F.cross_entropy(logits, y - 1, reduction="none"), mask)
    if model.objective == "sasrec-ss":
        neg = batches.uniform_negatives(n, (hp["negatives"],), model.gen, model.device)
        logits = losses._wide(losses.sampled_logits(h, w, y, neg))
        return masked_mean(F.cross_entropy(logits, torch.zeros_like(y), reduction="none"), mask)
    if model.objective == "sasrec-bce":
        neg = model.sampler.sample(batch.rows, batch.labels.shape[1:], model.gen).flatten()
        pos = losses._wide((h * w[y]).sum(-1))
        negs = losses._wide((h * w[neg]).sum(-1))
        return masked_mean(F.softplus(-pos) + F.softplus(negs), mask)
    k = hp["negatives"]
    neg = batches.uniform_negatives(n, (len(y), k), model.gen, model.device)
    beta = losses.gbce_beta(k, n, hp["t"])
    with torch.autocast(h.device.type, enabled=False):
        h, w = losses._wide(h), losses._wide(w)
        pos = (h * w[y]).sum(-1)
        neg_logits = losses.negative_logits(h, w, neg)
    keep = (neg != y[:, None]).to(neg_logits.dtype)
    per_position = beta * F.softplus(-pos) + (keep * F.softplus(neg_logits)).sum(1)
    return masked_mean(per_position / (1 + k), mask)


class Stepper:
    """One training step of `model` under the chosen modes, and an epoch loop like
    `NeuralModel.train_epoch` built on it."""

    def __init__(self, model, fixed_shapes=False, fused_adam=False, cuda_graph=False):
        if getattr(model, "objective", None) not in ARMS:
            raise ValueError(f"execution modes cover {', '.join(ARMS)}")
        if cuda_graph and model.device.type != "cuda":
            raise ValueError("a CUDA graph needs a CUDA device")
        if cuda_graph and not fixed_shapes:
            raise ValueError("a CUDA graph needs fixed shapes")
        self.model = model
        self.loss_fn = fixed_shape_loss if fixed_shapes else type(model).loss
        if fused_adam or cuda_graph:
            hp = model.hparams
            model.opt = torch.optim.Adam(
                model.net.parameters(), lr=hp["lr"], betas=(0.9, 0.999),
                weight_decay=hp["weight_decay"], fused=fused_adam or None,
                capturable=cuda_graph,
            )  # fmt: skip
        self.cuda_graph = cuda_graph
        self.graph = None

    def _loss(self, batch):
        m = self.model
        with torch.autocast(m.device.type, dtype=torch.bfloat16, enabled=m.amp,
                            cache_enabled=not self.cuda_graph):  # fmt: skip
            return self.loss_fn(m, batch)

    def eager_step(self, batch) -> torch.Tensor:
        m = self.model
        loss = self._loss(batch)
        m.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(m.net.parameters(), CLIP_NORM)
        m.opt.step()
        return loss.detach()

    def _capture(self, batch) -> None:
        """Warm up on a side stream, then capture one step on static copies of `batch`."""
        m = self.model
        self.static = batch
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.eager_step(self.static)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        self.graph.register_generator_state(m.gen)
        m.opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph):
            self.static_loss = self._loss(self.static)
            self.static_loss.backward()
            nn.utils.clip_grad_norm_(m.net.parameters(), CLIP_NORM)
            m.opt.step()

    def step(self, batch) -> torch.Tensor:
        if not self.cuda_graph or len(batch.rows) != BATCH_SIZE:
            return self.eager_step(batch)
        if self.graph is None:
            self._capture(batch)
        for name in ("rows", "inputs", "labels"):
            getattr(self.static, name).copy_(getattr(batch, name))
        self.graph.replay()
        return self.static_loss.detach().clone()

    def train_epoch(self, train, epoch: int) -> dict:
        m = self.model
        m.require(train)
        torch.set_float32_matmul_precision("high")
        m.net.train()
        rows = m.data.rows
        order = rows[torch.randperm(len(rows), generator=m.gen, device=m.device)]
        total = torch.zeros((), device=m.device)
        steps = 0
        for i, batch_rows in enumerate(order.split(BATCH_SIZE)):
            batch = m.make_batch(batch_rows, m.gen)
            if i == 0:
                m.check_batch(batch)
            total += self.step(batch)
            steps += 1
        return {"loss": total.item() / max(steps, 1)}
