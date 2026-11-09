"""Training loop shared by the neural arms (METHODS 5.1).

One epoch is one pass over the training rows in an order drawn from the model's own
generator, which also draws masks and negatives; its state is part of `state_dict`, so a
resumed run continues the same random stream.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from seqrec.data.splits import TrainSplit
from seqrec.models.batches import Batch, TrainData, batch_positions
from seqrec.runner.config import RunConfig
from seqrec.runner.run import max_len_of

BATCH_SIZE = 128
CLIP_NORM = 5.0
INIT_STD = 0.02
SHARED_DEFAULTS = {
    "lr": 1e-3,
    "dim": 64,
    "blocks": 2,
    "heads": 1,
    "dropout": 0.1,
    "weight_decay": 0.0,
}


def resolve(params: dict, defaults: dict) -> dict:
    unknown = sorted(set(params) - set(defaults))
    if unknown:
        raise ValueError(f"unknown hyperparameters {unknown}")
    hp = defaults | params
    if hp["dim"] % hp["heads"]:
        raise ValueError("dim must be divisible by heads")
    return hp


def init_weights(net: nn.Module) -> None:
    """Weights N(0, 0.02), biases 0, LayerNorm (1, 0), padding rows 0."""
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=INIT_STD)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=INIT_STD)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class NeuralModel:
    """Implements the runner's Model protocol; subclasses build batches, losses and scores."""

    def __init__(self, cfg: RunConfig, train: TrainSplit, device: torch.device, net, hp: dict):
        self.hparams = hp
        self.device = device
        self.max_len = max_len_of(cfg)
        self.n_items = train.n_items
        self.data = TrainData(train, device)
        init_weights(net)
        self.net = net.to(device)
        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=hp["lr"], betas=(0.9, 0.999), weight_decay=hp["weight_decay"]
        )
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(cfg.seed)
        self.amp = device.type == "cuda"

    def make_batch(self, rows: torch.Tensor, generator: torch.Generator) -> Batch:
        raise NotImplementedError

    def check_batch(self, batch: Batch) -> None:
        raise NotImplementedError

    def loss(self, batch: Batch) -> torch.Tensor:
        raise NotImplementedError

    def scores(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def require(self, train: TrainSplit) -> None:
        """Every statistic must come from the training split the runner passes in."""
        self.data.require(train)

    def train_epoch(self, train: TrainSplit, epoch: int) -> dict:
        self.require(train)
        torch.set_float32_matmul_precision("high")  # TF32 allowed in training (METHODS 5.1)
        self.net.train()
        rows = self.data.rows
        order = rows[torch.randperm(len(rows), generator=self.gen, device=self.device)]
        total = torch.zeros((), device=self.device)
        steps = 0
        for i, batch_rows in enumerate(order.split(BATCH_SIZE)):
            batch = self.make_batch(batch_rows, self.gen)
            if i == 0:
                self.check_batch(batch)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                loss = self.loss(batch)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), CLIP_NORM)
            self.opt.step()
            total += loss.detach()
            steps += 1
        return {"loss": total.item() / max(steps, 1)}

    @torch.no_grad()
    def score(self, inputs: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        return self.scores(inputs)

    def sample_positions(self) -> np.ndarray:
        """Full-log positions used by one batch of randomly chosen rows, for the runner's
        check that batches touch training rows only (METHODS 11.1)."""
        gen = torch.Generator(device=self.device)
        gen.manual_seed(0)
        pick = torch.randperm(len(self.data.rows), generator=gen, device=self.device)
        batch = self.make_batch(self.data.rows[pick[:BATCH_SIZE]], gen)
        return batch_positions(self.data, batch)

    def state_dict(self) -> dict:
        """A snapshot of the training state: later steps do not change it."""
        state = {"net": self.net.state_dict(), "opt": self.opt.state_dict()}
        return copy.deepcopy(state | {"gen": self.gen.get_state()})

    def weights(self) -> dict:
        """A snapshot of the network weights only, for the best checkpoint."""
        return copy.deepcopy({"net": self.net.state_dict()})

    def load_state_dict(self, state: dict) -> None:
        """Loads a training state or weights only; never aliases the given tensors."""
        state = copy.deepcopy(state)
        self.net.load_state_dict(state["net"])
        if "opt" in state:
            self.opt.load_state_dict(state["opt"])
            self.gen.set_state(state["gen"])
