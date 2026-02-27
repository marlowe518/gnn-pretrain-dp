from typing import Dict, Optional

import torch
from torch import Tensor
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import DeepGraphInfomax

from src.utils.trainer import BaseTrainer


class DGIPretrainer(BaseTrainer):
    """
    Plain PyTorch trainer for Deep Graph Infomax.
    """

    def __init__(
        self,
        model: DeepGraphInfomax,
        data: Data,
        lr: float,
        weight_decay: float,
        device: torch.device,
        logger: Optional[callable] = None,
    ) -> None:
        self.model = model.to(device)
        self.data = data.to(device)
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.logger = logger
        self._last_loss: float = 0.0

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger(msg)

    def _step(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        pos_z, neg_z, summary = self.model(self.data.x, self.data.edge_index)
        loss = self.model.loss(pos_z, neg_z, summary)
        loss.backward()
        self.optimizer.step()
        self._last_loss = float(loss.item())
        return self._last_loss

    def train(self, num_epochs: int) -> Dict[str, float]:
        last_loss = 0.0
        for epoch in range(1, num_epochs + 1):
            last_loss = self._step()
            self._log(f"[DGI] Epoch {epoch:03d} | loss={last_loss:.4f}")
        return {"pretrain_loss": last_loss}

    def evaluate(self, split: str = "train") -> Dict[str, float]:
        # DGI has no standard supervised metric; we expose the last loss.
        return {"pretrain_loss": self._last_loss}

    def dry_run_debug_step(self) -> Dict[str, float]:
        """
        Single forward/backward step with shape + grad norm diagnostics.
        """
        self.model.train()
        self.optimizer.zero_grad()

        pos_z, neg_z, summary = self.model(self.data.x, self.data.edge_index)
        loss = self.model.loss(pos_z, neg_z, summary)

        self._log(
            "[DGI dry] pos_z shape="
            f"{tuple(pos_z.shape)}, neg_z shape={tuple(neg_z.shape)}, "
            f"summary shape={tuple(summary.shape)}, loss={loss.item():.4f}"
        )

        loss.backward()

        total_norm_sq: Tensor = torch.tensor(0.0, device=self.device)
        for p in self.model.parameters():
            if p.grad is None:
                continue
            total_norm_sq += p.grad.detach().pow(2).sum()
        grad_norm = float(total_norm_sq.sqrt().item())

        self._log(f"[DGI dry] grad_norm={grad_norm:.4f}")

        self.optimizer.step()
        self._last_loss = float(loss.item())
        return {"pretrain_loss": self._last_loss, "grad_norm": grad_norm}

