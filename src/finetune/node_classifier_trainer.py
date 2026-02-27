from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam
from torch_geometric.data import Data

from src.eval.metrics import accuracy
from src.models.gnn_encoder import GCNEncoder
from src.utils.trainer import BaseTrainer


class NodeClassificationTrainer(BaseTrainer):
    """
    Supervised node classification on top of a shared encoder.
    """

    def __init__(
        self,
        encoder: GCNEncoder,
        num_classes: int,
        data: Data,
        lr: float,
        weight_decay: float,
        device: torch.device,
        logger: Optional[callable] = None,
    ) -> None:
        self.encoder = encoder.to(device)
        hidden_dim = encoder.conv2.out_channels
        self.classifier = nn.Linear(hidden_dim, num_classes).to(device)
        self.data = data.to(device)
        self.device = device
        self.optimizer = Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.logger = logger
        self._last_train_loss: float = 0.0

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger(msg)

    def _forward_logits(self) -> Tensor:
        self.encoder.train()
        self.classifier.train()
        z = self.encoder(self.data.x, self.data.edge_index)
        logits = self.classifier(z)
        return logits

    def _step(self) -> float:
        self.optimizer.zero_grad()
        logits = self._forward_logits()
        train_mask = self.data.train_mask
        loss = F.cross_entropy(logits[train_mask], self.data.y[train_mask])
        loss.backward()
        self.optimizer.step()
        self._last_train_loss = float(loss.item())
        return self._last_train_loss

    def train(self, num_epochs: int) -> Dict[str, float]:
        last_loss = 0.0
        for epoch in range(1, num_epochs + 1):
            last_loss = self._step()
            train_metrics = self.evaluate("train")
            val_metrics = self.evaluate("val")
            self._log(
                "[CLS] Epoch "
                f"{epoch:03d} | loss={last_loss:.4f} | "
                f"train_acc={train_metrics['accuracy']:.3f} | "
                f"val_acc={val_metrics['accuracy']:.3f}"
            )
        return {
            "train_loss": last_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
        }

    def evaluate(self, split: str) -> Dict[str, float]:
        self.encoder.eval()
        self.classifier.eval()
        mask = getattr(self.data, f"{split}_mask")
        with torch.no_grad():
            z = self.encoder(self.data.x, self.data.edge_index)
            logits = self.classifier(z)
            split_logits = logits[mask]
            split_labels = self.data.y[mask]
            acc = accuracy(split_logits, split_labels)
        return {"accuracy": acc}

    def dry_run_debug_step(self) -> Dict[str, float]:
        """
        Single forward/backward step with shape + grad norm diagnostics.
        """
        self.encoder.train()
        self.classifier.train()
        self.optimizer.zero_grad()

        z = self.encoder(self.data.x, self.data.edge_index)
        logits = self.classifier(z)
        train_mask = self.data.train_mask
        loss = F.cross_entropy(logits[train_mask], self.data.y[train_mask])

        self._log(
            "[CLS dry] z shape="
            f"{tuple(z.shape)}, logits shape={tuple(logits.shape)}, "
            f"train_nodes={int(train_mask.sum())}, loss={loss.item():.4f}"
        )

        loss.backward()

        total_norm_sq: Tensor = torch.tensor(0.0, device=self.device)
        for p in list(self.encoder.parameters()) + list(self.classifier.parameters()):
            if p.grad is None:
                continue
            total_norm_sq += p.grad.detach().pow(2).sum()
        grad_norm = float(total_norm_sq.sqrt().item())

        self._log(f"[CLS dry] grad_norm={grad_norm:.4f}")

        self.optimizer.step()
        self._last_train_loss = float(loss.item())
        return {"train_loss": self._last_train_loss, "grad_norm": grad_norm}

