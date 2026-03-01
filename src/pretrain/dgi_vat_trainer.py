from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import DeepGraphInfomax

from src.pretrain.vat import VATLoss
from src.utils.trainer import BaseTrainer


class DGIVATPretrainer(BaseTrainer):
    """
    Deep Graph Infomax pretraining with Virtual Adversarial Training (VAT).

    Total loss per step:
        L = L_dgi + lambda_vat * L_vat

    The auxiliary VAT head is used only during pretraining and is not exported
    to downstream finetuning. The encoder remains the same shared module that
    DGI uses.
    """

    def __init__(
        self,
        model: DeepGraphInfomax,
        encoder: nn.Module,
        data: Data,
        num_classes: int,
        lr: float,
        weight_decay: float,
        device: torch.device,
        logger: Optional[callable] = None,
        *,
        vat_lambda: float = 1.0,
        vat_eps: float = 1e-2,
        vat_xi: float = 1e-6,
        vat_ip: int = 1,
    ) -> None:
        self.model = model.to(device)
        self.encoder = encoder.to(device)
        self.data = data.to(device)
        self.device = device
        self.logger = logger

        hidden_dim = self.encoder.conv2.out_channels  # type: ignore[attr-defined]
        self.vat_head = nn.Linear(hidden_dim, num_classes).to(device)
        self.vat_loss_fn = VATLoss(eps=vat_eps, xi=vat_xi, ip=vat_ip)
        self.vat_lambda = vat_lambda

        self.optimizer = Adam(
            list(self.model.parameters()) + list(self.vat_head.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        self._last_pretrain_loss: float = 0.0
        self._last_dgi_loss: float = 0.0
        self._last_vat_loss: float = 0.0

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger(msg)

    def _forward_vat_logits(self, x: Tensor) -> Tensor:
        h = self.encoder(x, self.data.edge_index)
        logits = self.vat_head(h)
        return logits

    def _step(self) -> Dict[str, float]:
        self.model.train()
        self.encoder.train()
        self.vat_head.train()

        self.optimizer.zero_grad()

        pos_z, neg_z, summary = self.model(self.data.x, self.data.edge_index)
        dgi_loss_tensor = self.model.loss(pos_z, neg_z, summary)

        vat_loss_tensor = self.vat_loss_fn(self._forward_vat_logits, self.data.x)

        total_loss = dgi_loss_tensor + self.vat_lambda * vat_loss_tensor

        total_loss.backward()
        self.optimizer.step()

        dgi_loss = float(dgi_loss_tensor.item())
        vat_loss = float(vat_loss_tensor.item())
        total = float(total_loss.item())

        self._last_dgi_loss = dgi_loss
        self._last_vat_loss = vat_loss
        self._last_pretrain_loss = total

        return {
            "pretrain_loss": total,
            "dgi_loss": dgi_loss,
            "vat_loss": vat_loss,
        }

    def train(self, num_epochs: int) -> Dict[str, float]:
        last_stats: Dict[str, float] = {}
        for epoch in range(1, num_epochs + 1):
            last_stats = self._step()
            self._log(
                "[DGI+VAT] Epoch "
                f"{epoch:03d} | dgi_loss={last_stats['dgi_loss']:.4f} | "
                f"vat_loss={last_stats['vat_loss']:.4f} | "
                f"total={last_stats['pretrain_loss']:.4f}"
            )
        return last_stats

    def evaluate(self, split: str = "train") -> Dict[str, float]:
        # For pretraining we expose the last recorded losses.
        return {
            "pretrain_loss": self._last_pretrain_loss,
            "dgi_loss": self._last_dgi_loss,
            "vat_loss": self._last_vat_loss,
        }

    def dry_run_debug_step(self) -> Dict[str, float]:
        """
        Single forward/backward step with diagnostics for dry_run.
        """
        self.model.train()
        self.encoder.train()
        self.vat_head.train()

        self.optimizer.zero_grad()

        pos_z, neg_z, summary = self.model(self.data.x, self.data.edge_index)
        dgi_loss_tensor = self.model.loss(pos_z, neg_z, summary)
        vat_loss_tensor = self.vat_loss_fn(self._forward_vat_logits, self.data.x)
        total_loss = dgi_loss_tensor + self.vat_lambda * vat_loss_tensor

        self._log(
            "[DGI+VAT dry] "
            f"pos_z shape={tuple(pos_z.shape)}, "
            f"neg_z shape={tuple(neg_z.shape)}, "
            f"summary shape={tuple(summary.shape)}, "
            f"dgi_loss={dgi_loss_tensor.item():.4f}, "
            f"vat_loss={vat_loss_tensor.item():.4f}, "
            f"total={total_loss.item():.4f}"
        )

        total_loss.backward()

        total_norm_sq: Tensor = torch.tensor(0.0, device=self.device)
        for p in list(self.model.parameters()) + list(self.vat_head.parameters()):
            if p.grad is None:
                continue
            total_norm_sq += p.grad.detach().pow(2).sum()
        grad_norm = float(total_norm_sq.sqrt().item())

        self._log(f"[DGI+VAT dry] grad_norm={grad_norm:.4f}")

        self.optimizer.step()

        self._last_dgi_loss = float(dgi_loss_tensor.item())
        self._last_vat_loss = float(vat_loss_tensor.item())
        self._last_pretrain_loss = float(total_loss.item())

        return {
            "pretrain_loss": self._last_pretrain_loss,
            "dgi_loss": self._last_dgi_loss,
            "vat_loss": self._last_vat_loss,
            "grad_norm": grad_norm,
        }

