from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VATLoss(nn.Module):
    """
    Virtual Adversarial Training loss.

    Implementation follows:
      - no labels required
      - finite-difference power iteration
      - KL divergence between softmax distributions.
    """

    def __init__(self, eps: float = 1e-2, xi: float = 1e-6, ip: int = 1) -> None:
        super().__init__()
        self.eps = eps
        self.xi = xi
        self.ip = ip

    @staticmethod
    def _l2_normalize(d: Tensor) -> Tensor:
        # Flatten per node, normalize, then restore shape.
        orig_shape = d.shape
        d_flat = d.view(d.size(0), -1)
        d_norm = torch.norm(d_flat, p=2, dim=1, keepdim=True)
        d_flat = d_flat / (d_norm + 1e-8)
        return d_flat.view(orig_shape)

    def forward(
        self,
        forward_logits_fn: Callable[[Tensor], Tensor],
        x: Tensor,
    ) -> Tensor:
        """
        forward_logits_fn: callable(x_perturbed) -> logits
        x: node features tensor of shape [num_nodes, feat_dim]
        returns: scalar VAT loss tensor
        """
        with torch.no_grad():
            logits = forward_logits_fn(x)
            p = F.softmax(logits, dim=-1)

        # Initialize perturbation
        d = torch.randn_like(x)

        # Power iteration to approximate adversarial direction
        for _ in range(self.ip):
            d = self._l2_normalize(d)
            d = d * self.xi
            d.requires_grad_()

            logits_pert = forward_logits_fn(x + d)
            log_q = F.log_softmax(logits_pert, dim=-1)
            # KL(p || q) where p is fixed (no grad)
            kld = F.kl_div(log_q, p, reduction="batchmean")

            grad_d = torch.autograd.grad(kld, d, only_inputs=True)[0]
            d = grad_d.detach()

        # Final adversarial perturbation
        r_adv = self.eps * self._l2_normalize(d)

        logits_adv = forward_logits_fn(x + r_adv)
        log_q_adv = F.log_softmax(logits_adv, dim=-1)

        # Reuse p as reference distribution; do not backprop through p
        vat_loss = F.kl_div(log_q_adv, p, reduction="batchmean")
        return vat_loss

