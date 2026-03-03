"""
DP-SGD optimizer for DP-GNN (PyTorch reimplementation).

Equivalent to upstream: differentially_private_gnns/optimizers.py
- clip_by_norm: per-layer L2 clipping of per-example gradients
- dp_aggregate: clip, sum, add Gaussian noise (std = clip * base_sensitivity * noise_multiplier), then apply update.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from torch import Tensor, nn
from torch.optim import Adam, SGD, Optimizer


def clip_and_aggregate(
    per_example_grads: List[Dict[nn.Parameter, Tensor]],
    clip_norms: Dict[nn.Parameter, float],
    base_sensitivity: float,
    noise_multiplier: float,
    device: torch.device,
) -> Dict[nn.Parameter, Tensor]:
    """
    Clip each layer's per-example grads by clip_norms, sum, add Gaussian noise.
    Upstream: optimizers.dp_aggregate (clip_by_norm then sum then noise).

    per_example_grads: list of param->grad dicts (one per example in batch).
    clip_norms: per-parameter L2 clip threshold (can be scalar per layer).
    base_sensitivity: scales noise (upstream: compute_base_sensitivity).
    noise_multiplier: sigma in DP-SGD.

    Returns: aggregated noisy gradients (param -> tensor), ready for optimizer step.
    """
    if not per_example_grads:
        return {}

    default_clip = 1.0
    if clip_norms:
        default_clip = next(iter(clip_norms.values()), 1.0)
        if not isinstance(default_clip, (int, float)):
            default_clip = 1.0
        else:
            default_clip = float(default_clip)

    out: Dict[nn.Parameter, Tensor] = {}
    for p, g in per_example_grads[0].items():
        if not p.requires_grad or g is None:
            continue
        clip_val = float(clip_norms.get(p, default_clip))
        # Per-example: clip this param's grad by L2 norm (upstream: clip_by_norm).
        grads_for_p = [d[p] for d in per_example_grads if p in d and d[p] is not None]
        if not grads_for_p:
            continue
        stacked = torch.stack(grads_for_p, dim=0)  # [B, *shape]
        norms = stacked.view(stacked.size(0), -1).norm(2, dim=1, keepdim=True)
        scale = (clip_val / (norms + 1e-6)).clamp(max=1.0)
        clipped = stacked * scale.view(-1, *([1] * (stacked.dim() - 1)))
        summed = clipped.sum(dim=0)
        # Noise: std = clip_val * base_sensitivity * noise_multiplier (upstream)
        std = clip_val * base_sensitivity * noise_multiplier
        if std > 0:
            summed = summed + torch.randn_like(summed, device=device) * std
        out[p] = summed
    return out


class DPOptimizer(Optimizer):
    """
    Wraps a base optimizer (SGD/Adam) and applies per-microbatch clip + aggregate + noise
    before each step. Upstream: dpsgd / dpadam (dp_aggregate + sgd/adam).
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        clip_norms: Optional[Dict[nn.Parameter, float]] = None,
        clip_norm: Optional[float] = None,
        base_sensitivity: float = 1.0,
        noise_multiplier: float = 1.0,
        device: torch.device = torch.device("cpu"),
        weight_decay: float = 0.0,
        base_optimizer: str = "sgd",
        momentum: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.base_sensitivity = base_sensitivity
        self.noise_multiplier = noise_multiplier
        self.device = device

        params_list = list(params)
        if clip_norms is not None:
            self.clip_norms = dict(clip_norms)
        elif clip_norm is not None:
            self.clip_norms = {p: float(clip_norm) for p in params_list if p.requires_grad}
        else:
            self.clip_norms = {p: 1.0 for p in params_list if p.requires_grad}

        if base_optimizer == "adam":
            self._base_opt = Adam(params_list, lr=lr, weight_decay=weight_decay)
        else:
            self._base_opt = SGD(
                params_list,
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay,
            )

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._base_opt.zero_grad(set_to_none=set_to_none)

    def set_clip_norms(self, clip_norms: Dict[nn.Parameter, float]) -> None:
        """Update per-parameter clip thresholds (e.g. after percentile estimation)."""
        self.clip_norms = dict(clip_norms)

    def step_from_aggregated_grads(self, aggregated_grads: Dict[nn.Parameter, Tensor]) -> None:
        """Set param.grad from aggregated_grads and call base optimizer step."""
        for p in self.param_groups[0]["params"]:
            if p in aggregated_grads:
                if p.grad is None:
                    p.grad = aggregated_grads[p].detach().clone()
                else:
                    p.grad.copy_(aggregated_grads[p].detach())
        self._base_opt.step()

    def step(
        self,
        closure: Optional[Iterable[Dict[nn.Parameter, Tensor]]] = None,
    ) -> Optional[float]:
        """
        If closure is None, no-op. Otherwise closure should be an iterable of
        per-example gradient dicts (param -> grad). We clip, aggregate, add noise, then step.
        """
        if closure is None:
            return None
        per_example = list(closure)
        if not per_example:
            return None
        aggregated = clip_and_aggregate(
            per_example,
            self.clip_norms,
            self.base_sensitivity,
            self.noise_multiplier,
            self.device,
        )
        self.step_from_aggregated_grads(aggregated)
        return None
