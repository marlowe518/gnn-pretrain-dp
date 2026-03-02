"""
DP-SGD helpers: per-sample gradient clipping and noise for node-DP.
"""
from typing import Dict, Iterable

import torch
from torch import Tensor, nn


def _iter_trainable_params(params: Iterable[nn.Parameter]):
    return [p for p in params if p.requires_grad]


def clip_and_accumulate_grads(
    params: Iterable[nn.Parameter],
    accumulators: Dict[nn.Parameter, Tensor],
    max_norm: float,
) -> None:
    """
    Clip current grads of params to max_norm (L2) and add into accumulators.
    """
    params = _iter_trainable_params(params)
    grads = [p.grad for p in params if p.grad is not None and p.grad.numel() > 0]
    if not grads:
        return
    total_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2)
    clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach().mul(clip_coef)
        if p not in accumulators:
            accumulators[p] = torch.zeros_like(g)
        accumulators[p].add_(g)


def add_noise_and_set_grads(
    params: Iterable[nn.Parameter],
    accumulators: Dict[nn.Parameter, Tensor],
    max_norm: float,
    noise_multiplier: float,
    num_microbatches: int,
) -> None:
    """
    Add Gaussian noise to accumulated clipped grads, then set param.grad for optimizer step.
    """
    params = _iter_trainable_params(params)
    for p in params:
        if p not in accumulators:
            continue
        g = accumulators[p] / float(num_microbatches)
        if noise_multiplier > 0.0:
            noise = torch.randn_like(g, device=g.device) * (noise_multiplier * max_norm)
            g = g + noise
        p.grad = g
