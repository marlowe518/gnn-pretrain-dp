"""
Reusable DP-SGD optimizer wrapper.

Implements per-microbatch gradient clipping and Gaussian noise addition while exposing
an API similar to torch.optim.Optimizer.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import torch
from torch import Tensor, nn
from torch.optim import SGD, Optimizer


class DPSGD(Optimizer):
    """
    DP-SGD optimizer wrapper supporting per-microbatch clipping and Gaussian noise.

    Expects the trainer to provide a list of microbatch closures. Each closure:
      - Runs forward on a microbatch and returns a scalar loss.
      - MUST NOT call backward; DPSGD.step will call loss.backward() itself.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        clip_norm: float,
        noise_multiplier: float,
        microbatch_size: int,
        device: torch.device,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if clip_norm <= 0.0:
            raise ValueError("clip_norm must be > 0 for DPSGD.")
        if microbatch_size <= 0:
            raise ValueError("microbatch_size must be > 0 for DPSGD.")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)
        self.clip_norm = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.microbatch_size = int(microbatch_size)
        self.device = device
        # Underlying non-DP optimizer that will consume the privatized gradients.
        self._base_opt = SGD(self.param_groups, **defaults)

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        self._base_opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _clip_and_accumulate(
        self,
        accumulators: dict[nn.Parameter, Tensor],
    ) -> None:
        params: List[nn.Parameter] = [
            p for group in self.param_groups for p in group["params"] if p.requires_grad
        ]
        grads: List[Tensor] = [
            p.grad for p in params if p.grad is not None and p.grad.numel() > 0
        ]
        if not grads:
            return
        total_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2)
        clip_coef = (self.clip_norm / (total_norm + 1e-6)).clamp(max=1.0)
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach().mul(clip_coef)
            if p not in accumulators:
                accumulators[p] = torch.zeros_like(g, device=self.device)
            accumulators[p].add_(g.to(self.device))

    @torch.no_grad()
    def _add_noise_and_set_grads(
        self,
        accumulators: dict[nn.Parameter, Tensor],
        num_microbatches: int,
    ) -> None:
        if num_microbatches <= 0:
            return
        params: List[nn.Parameter] = [
            p for group in self.param_groups for p in group["params"] if p.requires_grad
        ]
        for p in params:
            if p not in accumulators:
                continue
            g = accumulators[p] / float(num_microbatches)
            if self.noise_multiplier > 0.0:
                noise = torch.randn_like(g, device=g.device) * (self.noise_multiplier * self.clip_norm)
                g = g + noise
            if p.grad is None:
                p.grad = torch.zeros_like(g, device=g.device)
            p.grad.copy_(g)

    def step(  # type: ignore[override]
        self,
        closure: Optional[Callable[[], float]] = None,
        *,
        loss_closures: Optional[List[Callable[[], Tensor]]] = None,
        num_microbatches: Optional[int] = None,
    ) -> Optional[float]:
        """
        Run DP-SGD step over a list of microbatch closures.

        Args:
            closure: unused (for Optimizer compatibility).
            loss_closures: list of functions returning a scalar loss for each microbatch.
            num_microbatches: expected number of microbatches (len(loss_closures)).
        Returns:
            Average loss over microbatches (Python float) if loss_closures is provided, else None.
        """
        if loss_closures is None or not loss_closures:
            # Nothing to do; fall back to base optimizer if a single closure is given.
            if closure is not None:
                loss = closure()
                loss.backward()
                self._base_opt.step()
                return float(loss.item())
            return None

        if num_microbatches is None:
            num_microbatches = len(loss_closures)

        accumulators: dict[nn.Parameter, Tensor] = {}
        total_loss = 0.0
        effective_mbs = 0

        for mb_closure in loss_closures:
            self._base_opt.zero_grad(set_to_none=True)
            loss = mb_closure()
            if not torch.is_tensor(loss):
                raise RuntimeError("Microbatch closure must return a torch.Tensor loss.")
            loss.backward()
            self._clip_and_accumulate(accumulators)
            total_loss += float(loss.item())
            effective_mbs += 1

        self._add_noise_and_set_grads(accumulators, num_microbatches=effective_mbs)
        self._base_opt.step()

        return total_loss / max(effective_mbs, 1)

