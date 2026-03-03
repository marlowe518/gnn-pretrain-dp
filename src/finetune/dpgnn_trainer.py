"""
DP-GNN finetuning trainer (PyTorch/PyG reimplementation).

Aligned with upstream: differentially_private_gnns/train.py
- Subgraph-based forward, loss only on root node.
- Per-example gradients, DPOptimizer (clip + aggregate + noise).
- Privacy accountant (Poisson for hops=0, multi-term for hops>=1).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data

from src.dp.dpgnn_sampler import DPGNNSampler
from src.dp.dp_optimizer import DPOptimizer, clip_and_aggregate
from src.dp.privacy_accountant_dpgnn import make_accountant
from src.eval.metrics import accuracy
from src.utils.trainer import BaseTrainer


def _compute_base_sensitivity(hops: int, max_degree: int) -> float:
    """Upstream: train.compute_base_sensitivity. Base sensitivity for noise scaling."""
    if hops == 0:
        return 1.0
    if hops == 1:
        return float(2 * (max_degree + 1))
    if hops == 2:
        return float(2 * (max_degree ** 2 + max_degree + 1))
    raise ValueError(f"Unsupported hops: {hops}")


def _compute_max_terms_per_node(hops: int, max_degree: int) -> int:
    """Upstream: train.compute_max_terms_per_node. For multi-term accountant."""
    if hops == 0:
        return 1
    if hops == 1:
        return max_degree + 1
    if hops == 2:
        return max_degree ** 2 + max_degree + 1
    raise ValueError(f"Unsupported hops: {hops}")


class DPGNNTrainer(BaseTrainer):
    """
    Differentially private GNN finetuning: subgraph sampling + per-example DP-SGD.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        data: Data,
        lr: float,
        weight_decay: float,
        device: torch.device,
        logger: Optional[callable] = None,
        *,
        dpgnn_hops: int = 1,
        dpgnn_max_degree: int = 10,
        dpgnn_pad_to: int = 0,
        dpgnn_dp_batch_size: int = 256,
        dpgnn_dp_microbatch_size: int = 1,
        dpgnn_noise_multiplier: float = 1.0,
        dpgnn_delta: float = "auto",
        dpgnn_clip_mode: str = "fixed",
        dpgnn_clip_norm: float = 1.0,
        dpgnn_clip_percentile: float = 95.0,
        dpgnn_num_estimation_samples: int = 500,
        dpgnn_max_epsilon: float = 10.0,
        dpgnn_train_encoder: bool = False,
        dpgnn_optimizer: str = "sgd",
    ) -> None:
        self.encoder = encoder.to(device)
        self.data = data.to(device)
        self.device = device
        self.logger = logger
        self.num_classes = num_classes
        self._last_train_loss = 0.0
        self._current_epsilon = 0.0

        self.dpgnn_hops = dpgnn_hops
        self.dpgnn_max_degree = dpgnn_max_degree
        self.dpgnn_pad_to = dpgnn_pad_to
        self.dpgnn_dp_batch_size = dpgnn_dp_batch_size
        self.dpgnn_dp_microbatch_size = dpgnn_dp_microbatch_size
        self.dpgnn_noise_multiplier = dpgnn_noise_multiplier
        self.dpgnn_delta = dpgnn_delta
        self.dpgnn_clip_mode = dpgnn_clip_mode
        self.dpgnn_clip_norm = dpgnn_clip_norm
        self.dpgnn_clip_percentile = dpgnn_clip_percentile
        self.dpgnn_num_estimation_samples = dpgnn_num_estimation_samples
        self.dpgnn_max_epsilon = dpgnn_max_epsilon
        self.dpgnn_train_encoder = dpgnn_train_encoder
        self.dpgnn_optimizer = dpgnn_optimizer

        # Hidden dim from encoder (GCNEncoder has conv2.out_channels)
        hidden_dim = getattr(encoder, "conv2", None)
        if hidden_dim is not None:
            hidden_dim = getattr(hidden_dim, "out_channels", None)
        if hidden_dim is None:
            hidden_dim = getattr(encoder, "out_channels", 64)
        self.hidden_dim = int(hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, num_classes).to(device)

        if not dpgnn_train_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        params = list(self.classifier.parameters())
        if dpgnn_train_encoder:
            params = list(self.encoder.parameters()) + params

        self.base_sensitivity = _compute_base_sensitivity(dpgnn_hops, dpgnn_max_degree)
        self.max_terms_per_node = _compute_max_terms_per_node(dpgnn_hops, dpgnn_max_degree)

        # Clip norms: fixed or will be set after percentile estimation
        self.clip_norms: Dict[nn.Parameter, float] = {p: dpgnn_clip_norm for p in params if p.requires_grad}

        self.dp_optimizer = DPOptimizer(
            params,
            lr=lr,
            clip_norm=dpgnn_clip_norm,
            base_sensitivity=self.base_sensitivity,
            noise_multiplier=dpgnn_noise_multiplier,
            device=device,
            weight_decay=weight_decay,
            base_optimizer=dpgnn_optimizer,
        )

        n_train = int(self.data.train_mask.sum().item())
        delta = dpgnn_delta if isinstance(dpgnn_delta, (int, float)) else (1.0 / (10.0 * n_train))
        accountant_mode = "poisson" if dpgnn_hops == 0 else "multiterm"
        self.get_epsilon = make_accountant(
            accountant_mode,
            n_train,
            dpgnn_dp_batch_size,
            dpgnn_noise_multiplier,
            self.max_terms_per_node,
        )

        self.sampler = DPGNNSampler(
            self.data.edge_index,
            self.data.num_nodes,
            self.data.train_mask,
            dpgnn_max_degree,
            num_hops=dpgnn_hops,
            pad_to=dpgnn_pad_to,
            device=device,
        )

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger(msg)

    def _forward_subgraph(self, sub_x: Tensor, sub_edge_index: Tensor) -> Tensor:
        """Forward on one subgraph; returns node embeddings (root at index 0)."""
        z = self.encoder(sub_x, sub_edge_index)
        return z

    def _loss_at_root(self, sub_x: Tensor, sub_edge_index: Tensor, root_label: Tensor) -> Tensor:
        """Loss for one root: forward subgraph, take logits at root (index 0), CE."""
        z = self._forward_subgraph(sub_x, sub_edge_index)
        root_z = z[0:1]
        logits = self.classifier(root_z)
        return F.cross_entropy(logits, root_label)

    def _estimate_clip_percentiles(self, seed: int = 0) -> None:
        """Upstream: estimate_clipping_thresholds. Sample training nodes, compute per-layer grad norms, set percentiles."""
        torch.manual_seed(seed)
        train_idx = self.data.train_mask.nonzero(as_tuple=False).squeeze(-1)
        n = min(self.dpgnn_num_estimation_samples, train_idx.size(0))
        indices = train_idx[torch.randperm(train_idx.size(0), device=self.device)[:n]]

        self.encoder.train()
        self.classifier.train()
        per_param_norms: Dict[nn.Parameter, List[float]] = {p: [] for p in self.clip_norms}

        for i in range(n):
            root = indices[i].item()
            subs = self.sampler.get_subgraph(
                torch.tensor([root], device=self.device, dtype=torch.long),
                self.data.x,
            )
            if not subs:
                continue
            sub_x, sub_edge_index, _ = subs[0]
            self.dp_optimizer.zero_grad()
            loss = self._loss_at_root(sub_x, sub_edge_index, self.data.y[root : root + 1])
            loss.backward()

            for p in self.clip_norms:
                if p.grad is not None and p.requires_grad:
                    norm = p.grad.detach().norm(2).item()
                    per_param_norms[p].append(norm)

        # Percentile per parameter
        import numpy as np
        for p, norms in per_param_norms.items():
            if norms:
                clip_val = float(np.percentile(norms, self.dpgnn_clip_percentile))
                self.clip_norms[p] = max(clip_val, 1e-6)
                self._log(f"[DP-GNN] clip_est param {id(p)} percentile={self.dpgnn_clip_percentile} -> {clip_val:.4f}")
        self.dp_optimizer.set_clip_norms(self.clip_norms)

    def train(self, num_epochs: int) -> Dict[str, float]:
        """Upstream: train loop with batch sampling, per-example grads, DP step, epsilon check."""
        n_train = int(self.data.train_mask.sum().item())
        train_idx = self.data.train_mask.nonzero(as_tuple=False).squeeze(-1)
        steps_per_epoch = max(1, (n_train + self.dpgnn_dp_batch_size - 1) // self.dpgnn_dp_batch_size)
        total_steps = 0

        train_metrics = self.evaluate("train")
        val_metrics = self.evaluate("val")

        if self.dpgnn_clip_mode == "percentile":
            t0 = time.perf_counter()
            self.sampler.resample(seed=0)
            self._estimate_clip_percentiles(seed=42)
            self._log(f"[DP-GNN] clip percentile estimation done in {time.perf_counter() - t0:.2f}s")

        for epoch in range(1, num_epochs + 1):
            self.sampler.resample(seed=epoch)
            dropped = self.sampler.dropped_count
            self._log(f"[DP-GNN] Epoch {epoch} resampled adjacency, dropped_nodes={dropped}")

            perm = torch.randperm(n_train, device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for b in range(0, n_train, self.dpgnn_dp_batch_size):
                batch_idx = train_idx[perm[b : b + self.dpgnn_dp_batch_size]]
                roots = batch_idx

                # Per-example gradients: one subgraph per root, loss at root
                per_example_grads: List[Dict[nn.Parameter, Tensor]] = []
                batch_losses = []

                for start in range(0, roots.size(0), self.dpgnn_dp_microbatch_size):
                    mb = roots[start : start + self.dpgnn_dp_microbatch_size]
                    subs = self.sampler.get_subgraph(mb, self.data.x)
                    for i, (sub_x, sub_edge_index, _) in enumerate(subs):
                        if start + i >= roots.size(0):
                            break
                        root = roots[start + i].item()
                        self.dp_optimizer.zero_grad()
                        loss = self._loss_at_root(sub_x, sub_edge_index, self.data.y[root : root + 1])
                        loss.backward()
                        batch_losses.append(loss.item())
                        grad_dict = {p: p.grad.clone() for p in self.dp_optimizer.param_groups[0]["params"] if p.grad is not None and p.requires_grad}
                        per_example_grads.append(grad_dict)

                if not per_example_grads:
                    continue

                # Aggregate with clip + noise and step (upstream: dp_aggregate then update)
                aggregated = clip_and_aggregate(
                    per_example_grads,
                    self.dp_optimizer.clip_norms,
                    self.base_sensitivity,
                    self.dpgnn_noise_multiplier,
                    self.device,
                )
                self.dp_optimizer.step_from_aggregated_grads(aggregated)

                total_steps += 1
                self._current_epsilon = self.get_epsilon(total_steps)
                if batch_losses:
                    epoch_loss += sum(batch_losses) / len(batch_losses)
                n_batches += 1

                if self._current_epsilon >= self.dpgnn_max_epsilon:
                    self._log(f"[DP-GNN] Epsilon {self._current_epsilon:.4f} >= max_epsilon {self.dpgnn_max_epsilon}, stopping.")
                    break

            if self._current_epsilon >= self.dpgnn_max_epsilon:
                break

            self._last_train_loss = epoch_loss / max(n_batches, 1)
            train_metrics = self.evaluate("train")
            val_metrics = self.evaluate("val")
            self._log(
                f"[DP-GNN] Epoch {epoch:03d} | loss={self._last_train_loss:.4f} | "
                f"train_acc={train_metrics['accuracy']:.3f} | val_acc={val_metrics['accuracy']:.3f} | "
                f"epsilon={self._current_epsilon:.4f}"
            )

        return {
            "train_loss": self._last_train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "epsilon": self._current_epsilon,
        }

    def evaluate(self, split: str) -> Dict[str, float]:
        """Evaluate on full graph (consistent choice; document in docstring)."""
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
        """One DP step for dry_run."""
        self.sampler.resample(seed=0)
        train_idx = self.data.train_mask.nonzero(as_tuple=False).squeeze(-1)
        roots = train_idx[: self.dpgnn_dp_batch_size]
        subs = self.sampler.get_subgraph(roots, self.data.x)
        per_example_grads = []
        for i, (sub_x, sub_edge_index, _) in enumerate(subs):
            if i >= roots.size(0):
                break
            root = roots[i].item()
            self.dp_optimizer.zero_grad()
            loss = self._loss_at_root(sub_x, sub_edge_index, self.data.y[root : root + 1])
            loss.backward()
            grad_dict = {p: p.grad.clone() for p in self.dp_optimizer.param_groups[0]["params"] if p.grad is not None and p.requires_grad}
            per_example_grads.append(grad_dict)
        last_loss = 0.0
        if per_example_grads:
            aggregated = clip_and_aggregate(
                per_example_grads,
                self.dp_optimizer.clip_norms,
                self.base_sensitivity,
                self.dpgnn_noise_multiplier,
                self.device,
            )
            self.dp_optimizer.step_from_aggregated_grads(aggregated)
            last_loss = loss.item()
        self._last_train_loss = last_loss
        self._current_epsilon = self.get_epsilon(1)
        train_metrics = self.evaluate("train")
        grad_norm = 0.0
        for p in self.dp_optimizer.param_groups[0]["params"]:
            if p.grad is not None:
                grad_norm += p.grad.detach().norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        self._log(f"[DP-GNN dry] loss={self._last_train_loss:.4f}, epsilon={self._current_epsilon:.4f}, grad_norm={grad_norm:.4f}")
        return {"train_loss": self._last_train_loss, "grad_norm": grad_norm}
