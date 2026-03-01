"""
GAP (Graph Aggregation Perturbation) DP finetuning trainer.
Reuses PMA from external/GAP for edge-level DP on aggregation.
"""
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam
from torch_geometric.data import Data

from src.dp.gap_utils import get_pma_class, sparse_aggregate
from src.eval.metrics import accuracy
from src.models.gnn_encoder import GCNEncoder
from src.utils.trainer import BaseTrainer


class GAPFinetuneTrainer(BaseTrainer):
    """
    DP finetuning via GAP: pretrained encoder + PMA-perturbed aggregation + classifier.
    Encoder weights are loaded (from caller); aggregation uses GAP's PMA for edge DP.
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
        *,
        epsilon: float = 1.0,
        delta: Union[float, str] = "auto",
        hops: int = 2,
        gap_debug: bool = False,
        gap_debug_strict: bool = False,
        gap_debug_resample_check: bool = False,
    ) -> None:
        self.encoder = encoder.to(device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.data = data.to(device)
        self.device = device
        self.logger = logger
        self.epsilon = epsilon
        self.delta = delta
        self.hops = hops
        self._last_train_loss: float = 0.0
        # Debug-related flags (do not change default behaviour when False).
        self.gap_debug = bool(gap_debug)
        self.gap_debug_strict = bool(gap_debug_strict and gap_debug)
        self.gap_debug_resample_check = bool(gap_debug_resample_check and gap_debug)
        self._debug_resample_done: bool = False

        hidden_dim = encoder.conv2.out_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self._aggregated_x: Optional[Tensor] = None

        PMA = get_pma_class()
        self.pma = PMA(noise_scale=0.0, hops=hops)
        num_edges = self.data.edge_index.size(1)
        if delta == "auto":
            delta_val = 0.0 if (epsilon == float("inf") or epsilon > 1e10) else 1.0 / (10 ** len(str(num_edges)))
        else:
            delta_val = float(delta)
        self.pma.noise_scale = self.pma.calibrate(eps=epsilon, delta=delta_val)
        if self.logger:
            self.logger(f"[GAP] PMA calibrated: epsilon={epsilon}, delta={delta_val}, noise_scale={self.pma.noise_scale:.4f}")

        self.classifier = nn.Linear((hops + 1) * hidden_dim, num_classes).to(device)
        self.optimizer = Adam(self.classifier.parameters(), lr=lr, weight_decay=weight_decay)

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger(msg)

    def _debug_log(self, msg: str) -> None:
        """
        Debug logging helper that falls back to print when logger is absent.
        """
        if not (self.gap_debug or self.gap_debug_strict or self.gap_debug_resample_check):
            return
        prefix = "[GAP-DEBUG] "
        if self.logger is not None:
            self.logger(prefix + msg)
        else:
            print(prefix + msg)

    def _compute_aggregations(self) -> Tensor:
        """Encoder forward + multi-hop aggregation with PMA. Returns [n, hidden_dim, hops+1]."""
        self.encoder.eval()
        with torch.no_grad():
            x = self.encoder(self.data.x, self.data.edge_index)
        x = F.normalize(x, p=2, dim=-1)
        n = x.size(0)
        edge_index = self.data.edge_index

        # Fast path: no debug instrumentation, keeps behaviour identical.
        if not (self.gap_debug or self.gap_debug_strict or self.gap_debug_resample_check):
            x_list = [x]
            for _ in range(self.hops):
                x = sparse_aggregate(edge_index, x, n)
                x = self.pma(x, sensitivity=1.0)
                x = F.normalize(x, p=2, dim=-1)
                x_list.append(x)
            return torch.stack(x_list, dim=-1)

        # Debug path: collect clean vs noisy stats, but still feed noisy stack
        # into the classifier exactly as in the fast path.
        clean_list = [x]
        noisy_list = [x]
        prev = x

        for hop in range(self.hops):
            # Clean aggregated representation (before this hop's noise).
            z_clean = sparse_aggregate(edge_index, prev, n)

            if self.gap_debug:
                norms = z_clean.norm(dim=1)
                mean_norm = norms.mean().item()
                std_norm = norms.std().item()
                max_norm = norms.max().item()
                self._debug_log(
                    f"hop={hop} clean agg: mean_norm={mean_norm:.6f}, "
                    f"std_norm={std_norm:.6f}, max_norm={max_norm:.6f}, "
                    f"shape={tuple(z_clean.shape)}, device={z_clean.device}, "
                    f"dtype={z_clean.dtype}"
                )

            # Optional resampling sanity check: privatize twice without
            # affecting RNG state seen by the main path.
            if self.gap_debug_resample_check and not self._debug_resample_done:
                # import torch.cuda

                cpu_state = torch.random.get_rng_state()
                cuda_states = None
                if torch.cuda.is_available():
                    cuda_states = torch.cuda.get_rng_state_all()
                with torch.no_grad():
                    z1 = self.pma(z_clean, sensitivity=1.0)
                    z2 = self.pma(z_clean, sensitivity=1.0)
                    diff = z1 - z2
                    mean_diff = diff.abs().mean().item()
                    std_diff = diff.std().item()
                    self._debug_log(
                        f"hop={hop} resample check: mean(|z1-z2|)={mean_diff:.6e}, "
                        f"std(z1-z2)={std_diff:.6e}"
                    )
                # Restore RNG state so main path samples are unchanged.
                torch.random.set_rng_state(cpu_state)
                if cuda_states is not None:
                    torch.cuda.set_rng_state_all(cuda_states)
                self._debug_resample_done = True

            # Apply PMA noise for this hop (main path).
            z_noisy = self.pma(z_clean, sensitivity=1.0)

            if self.gap_debug or self.gap_debug_strict:
                noise = z_noisy - z_clean
                noise_abs = noise.abs()
                noise_mean_abs = noise_abs.mean().item()
                noise_std = noise.std().item()
                noise_max = noise_abs.max().item()
                ratio = (
                    noise.norm(dim=1).mean()
                    / (z_clean.norm(dim=1).mean() + 1e-8)
                ).item()
                diff = z_noisy - z_clean
                diff_mean = diff.abs().mean().item()
                diff_std = diff.std().item()
                self._debug_log(
                    f"hop={hop} noise stats: mean_abs={noise_mean_abs:.6e}, "
                    f"std={noise_std:.6e}, max={noise_max:.6e}, "
                    f"ratio_noise_to_clean_norm={ratio:.6e}, "
                    f"mean(|z_noisy-z_clean|)={diff_mean:.6e}, "
                    f"std(z_noisy-z_clean)={diff_std:.6e}, "
                    f"shape={tuple(noise.shape)}, device={noise.device}, "
                    f"dtype={noise.dtype}"
                )
                if self.gap_debug_strict and noise_std < 1e-12:
                    raise RuntimeError(
                        "GAP debug strict: measured noise std is ~0; "
                        "PMA appears to be non-noisy."
                    )

            # Normalize both clean and noisy aggregations for comparison.
            z_clean_norm = F.normalize(z_clean, p=2, dim=-1)
            z_noisy_norm = F.normalize(z_noisy, p=2, dim=-1)

            clean_list.append(z_clean_norm)
            noisy_list.append(z_noisy_norm)
            prev = z_noisy_norm

        # Store clean stack only for debug assertions.
        self._debug_clean_stack = torch.stack(clean_list, dim=-1)
        return torch.stack(noisy_list, dim=-1)

    def _get_aggregated_features(self) -> Tensor:
        if self._aggregated_x is None:
            self._aggregated_x = self._compute_aggregations()
        return self._aggregated_x

    def _forward_logits(self) -> Tensor:
        feat = self._get_aggregated_features()
        flat = feat.permute(0, 2, 1).reshape(feat.size(0), -1)

        if self.gap_debug or self.gap_debug_strict:
            z_input = flat
            mean_val = z_input.mean().item()
            std_val = z_input.std().item()
            self._debug_log(
                f"classifier input: mean={mean_val:.6e}, std={std_val:.6e}, "
                f"shape={tuple(z_input.shape)}, device={z_input.device}, "
                f"dtype={z_input.dtype}"
            )
            if hasattr(self, "_debug_clean_stack"):
                clean_flat = self._debug_clean_stack.permute(0, 2, 1).reshape(
                    self._debug_clean_stack.size(0), -1
                )
                diff = z_input - clean_flat.to(z_input.device)
                diff_mean = diff.abs().mean().item()
                diff_std = diff.std().item()
                self._debug_log(
                    f"classifier input vs clean: mean_abs={diff_mean:.6e}, "
                    f"std={diff_std:.6e}"
                )
                if self.gap_debug_strict and diff_mean < 1e-12:
                    raise RuntimeError(
                        "GAP debug strict: Classifier is not receiving a "
                        "noisy representation (difference ~ 0)."
                    )

        return self.classifier(flat)

    def _step(self) -> float:
        self.classifier.train()
        self.optimizer.zero_grad()
        logits = self._forward_logits()
        train_mask = self.data.train_mask
        loss = F.cross_entropy(logits[train_mask], self.data.y[train_mask])
        loss.backward()
        self.optimizer.step()
        self._last_train_loss = float(loss.item())
        return self._last_train_loss

    def train(self, num_epochs: int) -> Dict[str, float]:
        self._aggregated_x = None
        self._aggregated_x = self._compute_aggregations()
        self._log(f"[GAP] Aggregations computed (hops={self.hops}, PMA applied). Training classifier.")
        last_loss = 0.0
        train_metrics = {}
        val_metrics = {}
        for epoch in range(1, num_epochs + 1):
            last_loss = self._step()
            train_metrics = self.evaluate("train")
            val_metrics = self.evaluate("val")
            self._log(
                "[GAP] Epoch "
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
        self.classifier.eval()
        mask = getattr(self.data, f"{split}_mask")
        with torch.no_grad():
            logits = self._forward_logits()
            split_logits = logits[mask]
            split_labels = self.data.y[mask]
            acc = accuracy(split_logits, split_labels)
        return {"accuracy": acc}

    def dry_run_debug_step(self) -> Dict[str, float]:
        """Single forward/backward step for dry_run; encoder is frozen."""
        self._aggregated_x = None
        self._aggregated_x = self._compute_aggregations()
        self.classifier.train()
        self.optimizer.zero_grad()
        logits = self._forward_logits()
        train_mask = self.data.train_mask
        loss = F.cross_entropy(logits[train_mask], self.data.y[train_mask])
        self._log(
            f"[GAP dry] aggregated shape={tuple(self._aggregated_x.shape)}, "
            f"logits shape={tuple(logits.shape)}, loss={loss.item():.4f}"
        )
        loss.backward()
        total_norm_sq: Tensor = torch.tensor(0.0, device=self.device)
        for p in self.classifier.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.detach().pow(2).sum()
        grad_norm = float(total_norm_sq.sqrt().item())
        self._log(f"[GAP dry] grad_norm={grad_norm:.4f}")
        self.optimizer.step()
        self._last_train_loss = float(loss.item())
        return {"train_loss": self._last_train_loss, "grad_norm": grad_norm}
