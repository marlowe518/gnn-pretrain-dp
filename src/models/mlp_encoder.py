"""
MLP encoder (feature-only, no graph structure).
Compatible with GAP and future MLP pretrain (MLPInit / supervised MLP).
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MLPEncoder(nn.Module):
    """
    Multi-layer MLP encoder: feature-only, ignores edge_index.
    Same API as GCNEncoder for drop-in use: forward(x, edge_index=None).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim if out_dim is not None else hidden_dim
        self.dropout = dropout
        self.use_bn = use_bn

        layers: list[nn.Module] = []
        prev = in_channels
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            prev = hidden_dim
        layers.append(nn.Linear(prev, self.out_dim))
        self.layers = nn.ModuleList(layers)

        # For GAP compatibility: same attribute name as GCNEncoder.conv2.out_channels
        self.out_channels = self.out_dim

    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        """Forward on node features only; edge_index is ignored."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, nn.Linear) and i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def reset_parameters(self) -> None:
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
