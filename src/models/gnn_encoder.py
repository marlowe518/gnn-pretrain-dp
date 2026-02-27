from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """
    Simple 2-layer GCN encoder to be shared between
    self-supervised pretraining and downstream tasks.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

