from typing import Tuple

import torch
from torch import Tensor
from torch_geometric.nn import DeepGraphInfomax

from .gnn_encoder import GCNEncoder


def _corruption(x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
    perm = torch.randperm(x.size(0))
    return x[perm], edge_index


def _summary(z: Tensor, *_args) -> Tensor:
    return torch.sigmoid(z.mean(dim=0))


def build_dgi_model(
    in_channels: int,
    hidden_channels: int,
) -> Tuple[DeepGraphInfomax, GCNEncoder]:
    """
    Construct a Deep Graph Infomax model with a shared GCN encoder.
    Returns (dgi_model, encoder).
    """
    encoder = GCNEncoder(in_channels, hidden_channels)
    model = DeepGraphInfomax(
        hidden_channels=hidden_channels,
        encoder=encoder,
        summary=_summary,
        corruption=_corruption,
    )
    return model, encoder

