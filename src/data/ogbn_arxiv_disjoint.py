"""
OGBN-Arxiv-Disjoint loader for DP-GNN experiments.

Semantics aligned with differentially_private_gnns/dataset_readers.OGBDisjointDataset:
- Base dataset: ogbn-arxiv (transductive OGB node property prediction)
- Splits: train/validation/test from OGB
- Edges: keep only edges (u, v) where u and v are in the same split
"""

from __future__ import annotations

from typing import Tuple

import torch
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import Data


def load_ogbn_arxiv_disjoint(root: str = "data") -> Tuple[Data, int, int]:
    """
    Load ogbn-arxiv with disjoint train/val/test edges.

    Returns:
        data: PyG Data with x, edge_index, y, train/val/test masks
        num_features: input feature dimension
        num_classes: number of classes
    """
    dataset = NodePropPredDataset(name="ogbn-arxiv", root=root)
    graph, labels = dataset[0]

    # Node features and labels
    x = torch.as_tensor(graph["node_feat"], dtype=torch.float)
    y = torch.as_tensor(labels, dtype=torch.long).view(-1)

    # Original edges (directed)
    edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long)

    split_idx = dataset.get_idx_split()
    train_nodes = split_idx["train"]
    val_nodes = split_idx["valid"]
    test_nodes = split_idx["test"]

    num_nodes = x.size(0)
    node_to_split = torch.full((num_nodes,), -1, dtype=torch.long)
    node_to_split[train_nodes] = 0
    node_to_split[val_nodes] = 1
    node_to_split[test_nodes] = 2

    src, dst = edge_index[0], edge_index[1]
    src_split = node_to_split[src]
    dst_split = node_to_split[dst]
    mask = (src_split >= 0) & (src_split == dst_split)
    edge_index_disjoint = edge_index[:, mask]

    # Boolean masks for splits
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask[test_nodes] = True

    data = Data(
        x=x,
        edge_index=edge_index_disjoint,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    num_features = x.size(1)
    num_classes = int(dataset.num_classes)
    return data, num_features, num_classes

