"""
Bounded-degree neighbor sampling for node-DP GAP.
"""
from typing import Optional

import torch
from torch import Tensor


def sample_bounded_degree_edge_index(
    edge_index: Tensor,
    num_nodes: int,
    max_degree: int,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Return a new edge_index where each node has at most max_degree incoming neighbors.

    For aggregation we use (source, target) = (j, i) meaning j->i; so we limit
    incoming degree per node i to at most max_degree (neighbors j that feed into i).

    Args:
        edge_index: [2, E] with (source, target) = (j, i) for edge j->i
        num_nodes: number of nodes
        max_degree: maximum neighbors per node (incoming)
        seed: optional RNG seed
        undirected: if True, build symmetric edges (each direction capped)
    """
    device = edge_index.device
    src, dst = edge_index[0].cpu(), edge_index[1].cpu()
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    # Incoming neighbors: for each node i, list of j such that (j,i) in edge_index
    incoming: list[list[int]] = [[] for _ in range(num_nodes)]
    for j, i in zip(src.tolist(), dst.tolist()):
        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            incoming[i].append(j)

    new_src, new_dst = [], []
    for i in range(num_nodes):
        nbrs = incoming[i]
        if not nbrs:
            continue
        k = min(len(nbrs), max_degree)
        perm = torch.randperm(len(nbrs), generator=generator)
        for idx in range(k):
            j = nbrs[perm[idx].item()]
            new_src.append(j)
            new_dst.append(i)

    if not new_src:
        return edge_index

    out = torch.stack(
        [
            torch.tensor(new_src, dtype=torch.long, device=device),
            torch.tensor(new_dst, dtype=torch.long, device=device),
        ],
        dim=0,
    )
    return out
