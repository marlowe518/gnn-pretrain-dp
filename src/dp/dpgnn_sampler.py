"""
DP-GNN subgraph sampler (PyTorch/PyG reimplementation).

Semantics aligned with upstream: differentially_private_gnns/sampler.py
- sample_adjacency_lists: degree-bounded sampling (Bernoulli over edges, in-degree from
  train nodes bounded by max_degree; nodes exceeding after sampling are dropped).
- get_subgraphs: for each node, subgraph = node + outgoing neighbors (up to hops), padded.

We build adjacency from data.edge_index (cached on CPU), then provide get_subgraph(root_nodes)
returning (sub_nodes, sub_edge_index, root_position=0) per root, with optional padding.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

# Subgraph padding value (upstream _SUBGRAPH_PADDING_VALUE = -1)
PAD_VALUE = -1


def _build_adjacency_from_edge_index(
    edge_index: Tensor,
    num_nodes: int,
) -> List[List[int]]:
    """
    Build outgoing adjacency lists: adj[u] = list of v such that (u,v) in edges.
    Upstream: get_adjacency_lists (senders -> receivers).
    """
    src, dst = edge_index[0].cpu().tolist(), edge_index[1].cpu().tolist()
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        if 0 <= u < num_nodes and 0 <= v < num_nodes and u != v:
            adj[u].append(v)
    return adj


def _reverse_edges(adj: List[List[int]], num_nodes: int) -> List[List[int]]:
    """Incoming adjacency: rev[v] = list of u with (u,v). Upstream: reverse_edges."""
    rev: List[List[int]] = [[] for _ in range(num_nodes)]
    for u in range(num_nodes):
        for v in adj[u]:
            rev[v].append(u)
    return rev


def sample_adjacency_lists(
    edge_index: Tensor,
    num_nodes: int,
    train_nodes: Tensor,
    max_degree: int,
    seed: Optional[int] = None,
) -> Tuple[List[List[int]], int]:
    """
    Degree-bounded sampling of adjacency lists (upstream: sampler.sample_adjacency_lists).

    For each node we bound in-degree from *training* nodes: Bernoulli sampling with
    sampling_prob = max_degree / (2 * in_degree). If after sampling a node has more than
    max_degree incoming train neighbors, that node is dropped (not used for in-edges).

    Returns:
        sampled_adj: outgoing adjacency list (list of lists), same format as built from edge_index.
        dropped_count: number of nodes that exceeded max_degree and were dropped.
    """
    adj = _build_adjacency_from_edge_index(edge_index, num_nodes)
    rev = _reverse_edges(adj, num_nodes)
    train_set = set(train_nodes.cpu().tolist())

    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(seed)

    # For each node u, sample incoming edges from train nodes so in-degree <= max_degree.
    sampled_rev: List[List[int]] = [[] for _ in range(num_nodes)]
    dropped_count = 0

    for u in range(num_nodes):
        incoming = rev[u]
        incoming_train = [v for v in incoming if v in train_set]
        if not incoming_train:
            continue

        in_degree = len(incoming_train)
        # Upstream: sampling_prob = max_degree / (2 * in_degree)
        sampling_prob = min(1.0, max_degree / (2.0 * in_degree))
        mask = torch.rand(in_degree, generator=rng) < sampling_prob
        selected = [incoming_train[i] for i in range(in_degree) if mask[i].item()]
        unique = list(dict.fromkeys(selected))  # preserve order, unique

        if len(unique) <= max_degree:
            sampled_rev[u] = unique
        else:
            dropped_count += 1

    # Convert back to outgoing adjacency (reverse again).
    sampled_adj = _reverse_edges(sampled_rev, num_nodes)

    # Non-train nodes: keep full outgoing list (upstream: for u not in train_nodes, sampled_edges[u] = edges[u]).
    for u in range(num_nodes):
        if u not in train_set:
            sampled_adj[u] = adj[u]

    return sampled_adj, dropped_count


def get_subgraph(
    root_nodes: Tensor,
    x: Tensor,
    adj: List[List[int]],
    num_hops: int,
    pad_to: int,
    device: torch.device,
) -> List[Tuple[Tensor, Tensor, int]]:
    """
    Build one subgraph per root (upstream: get_subgraphs + make_subgraph_from_indices).

    For each root we take root + k-hop outgoing neighborhood, remap to 0..n_sub-1,
    root always at 0. Optional padding to pad_to nodes (with PAD_VALUE indices).

    Returns:
        List of (sub_x, sub_edge_index, root_position) with root_position=0.
        sub_x: [n_sub, F], sub_edge_index: [2, E_sub].
    """
    assert num_hops in (0, 1, 2), "dpgnn_hops must be 0, 1, or 2"
    n_total = x.size(0)
    feat_dim = x.size(1)
    results: List[Tuple[Tensor, Tensor, int]] = []

    for root in root_nodes.cpu().tolist():
        root = int(root)
        if root < 0 or root >= n_total:
            continue

        # Collect subgraph node indices: root + neighbors up to num_hops (BFS).
        sub_indices = [root]
        frontier = [root]
        for _ in range(num_hops):
            next_frontier = []
            for u in frontier:
                for v in adj[u]:
                    if v not in sub_indices:
                        sub_indices.append(v)
                        next_frontier.append(v)
            frontier = next_frontier

        if pad_to > 0 and len(sub_indices) < pad_to:
            sub_indices = sub_indices + [PAD_VALUE] * (pad_to - len(sub_indices))
        elif pad_to > 0 and len(sub_indices) > pad_to:
            sub_indices = sub_indices[:pad_to]

        # Build node index -> local index (0 = root).
        node_to_local = {node: i for i, node in enumerate(sub_indices) if node != PAD_VALUE}
        if PAD_VALUE in sub_indices:
            pad_local = len(node_to_local)
            node_to_local[PAD_VALUE] = pad_local

        n_sub = len(sub_indices)
        # Node features: valid nodes get x[node], padding gets 0.
        sub_x = torch.zeros(n_sub, feat_dim, dtype=x.dtype, device=device)
        for i, node in enumerate(sub_indices):
            if node != PAD_VALUE:
                sub_x[i] = x[node].to(device)

        # Edges: only between valid nodes (and optionally padding node as sink).
        # Upstream: subgraph_senders, subgraph_receivers from subgraph_indices; root has edges to neighbors.
        # We have sub_indices = [root, n1, n2, ...]. Outgoing from root in original adj: root->v -> local 0 -> local node_to_local[v].
        sub_edges_src: List[int] = []
        sub_edges_dst: List[int] = []
        for i, u_global in enumerate(sub_indices):
            if u_global == PAD_VALUE:
                continue
            for v_global in adj[u_global]:
                if v_global not in node_to_local:
                    continue
                j = node_to_local[v_global]
                sub_edges_src.append(i)
                sub_edges_dst.append(j)

        if sub_edges_src:
            sub_edge_index = torch.stack(
                [
                    torch.tensor(sub_edges_src, dtype=torch.long, device=device),
                    torch.tensor(sub_edges_dst, dtype=torch.long, device=device),
                ],
                dim=0,
            )
        else:
            sub_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

        results.append((sub_x, sub_edge_index, 0))

    return results


class DPGNNSampler:
    """
    Caches adjacency and (optionally) degree-bounded sampled adjacency for DP-GNN.
    Provides get_subgraph(root_nodes) returning list of (sub_x, sub_edge_index, root_position).
    """

    def __init__(
        self,
        edge_index: Tensor,
        num_nodes: int,
        train_mask: Tensor,
        max_degree: int,
        num_hops: int = 1,
        pad_to: int = 0,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None,
    ) -> None:
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.train_mask = train_mask
        self.max_degree = max_degree
        self.num_hops = num_hops
        self.pad_to = pad_to
        self.device = device
        self._adj: Optional[list] = None  # list of list of int (outgoing adjacency)
        self._dropped_count = 0
        self._seed = seed

    def resample(self, seed: Optional[int] = None) -> int:
        """
        (Re)compute degree-bounded adjacency. Returns dropped_count.
        Upstream: call sample_adjacency_lists each epoch if desired.
        """
        s = seed if seed is not None else self._seed
        train_nodes = self.train_mask.nonzero(as_tuple=False).squeeze(-1)
        self._adj, self._dropped_count = sample_adjacency_lists(
            self.edge_index,
            self.num_nodes,
            train_nodes,
            self.max_degree,
            seed=s,
        )
        return self._dropped_count

    def get_adjacency(self) -> List[List[int]]:
        """Return current adjacency (sample first if not yet)."""
        if self._adj is None:
            self.resample()
        assert self._adj is not None
        return self._adj

    def get_subgraph(self, root_nodes: Tensor, x: Tensor) -> List[Tuple[Tensor, Tensor, int]]:
        """Return list of (sub_x, sub_edge_index, root_position=0) for each root."""
        adj = self.get_adjacency()
        return get_subgraph(
            root_nodes,
            x,
            adj,
            self.num_hops,
            self.pad_to,
            self.device,
        )

    @property
    def dropped_count(self) -> int:
        return self._dropped_count
