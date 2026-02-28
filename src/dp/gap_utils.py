"""
GAP-specific utilities: PMA import and aggregation helpers.
Isolates dependency on external/GAP.
"""
from pathlib import Path
import sys

# Allow importing from external GAP (core.privacy, etc.)
_GAP_ROOT = Path(__file__).resolve().parent.parent.parent / "external" / "GAP"
if _GAP_ROOT.exists() and str(_GAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_GAP_ROOT))


def get_pma_class():
    """Return GAP's PMA class (lazy import to avoid loading GAP until needed)."""
    from core.privacy.algorithms.graph.pma import PMA
    return PMA


def sparse_aggregate(edge_index, x, num_nodes):
    """
    Compute aggregation (adj_t @ x) using PyTorch sparse, so we don't require torch_sparse.
    edge_index: [2, E] with (source, target); output[i] = sum of x[j] for j in N_in(i).
    """
    import torch
    device = x.device
    n, d = x.size(0), x.size(1)
    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, d, device=device, dtype=x.dtype)
    # adj_t @ x: (adj_t)[i,j] = 1 if edge j->i, so row = target, col = source
    row = edge_index[1]  # target
    col = edge_index[0]  # source
    indices = torch.stack([row, col], dim=0)
    values = torch.ones(edge_index.size(1), device=device, dtype=x.dtype)
    adj_t = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
    out = torch.sparse.mm(adj_t, x)
    return out
