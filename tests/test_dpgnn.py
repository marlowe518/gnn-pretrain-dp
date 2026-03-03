"""
Small tests for DP-GNN components: DPOptimizer (shape + noise) and privacy accountant (epsilon monotonicity).
"""

import numpy as np
import torch
from torch import nn

from src.dp.dp_optimizer import clip_and_aggregate
from src.dp.privacy_accountant_dpgnn import make_accountant, multiterm_rdp_epsilon, poisson_subsampling_rdp_epsilon


def test_dp_optimizer_shape_and_noise():
    """DPOptimizer / clip_and_aggregate: output shapes match params; noise std > 0 when noise_multiplier > 0."""
    device = torch.device("cpu")
    model = nn.Linear(4, 3)
    params = list(model.parameters())
    clip_norms = {p: 1.0 for p in params}

    # Two "per-example" grads
    g1 = {p: torch.randn_like(p) for p in params}
    g2 = {p: torch.randn_like(p) for p in params}
    per_example = [g1, g2]

    out_zero = clip_and_aggregate(per_example, clip_norms, base_sensitivity=1.0, noise_multiplier=0.0, device=device)
    out_noisy = clip_and_aggregate(per_example, clip_norms, base_sensitivity=1.0, noise_multiplier=0.5, device=device)

    for p in params:
        assert p in out_zero and out_zero[p].shape == p.shape
        assert p in out_noisy and out_noisy[p].shape == p.shape
        # Noisy run should differ from no-noise run (with high probability)
        assert not torch.allclose(out_zero[p], out_noisy[p])


def test_accountant_epsilon_monotonicity():
    """Epsilon should increase (or stay same) as steps increase."""
    get_eps_poisson = make_accountant(
        "poisson",
        num_training_nodes=1000,
        batch_size=256,
        noise_multiplier=1.0,
    )
    eps_1 = get_eps_poisson(1)
    eps_10 = get_eps_poisson(10)
    eps_100 = get_eps_poisson(100)
    assert eps_1 <= eps_10 <= eps_100, "Poisson accountant should be monotonic in steps"

    get_eps_multi = make_accountant(
        "multiterm",
        num_training_nodes=1000,
        batch_size=256,
        noise_multiplier=1.0,
        max_terms_per_node=10,
    )
    eps_1m = get_eps_multi(1)
    eps_10m = get_eps_multi(10)
    assert eps_1m <= eps_10m, "Multiterm accountant should be monotonic in steps"


if __name__ == "__main__":
    test_dp_optimizer_shape_and_noise()
    print("DPOptimizer shape + noise test passed.")
    test_accountant_epsilon_monotonicity()
    print("Accountant epsilon monotonicity test passed.")
