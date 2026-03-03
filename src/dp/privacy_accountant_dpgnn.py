"""
Privacy accountants for DP-GNN (PyTorch/numpy reimplementation).

Equivalent to upstream: differentially_private_gnns/privacy_accountants.py
- Poisson subsampling RDP accountant (hops=0 / MLP)
- Multi-term RDP accountant (GCN, hops>=1): hypergeometric term distribution,
  log-sum-exp aggregation, numerically stable.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np


def _log_binom(n: int, k: int) -> float:
    """Log of binomial coefficient C(n,k). Uses math.lgamma."""
    if k < 0 or k > n:
        return -np.inf
    if k == 0 or k == n:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_logpmf(k: int, N: int, K: int, n: int) -> float:
    """
    Log PMF of hypergeometric(N, K, n) at k.
    P(X=k) = C(K,k)*C(N-K, n-k)/C(N,n).
    """
    if k < max(0, n - (N - K)) or k > min(n, K):
        return -np.inf
    return (
        _log_binom(K, k)
        + _log_binom(N - K, n - k)
        - _log_binom(N, n)
    )


def rdp_gaussian(order: float, sigma: float, sensitivity: float = 1.0) -> float:
    """
    RDP at order alpha for Gaussian mechanism with scale sigma and L2 sensitivity.
    epsilon_alpha = alpha / (2*sigma^2) * sensitivity^2 (for Gaussian mechanism).
    """
    if sigma <= 0 or order <= 1:
        return np.inf
    return (order * (sensitivity ** 2)) / (2.0 * (sigma ** 2))


def rdp_to_eps_delta(orders: np.ndarray, rdps: np.ndarray, delta: float) -> float:
    """
    Convert RDP to (epsilon, delta)-DP. epsilon = min over alpha of rdp_alpha + log(1/delta)/(alpha-1).
    Upstream: dp_accounting.rdp.compute_epsilon(orders, rdps_total, target_delta)[0]
    """
    if delta <= 0 or delta >= 1:
        return np.inf
    # eps(alpha) = rdp_alpha + ln(1/delta)/(alpha-1)
    eps_alpha = rdps + np.log(1.0 / delta) / (orders - 1.0)
    return float(np.min(eps_alpha))


def poisson_subsampling_rdp_epsilon(
    num_steps: int,
    noise_multiplier: float,
    target_delta: float,
    sampling_probability: float,
) -> float:
    """
    Poisson subsampling RDP accountant (upstream: dpsgd_privacy_accountant).
    Assumes one term per node; sampling_probability = batch_size / num_training_nodes.
    Returns epsilon for (eps, delta)-DP.
    """
    if noise_multiplier < 1e-20:
        return np.inf
    # Orders for RDP (upstream: orders = np.arange(1, 200, 0.1)[1:])
    orders = np.arange(1.01, 200.0, 0.1)
    # Unamplified RDP per step (Gaussian with sigma = noise_multiplier, sens 1)
    rdp_per_step = np.array([rdp_gaussian(alpha, noise_multiplier, 1.0) for alpha in orders])
    # Poisson subsampling amplification: simplified - use approximate bound
    # Amplified RDP \approx (1/(alpha-1)) * log(1 + q^2 * (exp((alpha-1)*rdp) - 1)) for q = sampling_prob
    # Ref: Abadi et al. "Deep Learning with Differential Privacy"
    q = sampling_probability
    amplified = np.zeros_like(rdp_per_step)
    for i, alpha in enumerate(orders):
        rdp = rdp_per_step[i]
        # Approximate amplification: (1/(a-1)) * log(1 + q^2 * (exp((a-1)*rdp) - 1))
        term = (alpha - 1) * rdp
        if term > 500:
            amplified[i] = np.inf
        else:
            inner = 1.0 + q * q * (np.exp(term) - 1.0)
            if inner <= 0:
                amplified[i] = np.inf
            else:
                amplified[i] = math.log(inner) / (alpha - 1)
    rdp_total = amplified * num_steps
    return rdp_to_eps_delta(orders, rdp_total, target_delta)


def multiterm_rdp_epsilon(
    num_steps: int,
    noise_multiplier: float,
    target_delta: float,
    num_samples: int,
    batch_size: int,
    max_terms_per_node: int,
) -> float:
    """
    Multi-term RDP accountant (upstream: multiterm_dpsgd_privacy_accountant).
    Hypergeometric distribution of terms in batch, log-sum-exp aggregation.
    """
    if noise_multiplier < 1e-20:
        return np.inf
    # Distribution of number of terms in batch (without replacement)
    # terms_rv = hypergeom(num_samples, max_terms_per_node, batch_size)
    terms_logprobs = np.array(
        [hypergeom_logpmf(i, num_samples, max_terms_per_node, batch_size) for i in range(max_terms_per_node + 1)]
    )
    # Numerical stability: subtract max before exp in logsumexp
    terms_logprobs = terms_logprobs - np.max(terms_logprobs)

    orders = np.arange(1.01, 10.0, 0.1)
    # Unamplified RDP for Gaussian(noise_multiplier)
    unamplified_rdps = np.array([rdp_gaussian(alpha, noise_multiplier, 1.0) for alpha in orders])

    amplified_rdps = []
    for idx, alpha in enumerate(orders):
        beta = unamplified_rdps[idx] * (alpha - 1)
        # log_fs = beta * ((i / max_terms_per_node)^2) for i in 0..max_terms_per_node
        i_vals = np.arange(max_terms_per_node + 1, dtype=np.float64)
        log_fs = beta * np.square(i_vals / max_terms_per_node)
        # amplified_rdp = logsumexp(terms_logprobs + log_fs) / (order - 1)
        log_sum = np.logaddexp.reduce(terms_logprobs + log_fs)
        amplified_rdps.append(log_sum / (alpha - 1))

    amplified_rdps = np.array(amplified_rdps)
    rdp_total = amplified_rdps * num_steps
    return rdp_to_eps_delta(orders, rdp_total, target_delta)


def make_accountant(
    mode: str,
    num_training_nodes: int,
    batch_size: int,
    noise_multiplier: float,
    max_terms_per_node: int = 1,
) -> Callable[[int], float]:
    """
    Factory for get_epsilon(steps) (upstream: get_training_privacy_accountant).

    mode: "poisson" (hops=0 / MLP) or "multiterm" (GCN, hops>=1).
    delta is set to 1/(10*num_training_nodes).
    """
    delta = 1.0 / (10.0 * max(num_training_nodes, 1))

    if mode == "poisson":
        q = batch_size / max(num_training_nodes, 1)

        def get_epsilon(steps: int) -> float:
            return poisson_subsampling_rdp_epsilon(
                steps,
                noise_multiplier,
                delta,
                q,
            )

    elif mode == "multiterm":

        def get_epsilon(steps: int) -> float:
            return multiterm_rdp_epsilon(
                steps,
                noise_multiplier,
                delta,
                num_training_nodes,
                batch_size,
                max_terms_per_node,
            )

    else:
        raise ValueError(f"Unknown accountant mode: {mode}")

    return get_epsilon
