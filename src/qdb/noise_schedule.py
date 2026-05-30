"""Pure-functional EDM-style log-normal noise scheduler for QDB.

All functions are deterministic given an explicit ``np.random.Generator``;
no torch, no module-level state. Used by the Quantum-Diffusion-Block training
loop to (a) draw per-step sigmas, (b) carve the sigma axis into equiprobable
blocks, and (c) compute EDM loss weighting.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def log_normal_sample(
    p_mean: float,
    p_std: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``size`` samples of sigma where log(sigma) ~ N(p_mean, p_std^2)."""
    log_sigma = rng.normal(loc=p_mean, scale=p_std, size=size)
    return np.exp(log_sigma)


def equiprob_boundaries(
    n_blocks: int,
    sigma_min: float,
    sigma_max: float,
    p_mean: float,
    p_std: float,
) -> np.ndarray:
    """Return ``n_blocks + 1`` monotonically-decreasing sigma boundaries.

    Partitions the (truncated) log-normal pdf on [sigma_min, sigma_max]
    into ``n_blocks`` equal-probability slabs. Output[0] = sigma_max,
    Output[-1] = sigma_min.
    """
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")
    q_min = norm.cdf((math.log(sigma_min) - p_mean) / p_std)
    q_max = norm.cdf((math.log(sigma_max) - p_mean) / p_std)
    # Ascending quantiles 0..n_blocks
    b_idx = np.arange(n_blocks + 1, dtype=np.float64)
    q = q_min + (b_idx / n_blocks) * (q_max - q_min)
    sigmas_asc = np.exp(p_mean + p_std * norm.ppf(q))
    # Pin endpoints exactly to avoid floating-point drift
    sigmas_asc[0] = sigma_min
    sigmas_asc[-1] = sigma_max
    # Caller wants descending order
    return sigmas_asc[::-1].copy()


def edm_weight(sigma: np.ndarray, sigma_data: float) -> np.ndarray:
    """EDM loss weighting: w(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2."""
    sigma = np.asarray(sigma, dtype=np.float64)
    num = sigma ** 2 + sigma_data ** 2
    den = (sigma * sigma_data) ** 2
    return num / den


def sample_sigma_in_block(
    rng: np.random.Generator,
    sigma_low: float,
    sigma_high: float,
    p_mean: float,
    p_std: float,
    size: int = 1,
) -> np.ndarray:
    """Sample sigma from log-normal restricted to (sigma_low, sigma_high].

    Inverse-CDF sampling: draw u uniform on [Phi(z_low), Phi(z_high)],
    then sigma = exp(p_mean + p_std * Phi^{-1}(u)).
    """
    if sigma_low >= sigma_high:
        raise ValueError("sigma_low must be < sigma_high")
    z_low = (math.log(sigma_low) - p_mean) / p_std
    z_high = (math.log(sigma_high) - p_mean) / p_std
    q_low = norm.cdf(z_low)
    q_high = norm.cdf(z_high)
    u = rng.uniform(low=q_low, high=q_high, size=size)
    # Guard against numerical eq at the boundary
    u = np.clip(u, q_low + 1e-300, q_high - 1e-300) if q_high > q_low else u
    sigma = np.exp(p_mean + p_std * norm.ppf(u))
    # Clamp to the open-lower / closed-upper interval for caller safety
    sigma = np.clip(sigma, np.nextafter(sigma_low, sigma_high), sigma_high)
    return sigma
