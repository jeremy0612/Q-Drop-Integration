"""TDD tests for qdb.noise_schedule (pure-functional EDM-style log-normal scheduler)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from qdb import noise_schedule as ns


# EDM paper defaults
P_MEAN = -1.2
P_STD = 1.2
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
SIGMA_DATA = 0.5


def test_log_normal_sample_returns_positive_array_of_correct_size():
    rng = np.random.default_rng(0)
    samples = ns.log_normal_sample(P_MEAN, P_STD, size=1000, rng=rng)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (1000,)
    assert np.all(samples > 0)
    log_mean = float(np.mean(np.log(samples)))
    assert abs(log_mean - P_MEAN) < 0.1


def test_log_normal_sample_is_deterministic_with_same_seed():
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    a = ns.log_normal_sample(P_MEAN, P_STD, size=256, rng=rng_a)
    b = ns.log_normal_sample(P_MEAN, P_STD, size=256, rng=rng_b)
    np.testing.assert_array_equal(a, b)


def test_equiprob_boundaries_returns_n_plus_one_descending():
    b = ns.equiprob_boundaries(
        n_blocks=4,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        p_mean=P_MEAN,
        p_std=P_STD,
    )
    assert b.shape == (5,)
    # Strictly decreasing
    assert np.all(np.diff(b) < 0)
    assert b[0] == pytest.approx(SIGMA_MAX)
    assert b[-1] == pytest.approx(SIGMA_MIN)


def test_equiprob_boundaries_equal_probability_mass_per_block():
    B = 4
    b = ns.equiprob_boundaries(
        n_blocks=B,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        p_mean=P_MEAN,
        p_std=P_STD,
    )
    # b descends, so block b spans (b[b+1], b[b]) on the sigma axis.
    target = 1.0 / B
    for i in range(B):
        hi = b[i]
        lo = b[i + 1]
        z_hi = (math.log(hi) - P_MEAN) / P_STD
        z_lo = (math.log(lo) - P_MEAN) / P_STD
        mass = norm.cdf(z_hi) - norm.cdf(z_lo)
        # Mass is fraction of the FULL log-normal; need fraction of the
        # truncated [sigma_min, sigma_max] mass.
        z_full_hi = (math.log(SIGMA_MAX) - P_MEAN) / P_STD
        z_full_lo = (math.log(SIGMA_MIN) - P_MEAN) / P_STD
        total = norm.cdf(z_full_hi) - norm.cdf(z_full_lo)
        assert abs(mass / total - target) < 1e-6


def test_equiprob_boundaries_b_equals_one_is_endpoints():
    b = ns.equiprob_boundaries(
        n_blocks=1,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        p_mean=P_MEAN,
        p_std=P_STD,
    )
    assert b.shape == (2,)
    assert b[0] == pytest.approx(SIGMA_MAX)
    assert b[1] == pytest.approx(SIGMA_MIN)


def test_edm_weight_at_sigma_data_equals_two_over_sigma_data_squared():
    sigma = np.array([SIGMA_DATA])
    w = ns.edm_weight(sigma, SIGMA_DATA)
    expected = 2.0 / SIGMA_DATA ** 2
    assert w.shape == (1,)
    assert float(w[0]) == pytest.approx(expected, rel=1e-12)


def test_edm_weight_array_shape_preserved():
    sigma = np.linspace(0.01, 10.0, 10)
    w = ns.edm_weight(sigma, SIGMA_DATA)
    assert w.shape == (10,)
    # Sanity: positive weights
    assert np.all(w > 0)


def test_sample_sigma_in_block_stays_within_bounds():
    rng = np.random.default_rng(7)
    lo, hi = 1.0, 5.0
    samples = ns.sample_sigma_in_block(
        rng, sigma_low=lo, sigma_high=hi,
        p_mean=P_MEAN, p_std=P_STD, size=500,
    )
    assert samples.shape == (500,)
    assert np.all(samples > lo)
    assert np.all(samples <= hi)


def test_sample_sigma_in_block_concentrates_near_pmean_when_block_covers_pmean():
    rng = np.random.default_rng(123)
    center = math.exp(P_MEAN)
    lo = center / 4.0
    hi = center * 4.0
    samples = ns.sample_sigma_in_block(
        rng, sigma_low=lo, sigma_high=hi,
        p_mean=P_MEAN, p_std=P_STD, size=4000,
    )
    median = float(np.median(samples))
    # Median should be close to exp(p_mean) (within 15%)
    assert abs(median - center) / center < 0.15
