"""Hierarchical pooled capability (spec Algorithm J; BDA3 sec 5.3-5.4; Hoff sec 8.3).
Gate: T1.16 (pooling limits), T2.9 (eight schools), T3.8 (coverage)."""
from __future__ import annotations

import numpy as np
import pytest

from mfgqc.bayes.pooled import (
    PooledCapabilityResult,
    _hier_conditionals,
    hierarchical_normal,
    pooled_capability,
)

from ._oracles import EIGHT_SCHOOLS


# ---- T1.16 pooling limits (conditional-formula level, no MC) -------------- #
def test_t1_16_complete_pooling_limit_tau_to_zero():
    """T1.16: as tau->0 the hierarchical model collapses to complete pooling: every
    position mean shrinks to the precision-weighted grand mean, V_j->0."""
    y = np.array([2.0, 5.0, 9.0, 4.0])
    sig2 = np.array([1.0, 2.0, 0.5, 1.5])
    mu_hat, V_mu, V_j = _hier_conditionals(y, sig2, tau=1e-8)

    pooled_mean = (y / sig2).sum() / (1.0 / sig2).sum()
    pooled_var = 1.0 / (1.0 / sig2).sum()
    assert abs(mu_hat - pooled_mean) <= 1e-6 * abs(pooled_mean)
    assert abs(V_mu - pooled_var) <= 1e-6 * pooled_var
    assert np.all(V_j <= 1e-10)  # position means collapse onto the grand mean


def test_t1_16_no_pooling_limit_tau_to_infinity():
    """T1.16: as tau->inf there is no shrinkage: each position mean posterior
    variance approaches its own sampling variance sigma_j^2."""
    y = np.array([2.0, 5.0, 9.0, 4.0])
    sig2 = np.array([1.0, 2.0, 0.5, 1.5])
    _, _, V_j = _hier_conditionals(y, sig2, tau=1e8)
    assert np.allclose(V_j, sig2, rtol=1e-6)


# ---- T2.9 eight schools (known sigma_j) ---------------------------------- #
def test_t2_9_eight_schools_reproduces_bda3():
    """T2.9: BDA3 sec 5.5 eight schools via the known-sigma_j hierarchical path.
    Data is exact (Table 5.2); Table 5.3 posterior medians match closely; the
    complete-pooling estimate is exact; text probabilities land in the book's
    (200-draw) Monte Carlo bands.

    Documented delta: the production pooled_capability ESTIMATES a single within
    sigma, whereas eight schools uses per-school KNOWN sigma_j fed straight through
    hierarchical_normal - so this validates the hierarchical-normal core.
    """
    y = np.array([d[1] for d in EIGHT_SCHOOLS["data"]])
    sig = np.array([d[2] for d in EIGHT_SCHOOLS["data"]])
    fit = hierarchical_normal(y, sig, draws=200_000, seed=0)

    # complete-pooling estimate is exact analytic
    pm, pse = fit.pooled_estimate()
    assert abs(pm - EIGHT_SCHOOLS["pooled_mean"]) <= 0.05
    assert abs(pse - EIGHT_SCHOOLS["pooled_se"]) <= 0.05

    # Table 5.3 posterior medians (200-draw reference): match within ~1.5 points
    names = "ABCDEFGH"
    for j, nm in enumerate(names):
        med = float(np.median(fit.theta[:, j]))
        assert abs(med - EIGHT_SCHOOLS["posterior_median"][nm]) <= 1.5

    # text probabilities (book values carry 200-draw MC noise)
    assert 0.60 <= float((fit.theta[:, 0] > fit.theta[:, 2]).mean()) <= 0.80  # Pr(A>C)~0.705
    assert 0.03 <= float((fit.theta.max(1) > 28.4).mean()) <= 0.18            # Pr(max>28.4)~0.11
    assert float((fit.tau > 25).mean()) <= 0.05                              # Pr(tau>25)~0


def test_t2_9_deterministic_given_seed():
    y = np.array([d[1] for d in EIGHT_SCHOOLS["data"]])
    sig = np.array([d[2] for d in EIGHT_SCHOOLS["data"]])
    a = hierarchical_normal(y, sig, draws=5000, seed=3)
    b = hierarchical_normal(y, sig, draws=5000, seed=3)
    assert np.array_equal(a.theta, b.theta)


# ---- T3.8 coverage under a hierarchical generator ------------------------ #
@pytest.mark.parametrize("tau_true", [0.3, 1.5, 6.0])
def test_t3_8_position_mean_coverage(tau_true):
    """T3.8: data generated from a true hierarchical process; the 90% credible
    intervals for each position mean cover the truth near the nominal rate across
    small/moderate/large between-position spread."""
    J, nj, sigma_true, mu_true = 5, 12, 2.0, 25.0
    covered = total = 0
    for i in range(150):
        rng = np.random.default_rng(3000 + i)
        theta = rng.normal(mu_true, tau_true, J)
        groups = [rng.normal(theta[j], sigma_true, nj) for j in range(J)]
        res = pooled_capability(groups, lower=15.0, upper=35.0, seed=1, draws=3000,
                                tau_points=201)
        for j in range(J):
            lo, hi = res.theta_interval(j, level=0.90)
            covered += lo <= theta[j] <= hi
            total += 1
    assert 0.83 <= covered / total <= 0.97


def test_pooled_capability_min_cpk_probability_and_report():
    rng = np.random.default_rng(7)
    groups = [rng.normal(25.0, 0.4, 15) for _ in range(4)]
    res = pooled_capability(groups, lower=23.0, upper=27.0, target=1.33, seed=1, draws=5000)
    assert isinstance(res, PooledCapabilityResult)
    p, mcse = res.prob_all_capable()
    assert 0.0 <= p <= 1.0 and mcse >= 0.0
    assert res.verify_provenance(res.provenance_digest()) is True
    assert "pool" in res.report().lower()


def test_pooled_rejects_small_positions():
    rng = np.random.default_rng(1)
    groups = [rng.normal(25.0, 0.4, 15), rng.normal(25.0, 0.4, 1), rng.normal(25.0, 0.4, 15)]
    res = pooled_capability(groups, lower=23.0, upper=27.0, seed=1, draws=2000)
    assert 1 in res.rejected_positions
    assert res.n_positions == 2


# ---- review regressions -------------------------------------------------- #
def test_zero_within_variance_raises_not_silent_nan():
    """Review finding 1/3 (critical/major): all-identical measurements per position
    give zero pooled within-variance; must raise a clear error, never silently
    return prob_capable=0.0 with NaN draws / a math-domain crash."""
    groups = [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]
    with pytest.raises(Exception) as exc:
        pooled_capability(groups, lower=0.0, upper=10.0, target=1.33, seed=1, draws=2000)
    assert "variance" in str(exc.value).lower()


def test_hierarchical_result_equality_and_hash_do_not_crash():
    """Review finding 4 (major): comparing/hashing a frozen HierarchicalResult with
    ndarray fields must not raise 'ambiguous truth value'. Two separate instances
    exercise the real path (a == a short-circuits on identity)."""
    y = np.array([1.0, 2.0, 3.0])
    sig = np.array([1.0, 1.0, 1.0])
    a = hierarchical_normal(y, sig, draws=1000, seed=1)
    b = hierarchical_normal(y, sig, draws=1000, seed=1)
    assert (a == b) in (True, False)  # must not raise
    assert isinstance(hash(a), int)   # must not raise (unhashable ndarray/dict)


def test_hierarchical_result_does_not_alias_caller_array():
    """Review finding 5 (minor): the frozen result must own its data, not alias the
    caller's mutable array."""
    y = np.array([1.0, 2.0, 3.0])
    sig = np.array([1.0, 1.0, 1.0])
    fit = hierarchical_normal(y, sig, draws=500, seed=1)
    before = fit.y_means.copy()
    y[0] = 999.0
    assert np.array_equal(fit.y_means, before)
