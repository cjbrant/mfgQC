"""Censored / truncated capability on the grid engine (spec Algorithm I; BDA3
sec 8.7). Gate: T1.14, T1.15, T3.6, T3.7, T6.6."""
from __future__ import annotations

import numpy as np
import pytest

from mfgqc.bayes.censored import BayesCensoredCapabilityResult, Censoring, capability_censored


# ---- T1.14 truncation reduction ------------------------------------------ #
def test_t1_14_truncation_at_infinity_equals_standard():
    """T1.14: truncation bounds at +/-inf leave the posterior identical to the
    untruncated fit (the truncation mass is exactly 1)."""
    y = np.random.default_rng(1).normal(10.0, 2.0, 60)
    base = capability_censored(y, lower=2.0, upper=18.0, seed=1, draws=1000)
    trunc = capability_censored(y, lower=2.0, upper=18.0,
                                truncation=(-np.inf, np.inf), seed=1, draws=1000)
    for param in ("mu", "sigma"):
        assert np.array_equal(base.posterior_quantile(param, [0.025, 0.5, 0.975]),
                              trunc.posterior_quantile(param, [0.025, 0.5, 0.975]))


# ---- T1.15 censoring reduction ------------------------------------------- #
def test_t1_15_all_false_flag_equals_standard():
    """T1.15: a censoring flag that is False everywhere leaves the posterior
    identical to the fully-observed fit."""
    y = np.random.default_rng(2).normal(10.0, 2.0, 60)
    base = capability_censored(y, lower=2.0, upper=18.0, seed=3, draws=1000)
    cens = capability_censored(y, lower=2.0, upper=18.0,
                               censoring=Censoring(lower=0.0, upper=100.0,
                                                   flag=np.zeros(60, dtype=bool)),
                               seed=3, draws=1000)
    assert cens.n_censored == 0
    for param in ("mu", "sigma"):
        assert np.array_equal(base.posterior_quantile(param, [0.1, 0.5, 0.9]),
                              cens.posterior_quantile(param, [0.1, 0.5, 0.9]))


# ---- T3.6 censored coverage ---------------------------------------------- #
def test_t3_6_censored_recovers_true_parameters_coverage():
    """T3.6: simulate, censor at known limits, and the 90% credible intervals for
    (mu, sigma) cover the truth near the nominal rate."""
    true_mu, true_sigma = 10.0, 2.0
    cl, cu = 7.0, 13.0
    mu_cov = sig_cov = 0
    reps = 120
    for i in range(reps):
        rng = np.random.default_rng(100 + i)
        y = rng.normal(true_mu, true_sigma, 80)
        y = np.clip(y, cl, cu)  # left/right censored at the limits
        res = capability_censored(y, lower=1.0, upper=19.0,
                                  censoring=Censoring(lower=cl, upper=cu),
                                  seed=7, draws=4000)
        mlo, mhi = res.posterior_quantile("mu", [0.05, 0.95])
        slo, shi = res.posterior_quantile("sigma", [0.05, 0.95])
        mu_cov += mlo <= true_mu <= mhi
        sig_cov += slo <= true_sigma <= shi
    assert 0.80 <= mu_cov / reps <= 1.0
    assert 0.80 <= sig_cov / reps <= 1.0


# ---- T3.7 truncation recovers pre-sort truth; naive is biased ------------ #
def test_t3_7_truncation_recovers_presort_truth_naive_biased():
    """T3.7: data sorted to keep only (lo, hi) has a deflated sample sd; the
    truncated model recovers the pre-sort sigma (coverage) while the naive
    estimate is biased low -- the feature's reason to exist."""
    true_mu, true_sigma = 10.0, 3.0
    lo, hi = 5.5, 14.5  # +/-1.5 sigma: sigma stays identifiable, naive stays biased
    covered = 0
    naive_all = []
    reps = 60
    for i in range(reps):
        rng = np.random.default_rng(500 + i)
        full = rng.normal(true_mu, true_sigma, 2000)
        kept = full[(full > lo) & (full < hi)]
        naive_all.append(kept.std(ddof=1))
        res = capability_censored(kept, lower=0.0, upper=20.0,
                                  truncation=(lo, hi), seed=5, draws=6000)
        slo, shi = res.posterior_quantile("sigma", [0.025, 0.975])
        covered += slo <= true_sigma <= shi
    assert covered / reps >= 0.8                       # truncated model recovers truth
    assert np.mean(naive_all) < true_sigma - 0.5       # naive sd is biased low


# ---- T6.6 guardrails ----------------------------------------------------- #
def test_t6_6_heavy_censoring_warns():
    """T6.6: a censoring fraction above 50% warns 'mostly tail information'."""
    rng = np.random.default_rng(9)
    y = rng.normal(10.0, 2.0, 100)
    y = np.clip(y, 9.5, 10.5)  # most points pinned to the limits
    res = capability_censored(y, lower=1.0, upper=19.0,
                              censoring=Censoring(lower=9.5, upper=10.5),
                              seed=1, draws=2000)
    chk = next(a for a in res.assumptions if a.name == "censoring_fraction")
    assert chk.passed is False and chk.recommendation is not None
    assert res.n_censored / res.n_total > 0.5


def test_t6_6_truncation_inside_data_range_raises():
    """T6.6: truncation bounds that exclude observed data are contradictory."""
    y = np.random.default_rng(4).normal(10.0, 2.0, 50)  # spans well outside [9,11]
    with pytest.raises(ValueError):
        capability_censored(y, lower=1.0, upper=19.0, truncation=(9.0, 11.0),
                            seed=1, draws=1000)


def test_t4_4_grid_noninformative_matches_closed_form_capability():
    """T4.4: with no censoring or truncation, the grid capability posterior matches
    the closed-form capability path (cross-module consistency)."""
    from mfgqc.bayes.capability import capability_from_values

    y = np.random.default_rng(8).normal(10.0, 2.0, 60)
    s = float(y.std(ddof=1))
    grid = capability_censored(y, lower=2.0, upper=18.0, seed=1, draws=1000)
    closed = capability_from_values(y, lower=2.0, upper=18.0, seed=1, draws=1000)
    for q in ("mu", "sigma"):
        g = grid.posterior_quantile(q, [0.025, 0.5, 0.975])
        c = np.array(closed.interval("mu" if q == "mu" else "sd"))
        assert abs(g[0] - c[0]) <= 1e-3 * s and abs(g[2] - c[1]) <= 1e-3 * s


def test_censored_result_type_and_report():
    y = np.random.default_rng(6).normal(10.0, 2.0, 60)
    res = capability_censored(y, lower=2.0, upper=18.0,
                              censoring=Censoring(lower=3.0, upper=17.0),
                              seed=1, draws=3000)
    assert isinstance(res, BayesCensoredCapabilityResult)
    text = res.report()
    assert "n_censored" in text.lower() or "censored" in text.lower()
    assert res.verify_provenance(res.provenance_digest()) is True
