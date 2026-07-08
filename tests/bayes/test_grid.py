"""Grid posterior engine (spec Algorithm H; Hoff sec 6.2). Gate: T1.13."""
from __future__ import annotations

import numpy as np
from scipy import stats

from mfgqc.bayes.grid import GridPosterior, default_bounds, fit_normal_grid, jeffreys_logprior


def _closed_form(y):
    n = y.size
    ybar, s2, nu = y.mean(), y.var(ddof=1), n - 1
    mu = stats.t.ppf([0.025, 0.5, 0.975], nu, loc=ybar, scale=np.sqrt(s2 / n))
    sig = np.sqrt(stats.invgamma(nu / 2, scale=nu * s2 / 2).ppf([0.025, 0.5, 0.975]))
    return mu, sig


def test_t1_13_default_grid_matches_closed_form_within_1e3_s():
    """T1.13: on an uncensored dataset the grid posterior quantiles for (mu, sigma)
    match engine A within 1e-3*s (default, convergence-refined path)."""
    y = np.random.default_rng(7).normal(10.0, 2.0, 50)
    s = float(y.std(ddof=1))
    gp, meta = fit_normal_grid(y)
    mu_cf, sig_cf = _closed_form(y)

    mu_g = gp.quantile("mu", [0.025, 0.5, 0.975])
    sig_g = gp.quantile("sigma", [0.025, 0.5, 0.975])
    assert np.max(np.abs(mu_g - mu_cf)) <= 1e-3 * s
    assert np.max(np.abs(sig_g - sig_cf)) <= 1e-3 * s
    assert meta["method"] == "grid"
    assert max(gp.shape) >= 201


def test_t1_13_refined_grid_matches_closed_form_within_1e4_s():
    """T1.13: at the refined resolution (cap 801) the match tightens to 1e-4*s."""
    y = np.random.default_rng(7).normal(10.0, 2.0, 50)
    s = float(y.std(ddof=1))
    n = y.size
    ybar, s2 = y.mean(), y.var(ddof=1)
    bounds = default_bounds(ybar, float(n), s2, s)
    ss = (n - 1) * s2

    def loglik(mu, sig):
        return -n * np.log(sig) - (ss + n * (ybar - mu) ** 2) / (2.0 * sig ** 2)

    gp = GridPosterior(loglik, jeffreys_logprior(), bounds, shape=(801, 801))
    mu_cf, sig_cf = _closed_form(y)
    assert np.max(np.abs(gp.quantile("mu", [0.025, 0.5, 0.975]) - mu_cf)) <= 1e-4 * s
    assert np.max(np.abs(gp.quantile("sigma", [0.025, 0.5, 0.975]) - sig_cf)) <= 1e-4 * s


def test_grid_sampling_reproduces_marginals():
    """Seeded inverse-CDF draws reproduce the grid marginal quantiles."""
    y = np.random.default_rng(3).normal(0.0, 1.0, 40)
    gp, _ = fit_normal_grid(y)
    mu, sig = gp.sample(200_000, seed=1)
    for p in (0.1, 0.5, 0.9):
        assert abs(np.quantile(mu, p) - gp.quantile("mu", [p])[0]) <= 0.02
        assert abs(np.quantile(sig, p) - gp.quantile("sigma", [p])[0]) <= 0.02
    # determinism: same seed -> same draws
    a, _ = gp.sample(500, seed=5)
    b, _ = gp.sample(500, seed=5)
    assert np.array_equal(a, b)
