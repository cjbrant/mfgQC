"""T2.6-T2.8 - Hoff oracle fixtures (values transcribed in _oracles.py)."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from mfgqc.bayes.capability import capability_from_values
from mfgqc.bayes.comparison import compare
from mfgqc.bayes.conjugate import gamma_posterior, gamma_update, mu_marginal, sigma2_marginal
from mfgqc.bayes.priors import NormalPrior

from ._oracles import BIRTH_RATES, MATH_SCORES


def test_t2_6_birth_rates_gamma_poisson_hoff_3_2_2():
    """T2.6: Hoff 3.2.2. gamma(2,1) prior; group posteriors gamma(219,112) and
    gamma(68,45); posterior means/modes/95% intervals and Pr(theta1>theta2)=0.97."""
    a, b = BIRTH_RATES["prior"]
    for key in ("group1", "group2"):
        g = BIRTH_RATES[key]
        assert gamma_update(a, b, g["sum_y"], g["n"]) == g["posterior"]
        post = gamma_posterior(a, b, g["sum_y"], g["n"])
        assert abs(float(post.mean()) - g["mean"]) <= 5e-6
        ap, rate = g["posterior"]
        assert abs((ap - 1) / rate - g["mode"]) <= 5e-6
        lo, hi = (float(v) for v in post.ppf([0.025, 0.975]))
        assert abs(lo - g["ci95"][0]) <= 5e-6
        assert abs(hi - g["ci95"][1]) <= 5e-6

    rng = np.random.default_rng(0)
    g1, g2 = BIRTH_RATES["group1"], BIRTH_RATES["group2"]
    t1 = gamma_posterior(a, b, g1["sum_y"], g1["n"]).rvs(300_000, random_state=rng)
    t2 = gamma_posterior(a, b, g2["sum_y"], g2["n"]).rvs(300_000, random_state=rng)
    assert round(float((t1 > t2).mean()), 2) == BIRTH_RATES["prob_g1_gt_g2"]


def _exact_moment(mean: float, sd: float, n: int, seed: int) -> np.ndarray:
    x = np.random.default_rng(seed).normal(0, 1, n)
    x = (x - x.mean()) / x.std(ddof=1)
    return mean + sd * x


def test_t2_7_two_group_comparison_hoff_8_1():
    """T2.7: Hoff 8.1 two-group math scores. The book states t=1.74, p=0.087 from
    full-precision data; those are not bit-reproducible from the rounded summaries
    (ybar1=50.81, ybar2=46.15, sp=10.44 give t=1.71), so we validate the Bayesian
    comparison instead: independent noninformative posteriors (pooled sd) give
    strong evidence that school 1 > school 2, consistent with the small p-value.
    """
    m = MATH_SCORES
    school2 = capability_from_values(_exact_moment(m["ybar2"], m["sp"], m["n2"], 1),
                                     lower=0.0, upper=100.0, seed=1, draws=100)
    school1 = capability_from_values(_exact_moment(m["ybar1"], m["sp"], m["n1"], 2),
                                     lower=0.0, upper=100.0, seed=2, draws=100)
    c = compare(school2, school1, seed=7, draws=1_000_000)  # P(school1 mean > school2 mean)
    assert 0.93 < c.prob_mean_gt < 0.98

    t = (m["ybar1"] - m["ybar2"]) / (m["sp"] * math.sqrt(1 / m["n1"] + 1 / m["n2"]))
    assert float(stats.t.sf(t, m["n1"] + m["n2"] - 2)) < 0.05  # one-sided p, strong evidence


def test_t2_8_from_expectations_hoff_5_5():
    """T2.8: Hoff 5.5 expectation-based elicitation. E[theta]=mu0 and E[sigma^2]=
    prior_var exactly; k0=n0, nu0=n0+3; the n0=1 case reduces to
    sigma^2 ~ inv-gamma(2, prior_var)."""
    for mu0, pv, n0 in [(10.0, 4.0, 1), (25.0, 0.25, 5), (0.0, 2.0, 3)]:
        p = NormalPrior.from_expectations(mu0=mu0, prior_var=pv, n0=n0)
        assert p.mu0 == mu0 and p.k0 == n0 and p.nu0 == n0 + 3
        assert abs(float(sigma2_marginal(p.nu0, p.s20).mean()) - pv) <= 1e-10
        assert abs(float(mu_marginal(p.mu0, p.k0, p.nu0, p.s20).mean()) - mu0) <= 1e-10

    p1 = NormalPrior.from_expectations(mu0=0.0, prior_var=3.0, n0=1)
    assert abs(p1.nu0 / 2 - 2.0) <= 1e-12          # inv-gamma shape 2
    assert abs(p1.nu0 * p1.s20 / 2 - 3.0) <= 1e-10  # inv-gamma scale = prior_var
