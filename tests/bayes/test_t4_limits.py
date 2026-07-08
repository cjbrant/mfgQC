"""T4 - frequentist-limit and cross-module consistency for bayes capability.

T4.1 pins the noninformative posterior to the classical Student-t interval; T4.3
pins the conjugate prior-weight limits; T6.5 pins the normality guardrail. T4.2 is
implemented as the location-calibration statement (see its docstring for the
documented deviation from the spec's literal Ppk wording).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, stats

from mfgqc.bayes.capability import capability_from_values
from mfgqc.bayes.conjugate import mu_marginal, sigma2_marginal
from mfgqc.bayes.priors import NormalPrior

from .fixtures_worked_example import USL, worked_example_data


def test_t4_1_noninformative_mu_interval_equals_classical_t_and_mean_is_xbar():
    """T4.1: the noninformative posterior mean of mu equals xbar, and the mu
    credible interval equals the classical Student-t interval. rtol 1e-12."""
    rng = np.random.default_rng(41)
    for _ in range(50):
        y = rng.normal(10.0, 2.0, size=int(rng.integers(5, 80)))
        r = capability_from_values(y, lower=0.0, upper=20.0, seed=1, draws=500)

        assert abs(r.mun - float(y.mean())) <= 1e-12 * (1.0 + abs(float(y.mean())))

        lo, hi = r.interval("mu", 0.95)
        n = y.size
        s = float(y.std(ddof=1))
        tlo, thi = (float(v) for v in stats.t.ppf([0.025, 0.975], n - 1,
                                                  loc=y.mean(), scale=s / np.sqrt(n)))
        assert abs(lo - tlo) <= 1e-12 * (1.0 + abs(tlo))
        assert abs(hi - thi) <= 1e-12 * (1.0 + abs(thi))


def test_t4_2a_location_calibration_exact():
    """T4.2a (erratum 001, supersedes spec v1.0 T4.2): noninformative prior,
    P(mu >= xbar | y) = 0.5 exactly by symmetry of the t marginal. rtol 1e-12.
    See tests/bayes/fixtures/erratum_001.md.
    """
    y = np.random.default_rng(42).normal(0.0, 1.0, 60)
    r = capability_from_values(y, lower=-6.0, upper=6.0, seed=3, draws=100)
    d = mu_marginal(r.mun, r.kn, r.nun, r.sn2)
    assert abs(float(d.sf(r.mun)) - 0.5) <= 1e-12
    assert abs(r.mun - float(y.mean())) <= 1e-12 * (1.0 + abs(float(y.mean())))


def test_t4_2b_scale_calibration_exact():
    """T4.2b (erratum 001): noninformative prior, nu = n-1. Since sigma^2 = nu*s^2/X
    with X ~ chi2_nu, P(sigma^2 >= s^2 | y) = P(X <= nu) = F_chi2_nu(nu) exactly
    (~0.52 at n=60; above 0.5 - the sigma side of the plug-in optimism). rtol 1e-12.
    See tests/bayes/fixtures/erratum_001.md.
    """
    y = np.random.default_rng(42).normal(0.0, 1.0, 60)
    nu = y.size - 1
    r = capability_from_values(y, lower=-6.0, upper=6.0, seed=3, draws=100)
    d = sigma2_marginal(r.nun, r.sn2)  # sn2 == s^2 on the noninformative path
    assert abs(float(d.sf(r.sn2)) - float(stats.chi2.cdf(nu, nu))) <= 1e-12


def test_t4_2c_capability_calibration_quadrature_pinned():
    """T4.2c (erratum 001): one-sided upper spec (Ppu vs USL, so the min side-switch
    cannot muddy the identity). With c = 3*ppu_hat and X ~ chi2_nu,
    P(Ppu >= ppu_hat | y) = E_X[ Phi( sqrt(n)*c*(sqrt(X/nu) - 1) ) ]. The module's
    Monte Carlo answer matches this 1-D quadrature at +/- 3*MCSE. The value is below
    0.5 and n-dependent: plug-in optimism is expected model behavior (cf. the T2.5
    worked example). Derivation in tests/bayes/fixtures/erratum_001.md.
    """
    y = worked_example_data()
    n = y.size
    nu = n - 1
    r = capability_from_values(y, upper=USL, seed=7, draws=1_000_000)
    phat = r.ppk               # one-sided -> Ppu point estimate
    c = 3.0 * phat

    mc_p, mcse = r.prob("ppk", phat)

    def integrand(x):
        return stats.norm.cdf(math.sqrt(n) * c * (math.sqrt(x / nu) - 1.0)) * stats.chi2.pdf(x, nu)

    quad, _ = integrate.quad(integrand, 0.0, np.inf, epsabs=1e-10, epsrel=1e-10, limit=400)
    assert abs(mc_p - quad) <= 3.0 * mcse
    assert mc_p < 0.5  # plug-in optimism, as documented


def test_t4_3_prior_weight_limits():
    """T4.3: k0 -> infinity pulls the posterior mean to the prior mean; k0 -> 0
    recovers the noninformative posterior mean (xbar)."""
    y = np.random.default_rng(43).normal(5.0, 1.0, 40)
    mu0 = 100.0

    big = capability_from_values(y, lower=-1e6, upper=1e6,
                                 prior=NormalPrior(mu0, 1e12, 1e12, 1.0), seed=1, draws=100)
    assert abs(big.mun - mu0) <= 1e-3

    small = capability_from_values(y, lower=-1e6, upper=1e6,
                                   prior=NormalPrior(mu0, 1e-9, 1e-9, 1.0), seed=1, draws=100)
    assert abs(small.mun - float(y.mean())) <= 1e-6


def test_t6_5_lognormal_triggers_normality_warning():
    """T6.5: strongly non-normal (lognormal) data trips the Anderson-Darling
    normality check in bayes capability (the same warning class as classical)."""
    y = np.random.default_rng(65).lognormal(0.0, 1.0, 300)
    r = capability_from_values(y, lower=0.0, upper=50.0, seed=1, draws=100)
    norm = next(a for a in r.assumptions if a.name == "normality")
    assert norm.test == "Anderson-Darling"
    assert norm.passed is False
