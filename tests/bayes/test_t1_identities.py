"""T1 - analytic identities for mfgqc.bayes (oracle-free, self-checking math).

Sources cited per test. These pin the conjugate engine against algebraically
independent routes, never against a prior mfgqc run.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, stats

from mfgqc.bayes.conjugate import (
    beta_posterior,
    beta_update,
    betabinom_pmf,
    gamma_posterior,
    gamma_update,
    mu_marginal,
    nbinom_predictive,
    predictive,
    sigma2_marginal,
    suffstats,
    update,
)


def test_t1_1_normal_update_matches_direct_sum_of_squares_decomposition():
    """T1.1: engine A update() equals the independent SS-about-combined-mean route.

    Oracle: BDA3 (Gelman et al., 3rd ed.) sec 3.3, Normal-Inverse-chi2 posterior.
    The between-source term k0*n/kn*(ybar-mu0)^2 is algebraically identical to
    k0*mu0^2 + n*ybar^2 - kn*mun^2 (sum of squares decomposed about the combined
    mean). Computing it the second way is an independent check that update()
    implements the conjugate posterior rather than restating its own formula.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        mu0 = float(rng.normal(0, 5))
        k0 = float(rng.uniform(0.1, 20))
        nu0 = float(rng.uniform(1, 30))
        s20 = float(rng.uniform(0.01, 4))
        n = int(rng.integers(2, 50))
        ybar = float(rng.normal(0, 5))
        s2 = float(rng.uniform(0.01, 4))

        mun, kn, nun, sn2 = update(mu0, k0, nu0, s20, n, ybar, s2)

        kn_d = k0 + n
        mun_d = (k0 * mu0 + n * ybar) / kn_d
        nun_d = nu0 + n
        between = k0 * mu0**2 + n * ybar**2 - kn_d * mun_d**2
        sn2_d = (nu0 * s20 + (n - 1) * s2 + between) / nun_d

        assert kn == kn_d
        assert nun == nun_d
        assert mun == mun_d
        assert abs(sn2 - sn2_d) <= 1e-9 * (1.0 + abs(sn2_d))


def test_t1_2_posterior_is_permutation_invariant():
    """T1.2: the posterior depends on the data only through (n, ybar, s2), so any
    permutation of the observations yields the same posterior. Sufficiency.
    'Exact' per spec conventions = within rtol 1e-12.
    """
    rng = np.random.default_rng(1)
    prior = (0.5, 3.0, 4.0, 0.2)  # mu0, k0, nu0, s20
    for _ in range(200):
        y = rng.normal(2.0, 1.3, size=int(rng.integers(2, 60)))
        p1 = update(*prior, *suffstats(y))
        p2 = update(*prior, *suffstats(rng.permutation(y)))
        assert np.allclose(p1, p2, rtol=1e-12, atol=0.0)


def test_t1_3_sequential_update_equals_batch():
    """T1.3: folding data as two sub-batches (each posterior becoming the prior
    for the next) equals a single batch update on the pooled data. Coherence of
    the conjugate chain (also exercises shortrun stage chaining). rtol 1e-12.
    """
    rng = np.random.default_rng(2)
    for _ in range(200):
        y = rng.normal(-1.0, 2.0, size=int(rng.integers(4, 80)))
        cut = int(rng.integers(2, y.size - 1))
        a, b = y[:cut], y[cut:]
        prior = (0.0, 2.0, 3.0, 1.0)
        batch = update(*prior, *suffstats(y))
        mun1, kn1, nun1, sn2_1 = update(*prior, *suffstats(a))
        seq = update(mun1, kn1, nun1, sn2_1, *suffstats(b))
        assert np.allclose(batch, seq, rtol=1e-12, atol=0.0)


def test_t1_6_marginals_match_independent_scipy_reparameterizations():
    """T1.6: posterior marginals agree with independent reparameterizations.

    sigma2 ~ Inv-chi2 is equivalent to nun*sn2/sigma2 ~ chi2_nun, so the invgamma
    CDF must equal the chi2 survival of the scaled variate (invgamma vs chi2 -
    genuinely independent). mu is Student-t location-scale about mun. rtol 1e-10.
    BDA3 sec 3.3.
    """
    rng = np.random.default_rng(6)
    for _ in range(200):
        mun = float(rng.normal(0, 3))
        kn = float(rng.uniform(1, 40))
        nun = float(rng.uniform(3, 60))
        sn2 = float(rng.uniform(0.05, 5))

        s2d = sigma2_marginal(nun, sn2)
        for q in (0.1, 0.5, 0.9):
            x = float(s2d.ppf(q))
            assert abs(float(s2d.cdf(x)) - float(stats.chi2.sf(nun * sn2 / x, nun))) <= 1e-10

        md = mu_marginal(mun, kn, nun, sn2)
        for q in (0.025, 0.5, 0.975):
            expect = mun + math.sqrt(sn2 / kn) * float(stats.t.ppf(q, nun))
            assert abs(float(md.ppf(q)) - expect) <= 1e-10 * (1.0 + abs(expect))


def test_t1_7_predictive_equals_numerically_integrated_mixture():
    """T1.7: the single-observation posterior predictive t-distribution equals
    the scale-mixture integral int N(ynew|mun, sqrt(sigma2*(1+1/kn))) p(sigma2)
    dsigma2, integrated numerically. Independent (quadrature vs closed-form t).
    rtol 1e-8. BDA3 sec 3.3.
    """
    rng = np.random.default_rng(7)
    for _ in range(30):
        mun = float(rng.normal(0, 2))
        kn = float(rng.uniform(2, 30))
        nun = float(rng.uniform(4, 40))
        sn2 = float(rng.uniform(0.1, 3))

        pred = predictive(mun, kn, nun, sn2)
        s2d = sigma2_marginal(nun, sn2)
        for ynew in (mun - 1.0, mun + 0.3, mun + 2.0):
            def integrand(s2, ynew=ynew, kn=kn, mun=mun):
                return stats.norm.pdf(ynew, mun, math.sqrt(s2 * (1.0 + 1.0 / kn))) * float(s2d.pdf(s2))

            val, _ = integrate.quad(integrand, 0.0, np.inf, epsabs=1e-13, epsrel=1e-13, limit=400)
            assert abs(float(pred.pdf(ynew)) - val) <= 1e-8 * (1.0 + abs(val))


def test_t1_4_beta_update_and_posterior_mean_identity():
    """T1.4: Beta posterior is Beta(a+y, b+n-y); its mean is the precision-weighted
    blend of the prior mean a/(a+b) and the data proportion y/n. Exact (rtol 1e-12).
    BDA3 sec 2.4; Hoff sec 3.1.
    """
    rng = np.random.default_rng(4)
    for _ in range(200):
        a = float(rng.uniform(0.5, 5))
        b = float(rng.uniform(0.5, 5))
        n = int(rng.integers(1, 200))
        y = int(rng.integers(0, n + 1))

        ap, bp = beta_update(a, b, y, n)
        assert (ap, bp) == (a + y, b + n - y)

        post = beta_posterior(a, b, y, n)
        post_mean = (a + y) / (a + b + n)
        assert abs(float(post.mean()) - post_mean) <= 1e-12 * (1.0 + abs(post_mean))

        w = (a + b) / (a + b + n)
        blend = w * (a / (a + b)) + (1.0 - w) * (y / n)
        assert abs(post_mean - blend) <= 1e-12 * (1.0 + abs(blend))


def test_t1_5_gamma_update_with_exposure_and_posterior_mean():
    """T1.5: Gamma-Poisson posterior is Gamma(a+sum_y, rate=b+sum_x) with posterior
    mean (a+sum_y)/(b+sum_x). Exact (rtol 1e-12). BDA3 sec 2.6; Hoff sec 3.2.
    """
    rng = np.random.default_rng(5)
    for _ in range(200):
        a = float(rng.uniform(0.5, 5))
        b = float(rng.uniform(0.1, 3))
        k = int(rng.integers(1, 40))
        counts = rng.integers(0, 20, size=k)
        exposure = rng.uniform(0.1, 5, size=k)
        sum_y = int(counts.sum())
        sum_x = float(exposure.sum())

        ap, rate_p = gamma_update(a, b, sum_y, sum_x)
        assert (ap, rate_p) == (a + sum_y, b + sum_x)

        post = gamma_posterior(a, b, sum_y, sum_x)
        mean = (a + sum_y) / (b + sum_x)
        assert abs(float(post.mean()) - mean) <= 1e-12 * (1.0 + abs(mean))


def test_t1_7_betabinomial_predictive_equals_integrated_mixture():
    """T1.7 (Beta-Binomial form): the predictive pmf for k successes in m future
    trials equals int Binomial(k|m,theta) Beta(theta|A,B) dtheta. Independent
    (quadrature vs the gammaln closed form). rtol 1e-8. BDA3 sec 2.4.
    """
    rng = np.random.default_rng(17)
    for _ in range(20):
        A = float(rng.uniform(0.5, 8))
        B = float(rng.uniform(0.5, 8))
        m = int(rng.integers(1, 12))
        for k in range(m + 1):
            def integrand(theta, k=k, m=m, A=A, B=B):
                return float(stats.binom.pmf(k, m, theta)) * float(stats.beta.pdf(theta, A, B))

            val, _ = integrate.quad(integrand, 0.0, 1.0, epsabs=1e-13, epsrel=1e-13, limit=200)
            assert abs(betabinom_pmf(k, m, A, B) - val) <= 1e-8 * (1.0 + abs(val))


def test_t1_7_negbinomial_predictive_equals_integrated_mixture():
    """T1.7 (negative-binomial form): the predictive pmf for a future count over
    exposure x_tilde equals int Poisson(k|lambda*x_tilde) Gamma(lambda|A,rate=B)
    dlambda. Independent (quadrature vs scipy nbinom). rtol 1e-8. BDA3 sec 2.6.
    """
    rng = np.random.default_rng(18)
    for _ in range(20):
        A = float(rng.uniform(1.0, 10))
        B = float(rng.uniform(0.5, 4))
        x_tilde = float(rng.uniform(0.5, 4))
        pred = nbinom_predictive(A, B, x_tilde)
        for k in (0, 1, 2, 5):
            def integrand(lam, k=k, x_tilde=x_tilde, A=A, B=B):
                return float(stats.poisson.pmf(k, lam * x_tilde)) * float(stats.gamma.pdf(lam, A, scale=1.0 / B))

            val, _ = integrate.quad(integrand, 0.0, np.inf, epsabs=1e-13, epsrel=1e-13, limit=400)
            assert abs(float(pred.pmf(k)) - val) <= 1e-8 * (1.0 + abs(val))
