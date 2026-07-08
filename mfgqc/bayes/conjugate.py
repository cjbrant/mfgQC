"""Closed-form conjugate engines (deterministic core).

Engine A: Normal with unknown mean and variance, Normal-Inverse-chi2 prior
(BDA3 sec 3.3; Hoff sec 5.3).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.special import gammaln


def suffstats(y) -> tuple:
    """Sufficient statistics (n, ybar, s2) for the Normal model, NaN dropped.

    s2 is the ddof=1 sample variance (NaN when n < 2). Computed with
    ``numpy.mean``/``numpy.var(ddof=1)`` so the noninformative posterior stays
    byte-identical to the classical capability suffstats (capability.py:304-308),
    which the T4.1 t-interval match to rtol 1e-12 depends on. This is a
    deliberate deviation from the spec's suggested single Welford pass.
    """
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    n = int(y.size)
    ybar = float(y.mean()) if n else float("nan")
    s2 = float(y.var(ddof=1)) if n >= 2 else float("nan")
    return n, ybar, s2


def update(mu0: float, k0: float, nu0: float, s20: float,
           n: int, ybar: float, s2: float) -> tuple:
    """Normal-Inverse-chi2 posterior update from sufficient statistics.

    Given a N-Inv-chi2(mu0, k0, nu0, s20) prior and data summarized by
    (n, ybar, s2) with s2 the ddof=1 sample variance, return the posterior
    hyperparameters (mun, kn, nun, sn2). BDA3 sec 3.3.
    """
    kn = k0 + n
    mun = (k0 * mu0 + n * ybar) / kn
    nun = nu0 + n
    nunsn2 = nu0 * s20 + (n - 1) * s2 + (k0 * n / kn) * (ybar - mu0) ** 2
    return mun, kn, nun, nunsn2 / nun


def sigma2_marginal(nun: float, sn2: float):
    """Posterior marginal for the variance: Inv-chi2(nun, sn2), as a frozen
    scipy ``invgamma(nun/2, scale=nun*sn2/2)``. BDA3 sec 3.3."""
    return stats.invgamma(nun / 2.0, scale=nun * sn2 / 2.0)


def mu_marginal(mun: float, kn: float, nun: float, sn2: float):
    """Posterior marginal for the mean: t(df=nun, loc=mun, scale=sqrt(sn2/kn)),
    as a frozen scipy object. BDA3 sec 3.3."""
    return stats.t(df=nun, loc=mun, scale=math.sqrt(sn2 / kn))


def predictive(mun: float, kn: float, nun: float, sn2: float):
    """Posterior predictive for a single new observation:
    t(df=nun, loc=mun, scale=sqrt(sn2*(1+1/kn))), as a frozen scipy object.
    BDA3 sec 3.3."""
    return stats.t(df=nun, loc=mun, scale=math.sqrt(sn2 * (1.0 + 1.0 / kn)))


# --------------------------------------------------------------------------- #
# Engine B: Beta-Binomial (proportions) - BDA3 sec 2.4; Hoff sec 3.1
# --------------------------------------------------------------------------- #
def beta_update(a: float, b: float, y: int, n: int) -> tuple:
    """Beta posterior hyperparameters (a+y, b+n-y) from y successes in n trials."""
    return a + y, b + n - y


def beta_posterior(a: float, b: float, y: int, n: int):
    """Posterior Beta(a+y, b+n-y) as a frozen scipy object."""
    ap, bp = beta_update(a, b, y, n)
    return stats.beta(ap, bp)


def betabinom_pmf(k: int, m: int, a: float, b: float) -> float:
    """Beta-Binomial predictive pmf: P(k successes in m future trials) under a
    Beta(a, b), computed in log space with gammaln. BDA3 sec 2.4."""
    log_choose = gammaln(m + 1) - gammaln(k + 1) - gammaln(m - k + 1)
    log_pmf = (log_choose
               + gammaln(a + b) - gammaln(a) - gammaln(b)
               + gammaln(a + k) + gammaln(b + m - k) - gammaln(a + b + m))
    return float(np.exp(log_pmf))


# --------------------------------------------------------------------------- #
# Engine B: Gamma-Poisson (rates with exposure) - BDA3 sec 2.6; Hoff sec 3.2
# --------------------------------------------------------------------------- #
def gamma_update(a: float, b: float, sum_y: int, sum_x: float) -> tuple:
    """Gamma posterior hyperparameters (a+sum_y, rate=b+sum_x) for a rate with
    total count sum_y over total exposure sum_x."""
    return a + sum_y, b + sum_x


def gamma_posterior(a: float, b: float, sum_y: int, sum_x: float):
    """Posterior Gamma(a+sum_y, rate=b+sum_x) as a frozen scipy object."""
    ap, rate_p = gamma_update(a, b, sum_y, sum_x)
    return stats.gamma(ap, scale=1.0 / rate_p)


def nbinom_predictive(a_post: float, b_post: float, x_tilde: float):
    """Predictive for a future count over exposure x_tilde:
    nbinom(n=a_post, p=b_post/(b_post+x_tilde)) as a frozen scipy object.
    BDA3 sec 2.6."""
    return stats.nbinom(a_post, b_post / (b_post + x_tilde))
