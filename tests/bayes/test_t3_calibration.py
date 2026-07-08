"""T3 - simulation-based calibration for mfgqc.bayes.

Seeded fixed-N loops (hypothesis is not a dependency). T3.1 checks that credible
intervals have nominal Bayesian coverage when the true parameter is drawn from the
prior; T3.2 checks SBC rank uniformity; T3.3 checks the capability push-through
against dense-grid brute force. A wrong posterior would fail these.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from mfgqc.bayes.capability import capability_from_values
from mfgqc.bayes.conjugate import beta_update, gamma_update, update

_LEVELS = (0.5, 0.9, 0.95)
_R = 4000


def _tol(level: float, reps: int = _R) -> float:
    return 4.0 * np.sqrt(level * (1.0 - level) / reps)


def _normal_coverage(level: float, n: int = 20, seed: int = 1) -> float:
    rng = np.random.default_rng(seed)
    mu0, k0, nu0, s20 = 0.0, 2.0, 6.0, 1.0
    sigma2 = stats.invgamma(nu0 / 2, scale=nu0 * s20 / 2).rvs(_R, random_state=rng)
    mu = rng.normal(mu0, np.sqrt(sigma2 / k0))
    y = rng.normal(mu[:, None], np.sqrt(sigma2)[:, None], size=(_R, n))
    ybar, s2 = y.mean(1), y.var(1, ddof=1)
    mun, kn, nun, sn2 = update(mu0, k0, nu0, s20, n, ybar, s2)
    scale = np.sqrt(sn2 / kn)
    lo = stats.t.ppf((1 - level) / 2, nun, loc=mun, scale=scale)
    hi = stats.t.ppf((1 + level) / 2, nun, loc=mun, scale=scale)
    return float(((mu >= lo) & (mu <= hi)).mean())


def _beta_coverage(level: float, n: int = 25, seed: int = 2) -> float:
    rng = np.random.default_rng(seed)
    a, b = 2.0, 3.0
    theta = rng.beta(a, b, _R)
    y = rng.binomial(n, theta)
    ap, bp = beta_update(a, b, y, n)
    lo = stats.beta.ppf((1 - level) / 2, ap, bp)
    hi = stats.beta.ppf((1 + level) / 2, ap, bp)
    return float(((theta >= lo) & (theta <= hi)).mean())


def _gamma_coverage(level: float, x: float = 2.0, seed: int = 3) -> float:
    rng = np.random.default_rng(seed)
    a, b = 3.0, 2.0
    lam = rng.gamma(a, 1.0 / b, _R)
    y = rng.poisson(lam * x)
    ap, rate = gamma_update(a, b, y, x)
    lo = stats.gamma.ppf((1 - level) / 2, ap, scale=1.0 / rate)
    hi = stats.gamma.ppf((1 + level) / 2, ap, scale=1.0 / rate)
    return float(((lam >= lo) & (lam <= hi)).mean())


def test_t3_1_credible_interval_coverage_is_nominal():
    """T3.1: 50/90/95% credible intervals cover the prior-drawn truth at their
    nominal rates (+/- 4 binomial SE), for the normal, beta, and gamma models."""
    for cover in (_normal_coverage, _beta_coverage, _gamma_coverage):
        for level in _LEVELS:
            emp = cover(level)
            assert abs(emp - level) <= _tol(level), f"{cover.__name__} level={level}: {emp}"


# --------------------------------------------------------------------------- #
# T3.2 - simulation-based calibration (rank uniformity)
# --------------------------------------------------------------------------- #
_N_SBC, _L = 2000, 99


def _sbc_pvalue(ranks: np.ndarray, bins: int = _L + 1) -> float:
    counts = np.bincount(ranks, minlength=bins)
    exp = ranks.size / bins
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    return float(stats.chi2.sf(chi2, bins - 1))


def _normal_sbc_ranks(seed: int, n: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu0, k0, nu0, s20 = 0.0, 2.0, 6.0, 1.0
    sigma2 = stats.invgamma(nu0 / 2, scale=nu0 * s20 / 2).rvs(_N_SBC, random_state=rng)
    mu_true = rng.normal(mu0, np.sqrt(sigma2 / k0))
    y = rng.normal(mu_true[:, None], np.sqrt(sigma2)[:, None], size=(_N_SBC, n))
    mun, kn, nun, sn2 = update(mu0, k0, nu0, s20, n, y.mean(1), y.var(1, ddof=1))
    t = stats.t.rvs(nun, size=(_N_SBC, _L), random_state=rng)
    mu_s = mun[:, None] + np.sqrt(sn2 / kn)[:, None] * t
    return (mu_s < mu_true[:, None]).sum(1)


def _beta_sbc_ranks(seed: int, n: int = 25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a, b = 2.0, 3.0
    theta = rng.beta(a, b, _N_SBC)
    ap, bp = beta_update(a, b, rng.binomial(n, theta), n)
    theta_s = rng.beta(ap[:, None], bp[:, None], size=(_N_SBC, _L))
    return (theta_s < theta[:, None]).sum(1)


def _gamma_sbc_ranks(seed: int, x: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a, b = 3.0, 2.0
    lam = rng.gamma(a, 1.0 / b, _N_SBC)
    ap, rate = gamma_update(a, b, rng.poisson(lam * x), x)
    lam_s = rng.gamma(ap[:, None], 1.0 / rate, size=(_N_SBC, _L))
    return (lam_s < lam[:, None]).sum(1)


def test_t3_2_sbc_rank_uniformity():
    """T3.2: SBC ranks of the prior-drawn truth among posterior samples are uniform
    (chi-square not rejected at alpha=0.001) for the normal, beta, and gamma models.
    A miscalibrated posterior would concentrate the ranks and be rejected.
    """
    for ranks_fn, seed in ((_normal_sbc_ranks, 11), (_beta_sbc_ranks, 12), (_gamma_sbc_ranks, 13)):
        p = _sbc_pvalue(ranks_fn(seed))
        assert p > 0.001, f"{ranks_fn.__name__}: SBC chi-square p={p}"


# --------------------------------------------------------------------------- #
# T3.3 - capability push-through vs dense-grid brute force
# --------------------------------------------------------------------------- #
def _grid_ppk_mean(y, lower: float, upper: float, grid: int = 600) -> float:
    values = np.asarray(y, dtype=float)
    values = values[~np.isnan(values)]
    n = values.size
    mun, s2 = float(values.mean()), float(values.var(ddof=1))
    s = math.sqrt(s2)
    kn, nun, sn2 = float(n), float(n - 1), s2  # noninformative
    sigma = np.exp(np.linspace(math.log(s / 5), math.log(5 * s), grid))
    mu = np.linspace(mun - 8 * math.sqrt(sn2 / kn), mun + 8 * math.sqrt(sn2 / kn), grid)
    p_sigma = stats.invgamma(nun / 2, scale=nun * sn2 / 2).pdf(sigma ** 2) * 2 * sigma
    mu_g, sig_g = np.meshgrid(mu, sigma, indexing="ij")
    joint = stats.norm.pdf(mu_g, mun, np.sqrt(sig_g ** 2 / kn)) * p_sigma[None, :]
    ppk = np.minimum((upper - mu_g) / (3 * sig_g), (mu_g - lower) / (3 * sig_g))
    num = np.trapezoid(np.trapezoid(ppk * joint, mu, axis=0), sigma)
    den = np.trapezoid(np.trapezoid(joint, mu, axis=0), sigma)
    return float(num / den)


def test_t3_3_pushthrough_matches_dense_grid():
    """T3.3: the seeded MC push-through posterior mean of Ppk matches a dense-grid
    numerical integration of the same posterior to rtol 1e-3."""
    y = np.random.default_rng(33).normal(25.0, 0.5, 50)
    lower, upper = 23.0, 27.0
    r = capability_from_values(y, lower=lower, upper=upper, seed=5, draws=1_000_000)
    mc = float(r._draws["ppk"].mean())
    grid = _grid_ppk_mean(y, lower, upper)
    assert abs(mc - grid) <= 1e-3 * grid
