"""Grid posterior engine (spec Algorithm H; Hoff sec 6.2; BDA3 ch. 10).

A deterministic 2-D grid over (mu, sigma) for models whose posterior has no
closed form (censored/truncated capability). The grid is uniform in mu and in
log sigma; the log posterior is evaluated on it, a log-sigma Jacobian is added so
cell masses are correct under the log spacing, and the whole thing is normalized
by log-sum-exp. Marginals come from summation, quantiles from CDF interpolation,
and draws from a seeded inverse-CDF sample (only when requested). A convergence
check doubles the resolution and adopts the finer grid until the reported
quantiles stop moving by more than 1e-3*s, capped at 801 per axis.

For the fully-observed normal model the grid reproduces the closed-form engine A
quantiles (T1.13), which is what validates the machinery before it is trusted on
the censored likelihoods that have no closed form.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

_DEFAULT_SHAPE = (201, 201)
_SHAPE_CAP = 801


def default_bounds(ybar: float, kn: float, sn2: float, s: float) -> tuple:
    """Grid bounds from the noninformative closed-form posterior: mu within
    8 posterior SDs of the mean, sigma over [s/5, 5s] (log-spaced)."""
    half = 8.0 * np.sqrt(sn2 / kn)
    return ((float(ybar - half), float(ybar + half)), (float(s / 5.0), float(5.0 * s)))


def jeffreys_logprior():
    """Noninformative prior p(mu, sigma) proportional to 1/sigma (BDA3 sec 3.2)."""
    return lambda mu, sig: -np.log(sig)


class GridPosterior:
    """Normalized posterior of (mu, sigma) on a fixed grid.

    ``loglik`` and ``logprior`` are callables of broadcast (mu, sigma) arrays
    returning log densities in (mu, sigma) space. ``bounds`` is
    ((mu_lo, mu_hi), (sigma_lo, sigma_hi)); sigma is placed on a log-spaced axis.
    """

    def __init__(self, loglik, logprior, bounds, shape=_DEFAULT_SHAPE):
        (mu_lo, mu_hi), (sig_lo, sig_hi) = bounds
        self.shape = tuple(int(v) for v in shape)
        self.bounds = ((float(mu_lo), float(mu_hi)), (float(sig_lo), float(sig_hi)))
        self.mu_axis = np.linspace(mu_lo, mu_hi, self.shape[0])
        self.sig_axis = np.exp(np.linspace(np.log(sig_lo), np.log(sig_hi), self.shape[1]))
        mu_grid, sig_grid = np.meshgrid(self.mu_axis, self.sig_axis, indexing="ij")
        # +log(sigma): Jacobian of the log-spaced sigma axis, so cell masses are
        # correct regardless of the prior.
        lp = loglik(mu_grid, sig_grid) + logprior(mu_grid, sig_grid) + np.log(sig_grid)
        # Pathological cells (a far-tail likelihood that overflowed) get zero
        # weight rather than poisoning the normalization.
        lp = np.where(np.isfinite(lp), lp, -np.inf)
        lp -= logsumexp(lp)
        self._w = np.exp(lp)
        self._p_mu = self._w.sum(axis=1)
        self._p_sig = self._w.sum(axis=0)

    def _marginal(self, param: str):
        if param in ("mu", "mean"):
            return self.mu_axis, self._p_mu
        if param in ("sigma", "sd"):
            return self.sig_axis, self._p_sig
        raise ValueError(f"unknown grid parameter {param!r}; use 'mu' or 'sigma'.")

    def quantile(self, param: str, probs) -> np.ndarray:
        """Marginal quantiles by interpolating the cell-centered CDF."""
        axis, p = self._marginal(param)
        cdf = np.cumsum(p) - 0.5 * p
        return np.interp(np.asarray(probs, dtype=float), cdf, axis)

    def mean(self, param: str) -> float:
        axis, p = self._marginal(param)
        return float((axis * p).sum())

    def sample(self, draws: int, *, seed: int) -> tuple:
        """Seeded inverse-CDF draws of (mu, sigma) with uniform in-cell jitter so
        pushed-through quantities are continuous."""
        rng = np.random.default_rng(seed)
        flat = self._w.ravel()
        flat = flat / flat.sum()
        idx = rng.choice(flat.size, size=int(draws), p=flat)
        i, j = np.unravel_index(idx, self.shape)
        dmu = self.mu_axis[1] - self.mu_axis[0]
        dlogsig = np.log(self.sig_axis[1]) - np.log(self.sig_axis[0])
        mu = self.mu_axis[i] + rng.uniform(-0.5, 0.5, size=idx.size) * dmu
        sig = self.sig_axis[j] * np.exp(rng.uniform(-0.5, 0.5, size=idx.size) * dlogsig)
        return mu, sig


def _reported_quantiles(gp: GridPosterior) -> np.ndarray:
    q = [0.025, 0.5, 0.975]
    return np.concatenate([gp.quantile("mu", q), gp.quantile("sigma", q)])


def converge_grid(build, s: float, *, shape=_DEFAULT_SHAPE, tol: float = 1e-3,
                  cap: int = _SHAPE_CAP) -> tuple:
    """Build a GridPosterior via ``build(shape)`` and refine by doubling until the
    reported (mu, sigma) quantiles move by <= tol*s, adopting the finer grid.

    Returns (grid, metadata). Metadata records the final shape, the bounds, the
    number of refinements, whether it converged, and the ``method: grid`` tag.
    """
    gp = build(shape)
    q = _reported_quantiles(gp)
    refinements = 0
    converged = False
    while max(gp.shape) < cap:
        finer = tuple(min(2 * v - 1, cap) for v in gp.shape)
        gp2 = build(finer)
        q2 = _reported_quantiles(gp2)
        move = float(np.max(np.abs(q2 - q)))
        gp, q = gp2, q2
        refinements += 1
        if move <= tol * s:
            converged = True
            break
    meta = {
        "method": "grid", "shape": gp.shape, "bounds": gp.bounds,
        "refinements": refinements, "converged": converged, "tol": tol,
    }
    return gp, meta


def fit_normal_grid(y, *, prior=None, shape=_DEFAULT_SHAPE, tol: float = 1e-3,
                    cap: int = _SHAPE_CAP) -> tuple:
    """Grid posterior for the fully-observed normal model (validation path).

    Uses the sufficient-statistic normal log-likelihood so each grid evaluation is
    O(grid), the Jeffreys prior when ``prior`` is None, and the default bounds. The
    convergence check refines from ``shape`` up to ``cap``.
    """
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    n = int(y.size)
    ybar = float(y.mean())
    s2 = float(y.var(ddof=1))
    s = float(np.sqrt(s2))
    ss = (n - 1) * s2

    def loglik(mu, sig):
        return -n * np.log(sig) - (ss + n * (ybar - mu) ** 2) / (2.0 * sig ** 2)

    logprior = jeffreys_logprior() if prior is None else _normal_logprior(prior)
    bounds = default_bounds(ybar, float(n), s2, s)
    return converge_grid(lambda sh: GridPosterior(loglik, logprior, bounds, sh),
                         s, shape=shape, tol=tol, cap=cap)


def _normal_logprior(prior):
    """Log density of a Normal-Inverse-chi2 prior in (mu, sigma) space."""
    mu0, k0, nu0, s20 = prior.mu0, prior.k0, prior.nu0, prior.s20

    def logprior(mu, sig):
        # sigma^2 ~ Inv-chi2(nu0, s20); mu | sigma ~ N(mu0, sigma^2/k0).
        var = sig ** 2
        log_sig2 = -(nu0 / 2.0 + 1.0) * np.log(var) - nu0 * s20 / (2.0 * var)
        log_mu = -0.5 * np.log(var / k0) - k0 * (mu - mu0) ** 2 / (2.0 * var)
        return log_sig2 + log_mu

    return logprior
