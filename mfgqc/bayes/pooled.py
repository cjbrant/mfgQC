"""Hierarchical (pooled) capability across measurement positions (spec Algorithm J;
BDA3 sec 5.3-5.4; Hoff sec 8.3).

A two-stage model over J positions. Stage 1: one pooled within-position variance
sigma_w^2 (Inverse-chi2 posterior, engine A). Stage 2: the BDA3 5.4 one-way normal
random-effects model on the J position means, with each position's sampling
variance plugged in as sigma_j^2 = sigma_hat_w^2 / n_j. The between-position sd tau
lives on a log-spaced grid with a uniform prior; on the grid the grand-mean and
position-mean conditional posteriors are conjugate normal and the marginal
p(tau|y) is closed form up to normalization (BDA3 eq 5.21). Draws follow the exact
factorization p(theta, mu, tau | y) = p(tau|y) p(mu|tau, y) p(theta|mu, tau, y),
then pair with independent sigma_w^2 draws for per-position Cpk and
P(min_j Cpk_j >= target).

Two stated approximations, disclosed in the report: (1) the single pooled sigma_w
assumes equal within-position variance; (2) the means-model uses the plug-in
sigma_hat_w^2 while the Cpk denominator uses fresh sigma_w^2 draws (the "two
sigmas" decoupling). The log grid carries a +log(tau) Jacobian for the uniform
prior; omitting it would silently impose p(log tau) ~ 1, which is improper here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from mfgqc._result import QCResult
from mfgqc.data import Step, _Limits
from mfgqc.errors import MissingPrerequisiteError

from ._results import _assumption_step, _now, data_digest
from .capability import _index_draws
from .guardrails import small_sample_check


def _hier_conditionals(y_means, sigma2_j, tau: float):
    """BDA3 5.17/5.20 conditional-posterior pieces at a single tau.

    Returns (mu_hat, V_mu, V_j): the grand-mean posterior mean/variance given tau
    and the per-position mean posterior variance. Exposed for the pooling-limit
    tests (tau->0 complete pooling, tau->inf no pooling)."""
    y_means = np.asarray(y_means, dtype=float)
    sigma2_j = np.asarray(sigma2_j, dtype=float)
    w = 1.0 / (sigma2_j + tau ** 2)
    V_mu = 1.0 / w.sum()
    mu_hat = float((w * y_means).sum() * V_mu)
    V_j = 1.0 / (1.0 / sigma2_j + 1.0 / tau ** 2)
    return mu_hat, float(V_mu), V_j


def _log_tau_grid(y_means, sigma2_j, tau_grid):
    """Vectorized log p(tau|y) + log(tau) Jacobian over the grid, with mu_hat(tau)
    and V_mu(tau). BDA3 eq 5.21 with a uniform prior on tau."""
    t2 = tau_grid[:, None] ** 2
    denom = sigma2_j[None, :] + t2               # (K, J)
    w = 1.0 / denom
    V_mu = 1.0 / w.sum(axis=1)                    # (K,)
    mu_hat = (w * y_means[None, :]).sum(axis=1) * V_mu
    logp = (0.5 * np.log(V_mu)
            - 0.5 * np.log(denom).sum(axis=1)
            - 0.5 * (w * (y_means[None, :] - mu_hat[:, None]) ** 2).sum(axis=1))
    logw = logp + np.log(tau_grid)               # +log(tau) Jacobian (uniform prior)
    logw = np.where(np.isfinite(logw), logw, -np.inf)
    logw -= logsumexp(logw)
    return logw, mu_hat, V_mu


@dataclass(frozen=True, eq=False, repr=False)
class HierarchicalResult:
    """Posterior draws of the one-way normal random-effects model (BDA3 5.4).

    ``eq=False`` keeps identity equality/hash: the fields are large numpy arrays for
    which the generated ``__eq__``/``__hash__`` would raise (ambiguous truth value /
    unhashable), and these results are containers, not value objects."""

    y_means: np.ndarray
    sigma2_j: np.ndarray
    theta: np.ndarray          # (draws, J) position-mean draws
    mu: np.ndarray             # (draws,) grand-mean draws
    tau: np.ndarray            # (draws,) between-position sd draws
    grid: dict

    def pooled_estimate(self) -> tuple:
        """Complete-pooling common-effect estimate (mean, sd): the precision-weighted
        grand mean and its standard error (the tau->0 analytic limit)."""
        prec = (1.0 / self.sigma2_j).sum()
        mean = float((self.y_means / self.sigma2_j).sum() / prec)
        return mean, float(math.sqrt(1.0 / prec))


def _tau_bounds(y_means, sigma2_j) -> tuple:
    s_between = float(np.std(y_means, ddof=1)) if y_means.size > 1 else 0.0
    floor = math.sqrt(float(np.min(sigma2_j)))
    hi = 10.0 * s_between if s_between > 0 else 10.0 * floor
    lo = (s_between / 1000.0) if s_between > 0 else floor / 1000.0
    return lo, hi


def hierarchical_normal(y_means, sigma, *, draws: int = 100_000, seed: int,
                        tau_points: int = 401, tau_bounds: tuple | None = None
                        ) -> HierarchicalResult:
    """Fit the BDA3 5.4 one-way normal random-effects model with KNOWN per-group
    sampling sd ``sigma`` to the group means ``y_means``.

    This is the hierarchical core used both by :func:`pooled_capability` (which
    supplies sigma_j = sigma_hat_w / sqrt(n_j)) and by the eight-schools validation
    (which supplies the published per-school standard errors directly). Sampling
    order is normative (tau via the grid, then mu, then theta) for reproducibility.
    """
    y_means = np.array(y_means, dtype=float)   # copy: the frozen result owns its data
    sigma2_j = np.asarray(sigma, dtype=float) ** 2
    if tau_bounds is None:
        tau_bounds = _tau_bounds(y_means, sigma2_j)
    lo, hi = tau_bounds
    tau_grid = np.exp(np.linspace(math.log(lo), math.log(hi), tau_points))

    logw, mu_hat, V_mu = _log_tau_grid(y_means, sigma2_j, tau_grid)
    p = np.exp(logw)
    cdf = np.cumsum(p) - 0.5 * p

    rng = np.random.default_rng(seed)
    # tau via inverse-CDF with uniform in-log-cell jitter
    u = rng.random(draws)
    idx = np.clip(np.searchsorted(cdf, u), 0, tau_points - 1)
    dlog = math.log(tau_grid[1]) - math.log(tau_grid[0])
    tau_d = tau_grid[idx] * np.exp(rng.uniform(-0.5, 0.5, draws) * dlog)
    mu_d = rng.normal(mu_hat[idx], np.sqrt(V_mu[idx]))

    t2 = tau_d[:, None] ** 2
    prec = 1.0 / sigma2_j[None, :] + 1.0 / t2
    V_j = 1.0 / prec
    theta_hat = (y_means[None, :] / sigma2_j[None, :] + mu_d[:, None] / t2) * V_j
    theta = rng.normal(theta_hat, np.sqrt(V_j))

    grid = {"method": "grid", "tau_points": int(tau_points),
            "tau_bounds": (float(lo), float(hi))}
    return HierarchicalResult(y_means=y_means, sigma2_j=sigma2_j,
                              theta=theta, mu=mu_d, tau=tau_d, grid=grid)


@dataclass(frozen=True, repr=False)
class PooledCapabilityResult(QCResult):
    """Hierarchical pooled capability across positions (immutable)."""

    n_positions: int
    position_n: tuple
    position_mean: tuple
    rejected_positions: tuple
    spec: _Limits
    target: float
    seed: int
    draws: int
    cred_level: float
    grid: dict
    nu_w: float
    sw2: float
    prob_capable: float
    prob_capable_mcse: float
    ppk_position: tuple
    prior_family: str | None = None
    _theta: object = field(default=None, repr=False, compare=False)
    _min_cpk: object = field(default=None, repr=False, compare=False)
    assumptions: list = field(default_factory=list)
    history: tuple = field(default_factory=tuple)

    def theta_interval(self, position: int, level: float | None = None) -> tuple:
        level = self.cred_level if level is None else level
        lo, hi = np.quantile(self._theta[:, position], [(1 - level) / 2, (1 + level) / 2])
        return float(lo), float(hi)

    def prob_all_capable(self) -> tuple:
        """(p_hat, mcse) for P(min_j Cpk_j >= target)."""
        return self.prob_capable, self.prob_capable_mcse

    def _title(self) -> str:
        return "Bayesian Pooled Capability (hierarchical)"

    def _summary_lines(self) -> list[str]:
        conf = round(self.cred_level * 100)
        lines = [
            f"positions = {self.n_positions}   pooled within sd = {math.sqrt(self.sw2):.4g}",
            f"P(min_j Cpk_j >= {self.target:.3g}) = {self.prob_capable:.3g} "
            f"+/- {self.prob_capable_mcse:.2g} (MC)",
        ]
        for i, (n, m, ppk) in enumerate(zip(self.position_n, self.position_mean, self.ppk_position)):
            lo, hi = self.theta_interval(i)
            ppk_txt = "n/a" if ppk is None else f"{ppk:.3g}"
            lines.append(f"  position {i}: n={n} mean={m:.4g} Cpk~{ppk_txt} "
                         f"mean {conf}% CI=({lo:.4g},{hi:.4g})")
        if self.rejected_positions:
            lines.append(f"rejected positions (n<2): {list(self.rejected_positions)}")
        lines.append("Approximation: one pooled within-sigma (equal-variance assumption) and a "
                     "plug-in sigma_hat_w in the means model vs independent sigma_w draws in Cpk "
                     "(two-sigma decoupling). tau: uniform prior on a log grid.")
        return lines

    def _render_standalone(self, fig, kind, **kwargs) -> None:
        from . import plotting
        if kind is None:
            plotting.pooled_panels(fig, self)
            return
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs) -> None:
        from . import plotting
        if kind in (None, "min_cpk", "capable"):
            plotting.pooled_min_cpk_axes(ax, self)
        elif kind in ("positions", "means"):
            plotting.pooled_positions_axes(ax, self)
        else:
            raise ValueError(f"unknown pooled view kind={kind!r}; use None, 'min_cpk', "
                             f"or 'positions'.")


def pooled_capability(groups, *, lower: float | None = None, upper: float | None = None,
                      target: float = 1.33, prior=None, seed: int, draws: int = 100_000,
                      cred_level: float = 0.95, tau_points: int = 401,
                      base_history: tuple = ()) -> PooledCapabilityResult:
    """Hierarchical pooled capability over measurement positions.

    ``groups`` is a list of per-position measurement arrays (or a 2-D array with
    one row per position). Positions with fewer than two measurements are rejected
    (recorded, not fatal). Returns per-position capability plus
    P(min_j Cpk_j >= target) from the hierarchical posterior.
    """
    raw = [np.asarray(g, dtype=float) for g in groups]
    raw = [g[~np.isnan(g)] for g in raw]
    spec = _Limits(lower=lower, upper=upper, target=None)
    if not spec.has_any():
        raise MissingPrerequisiteError(
            "pooled capability requires at least one spec limit (lower or upper).",
            analysis="bayes_pooled_capability", missing=["spec"])

    kept, means, ns, rejected = [], [], [], []
    for i, g in enumerate(raw):
        if g.size < 2:
            rejected.append(i)
            continue
        kept.append(g)
        means.append(float(g.mean()))
        ns.append(int(g.size))
    if len(kept) < 2:
        raise MissingPrerequisiteError(
            f"pooled capability needs at least 2 positions with n>=2; got {len(kept)}.",
            analysis="bayes_pooled_capability", missing=["subgroup"])

    means = np.array(means)
    ns = np.array(ns)
    ss_w = float(sum((g.size - 1) * g.var(ddof=1) for g in kept))
    nu_w = float((ns - 1).sum())
    if prior is None:
        nu_wn, sw2 = nu_w, ss_w / nu_w
    else:
        nu_wn = prior.nu0 + nu_w
        sw2 = (prior.nu0 * prior.s20 + ss_w) / nu_wn
    sigma_hat_w2 = sw2
    if sigma_hat_w2 <= 0.0:
        raise MissingPrerequisiteError(
            "pooled capability is degenerate: every position has zero within-position "
            "variance (all measurements identical). The equal-variance hierarchical "
            "model has no scale. Check the gauge resolution or supply a prior.",
            analysis="bayes_pooled_capability", missing=["variance"])
    sigma2_j = sigma_hat_w2 / ns

    fit = hierarchical_normal(means, np.sqrt(sigma2_j), draws=draws, seed=seed,
                              tau_points=tau_points)

    # Independent within-sigma draws for the Cpk denominator (two-sigma decoupling).
    rng = np.random.default_rng(seed + 1)
    sig_w = np.sqrt(nu_wn * sw2 / rng.chisquare(nu_wn, draws))

    cpk = np.empty((draws, len(kept)))
    ppk_point = []
    for j in range(len(kept)):
        _, _, _, ppk_j = _index_draws(fit.theta[:, j], sig_w, spec)
        cpk[:, j] = ppk_j
        _, _, _, ppk_pt = _index_draws(np.array([means[j]]), np.array([math.sqrt(sigma_hat_w2)]), spec)
        ppk_point.append(float(ppk_pt[0]))
    min_cpk = cpk.min(axis=1)
    p_cap = float((min_cpk >= target).mean())
    mcse = math.sqrt(p_cap * (1.0 - p_cap) / draws)

    checks = [small_sample_check(int(ns.min()))]

    step = Step(
        operation="bayes.pooled_capability",
        params={
            "prior": None if prior is None else prior.to_params(),
            "data_sha256": data_digest(np.concatenate(kept)),
            "seed": int(seed), "draws": int(draws),
            "grid": {"method": "grid", "tau_points": int(tau_points),
                     "tau_bounds": list(fit.grid["tau_bounds"])},
            "target": float(target), "cred_level": float(cred_level),
            "spec": {"lower": lower, "upper": upper},
            "rejected_positions": rejected,
        },
        n_affected=int(sum(ns)),
        timestamp=_now(),
    )
    history = tuple(base_history) + (step,) + tuple(_assumption_step(a) for a in checks)

    return PooledCapabilityResult(
        n_positions=len(kept), position_n=tuple(int(v) for v in ns),
        position_mean=tuple(float(v) for v in means), rejected_positions=tuple(rejected),
        spec=spec, target=float(target), seed=int(seed), draws=int(draws),
        cred_level=float(cred_level), grid=fit.grid, nu_w=float(nu_wn), sw2=float(sw2),
        prob_capable=p_cap, prob_capable_mcse=mcse, ppk_position=tuple(ppk_point),
        prior_family=None if prior is None else prior.to_params()["family"],
        _theta=fit.theta, _min_cpk=min_cpk, assumptions=checks, history=history,
    )
