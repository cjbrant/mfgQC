"""Censored / truncated capability (spec Algorithm I; BDA3 sec 8.7).

When measurements pile up at a gauge limit (censoring) or when parts outside a
window were removed before measuring (truncation), the fully-observed normal
likelihood is wrong and the classical sample sd is biased. This module builds the
correct log-likelihood -- exact normal density for observed values, a tail
probability for each censored value, and a truncation-mass normalization -- and
solves it on the grid engine. The capability indices are computed from the
posterior (mu, sigma), so they describe the *process* before sorting, which is
the whole point: the truncated fit recovers the pre-sort spread that the naive
estimate understates (T3.7).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import log_ndtr

from mfgqc._result import QCResult
from mfgqc.assumptions import AssumptionCheck
from mfgqc.assumptions import reliability as _reliability
from mfgqc.data import Step, _Limits
from mfgqc.errors import MissingPrerequisiteError

from ._results import _assumption_step, _now, data_digest
from .capability import _index_draws, _ppm_draws
from .grid import GridPosterior, converge_grid, jeffreys_logprior, _normal_logprior


@dataclass(frozen=True)
class Censoring:
    """Censoring specification. ``lower``/``upper`` are the limits at which values
    pile up (left/right censoring). ``flag`` optionally marks censored rows
    explicitly (a boolean array); when given, only flagged rows are censored."""

    lower: float | None = None
    upper: float | None = None
    flag: object = None

    def to_params(self) -> dict:
        return {"lower": self.lower, "upper": self.upper,
                "flag": None if self.flag is None else [bool(v) for v in self.flag]}


def _classify(y: np.ndarray, censoring: Censoring | None) -> tuple:
    """Split ``y`` into (observed values, n_left, n_right) given a censoring spec."""
    if censoring is None:
        return y, 0, 0
    lo, hi, flag = censoring.lower, censoring.upper, censoring.flag
    if flag is not None:
        flag = np.asarray(flag, dtype=bool)
        if flag.shape != y.shape:
            raise ValueError("censoring flag must match the data length.")
        censored = flag
    else:
        censored = np.zeros(y.shape, dtype=bool)
        if lo is not None:
            censored |= y <= lo
        if hi is not None:
            censored |= y >= hi
    left = censored & (lo is not None) & (y <= (lo if lo is not None else -np.inf))
    right = censored & (hi is not None) & (y >= (hi if hi is not None else np.inf))
    # a flagged row that matches neither side is attributed to the nearer limit
    unresolved = censored & ~left & ~right
    if unresolved.any():
        if lo is not None and hi is not None:
            mid = 0.5 * (lo + hi)
            left = left | (unresolved & (y <= mid))
            right = right | (unresolved & (y > mid))
        elif lo is not None:
            left = left | unresolved
        else:
            right = right | unresolved
    observed = y[~censored]
    return observed, int(left.sum()), int(right.sum())


def _log_mass(a, b):
    """log(Phi(b) - Phi(a)) for a <= b, stable in the tails. Positive intervals are
    mirrored to the left tail so the dominant term keeps precision."""
    mirror = a > 0
    lo = np.where(mirror, -b, a)
    hi = np.where(mirror, -a, b)
    with np.errstate(divide="ignore"):
        return log_ndtr(hi) + np.log1p(-np.exp(log_ndtr(lo) - log_ndtr(hi)))


def _censored_loglik(n_obs: int, sy: float, syy: float, n_left: int, n_right: int,
                     cens_lo, cens_hi, truncation):
    """O(grid) log-likelihood closure for the censored / truncated normal model."""
    lo_t, hi_t = (truncation if truncation is not None else (None, None))
    n_retained = n_obs + n_left + n_right

    def loglik(mu, sig):
        ll = np.zeros_like(mu)
        if n_obs:
            ss = syy - 2.0 * mu * sy + n_obs * mu ** 2
            ll = ll - n_obs * np.log(sig) - ss / (2.0 * sig ** 2)
        if n_left:
            ll = ll + n_left * log_ndtr((cens_lo - mu) / sig)
        if n_right:
            ll = ll + n_right * log_ndtr((mu - cens_hi) / sig)
        if truncation is not None:
            ll = ll - n_retained * _log_mass((lo_t - mu) / sig, (hi_t - mu) / sig)
        return ll

    return loglik


@dataclass(frozen=True, repr=False)
class BayesCensoredCapabilityResult(QCResult):
    """Grid posterior capability for censored / truncated data (immutable). All
    quantities derive from the posterior (mu, sigma), so they describe the process
    before sorting."""

    n_total: int
    n_obs: int
    n_censored: int
    n_left: int
    n_right: int
    spec: _Limits
    truncation: tuple | None
    censoring: dict | None
    seed: int
    draws: int
    cred_level: float
    grid: dict
    pp: float | None
    ppu: float | None
    ppl: float | None
    ppk: float | None
    ppm: float
    prior_family: str | None = None
    _grid: object = field(default=None, repr=False, compare=False)
    _draws: dict = field(default_factory=dict, repr=False)
    assumptions: list = field(default_factory=list)
    history: tuple = field(default_factory=tuple)

    def posterior_quantile(self, param: str, probs) -> np.ndarray:
        """Grid marginal quantiles for 'mu' or 'sigma' (deterministic, draw-free)."""
        return self._grid.quantile(param, probs)

    def interval(self, quantity: str, level: float | None = None) -> tuple:
        level = self.cred_level if level is None else level
        lo_p, hi_p = (1.0 - level) / 2.0, (1.0 + level) / 2.0
        if quantity in ("mu", "mean", "sigma", "sd"):
            param = "mu" if quantity in ("mu", "mean") else "sigma"
            lo, hi = self.posterior_quantile(param, [lo_p, hi_p])
            return float(lo), float(hi)
        lo, hi = np.quantile(self._draws[quantity], [lo_p, hi_p])
        return float(lo), float(hi)

    def quantiles(self, quantity: str, probs) -> list:
        return [float(v) for v in np.quantile(self._draws[quantity], probs)]

    def prob(self, quantity: str, threshold: float, direction: str = ">=") -> tuple:
        arr = self._draws[quantity]
        p = float((arr >= threshold).mean()) if direction == ">=" else float((arr <= threshold).mean())
        return p, math.sqrt(p * (1.0 - p) / arr.size)

    def _title(self) -> str:
        return "Bayesian Capability (censored/truncated, grid)"

    def _summary_lines(self) -> list[str]:
        conf = round(self.cred_level * 100)

        def fmt(v):
            return "  n/a" if v is None else f"{v:.4g}"

        mu_lo, mu_hi = self.interval("mu")
        ppk_lo, ppk_hi = self.interval("ppk")
        lines = [
            f"n_total = {self.n_total}   n_observed = {self.n_obs}   "
            f"n_censored = {self.n_censored} (left {self.n_left}, right {self.n_right})",
            f"mu {conf}% credible interval = ({mu_lo:.5g}, {mu_hi:.5g})",
            f"Pp  = {fmt(self.pp)}    Ppk (posterior median) = {fmt(self.ppk)}",
            f"Ppk {conf}% credible interval = ({ppk_lo:.3g}, {ppk_hi:.3g})",
            f"ppm (posterior median) = {self.ppm:.0f}",
            f"grid: shape {self.grid['shape']}, refinements {self.grid['refinements']}, "
            f"converged {self.grid['converged']}",
        ]
        if self.truncation is not None:
            lines.append(f"truncation window = ({self.truncation[0]:.4g}, {self.truncation[1]:.4g})")
        if self.censoring is not None:
            lines.append(f"censoring limits = (lower {self.censoring.get('lower')}, "
                         f"upper {self.censoring.get('upper')})")
        if self.prior_family is not None:
            lines.append(f"prior = {self.prior_family}")
        return lines

    def _render_standalone(self, fig, kind, **kwargs) -> None:
        from . import plotting
        if kind is None:
            plotting.capability_panels(fig, self)
            return
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs) -> None:
        from . import plotting
        if kind in (None, "ppk", "capability"):
            plotting.capability_ppk_axes(ax, self)
        elif kind in ("mu", "mean"):
            plotting.capability_mu_axes(ax, self)
        elif kind == "ppm":
            plotting.capability_ppm_axes(ax, self)
        elif kind in ("predictive", "histogram"):
            plotting.capability_predictive_axes(ax, self)
        else:
            raise ValueError(f"unknown censored view kind={kind!r}; use None, 'ppk', "
                             f"'mu', 'ppm', or 'predictive'.")


def _censoring_fraction_check(n_censored: int, n_total: int) -> AssumptionCheck:
    frac = n_censored / n_total if n_total else 0.0
    passed = frac <= 0.5
    return AssumptionCheck(
        name="censoring_fraction", test="censored fraction <= 0.5",
        statistic=float(frac), p_value=None, passed=bool(passed),
        magnitude=float(frac), magnitude_label="censored fraction",
        reliability=_reliability(n_total - n_censored), n=int(n_total),
        recommendation=None if passed else (
            f"{frac:.0%} of observations are censored; the fit is mostly tail "
            f"information and the posterior is prior/limit sensitive. Treat the "
            f"indices as indicative and collect uncensored measurements."),
    )


def capability_censored(y, *, lower: float | None = None, upper: float | None = None,
                        target: float | None = None, truncation: tuple | None = None,
                        censoring: Censoring | None = None, prior=None, seed: int,
                        draws: int = 100_000, cred_level: float = 0.95,
                        shape: tuple = (201, 201),
                        base_history: tuple = ()) -> BayesCensoredCapabilityResult:
    """Bayesian capability for censored and/or truncated measurements.

    ``lower``/``upper`` are the spec limits (LSL/USL). ``censoring`` marks values
    piled up at a gauge limit; ``truncation=(lo, hi)`` says the data were
    conditioned on falling inside that window. The model is solved on the grid
    engine and the capability indices come from the posterior (mu, sigma), so they
    describe the process before censoring/sorting.
    """
    values = np.asarray(y, dtype=float)
    values = values[~np.isnan(values)]
    n_total = int(values.size)

    spec = _Limits(lower=lower, upper=upper, target=target)
    if not spec.has_any():
        raise MissingPrerequisiteError(
            "censored capability requires at least one spec limit (lower or upper).",
            analysis="bayes_capability_censored", missing=["spec"])

    if truncation is not None:
        lo_t, hi_t = truncation
        if np.isfinite(lo_t) and np.any(values < lo_t) or np.isfinite(hi_t) and np.any(values > hi_t):
            raise ValueError(
                f"truncation window ({lo_t}, {hi_t}) excludes observed data; the "
                f"retained data must lie inside the truncation bounds.")

    observed, n_left, n_right = _classify(values, censoring)
    n_obs = int(observed.size)
    n_censored = n_left + n_right
    if n_obs < 2:
        raise ValueError(
            f"need at least 2 uncensored observations to anchor the grid scale; "
            f"got n_observed={n_obs}.")

    sy = float(observed.sum())
    syy = float((observed ** 2).sum())
    ybar = sy / n_obs
    s2 = float(observed.var(ddof=1))
    s = math.sqrt(s2)

    cens_lo = censoring.lower if censoring is not None else None
    cens_hi = censoring.upper if censoring is not None else None
    loglik = _censored_loglik(n_obs, sy, syy, n_left, n_right, cens_lo, cens_hi, truncation)
    logprior = jeffreys_logprior() if prior is None else _normal_logprior(prior)
    prior_family = None if prior is None else prior.to_params()["family"]

    # Two-stage bounds: censoring/truncation deflate the observed spread, so a
    # generous coarse grid first locates the posterior, then the fine grid brackets
    # it tightly (fast convergence and fine resolution near the mode). Identical
    # log-likelihoods produce identical stages, so the T1.14/T1.15 reductions stay
    # bit-exact.
    se = s / math.sqrt(n_obs)
    wide = ((ybar - 12.0 * se - 5.0 * s, ybar + 12.0 * se + 5.0 * s), (s / 6.0, 12.0 * s))
    locate = GridPosterior(loglik, logprior, wide, shape=(129, 129))
    mlo, mhi = locate.quantile("mu", [0.0005, 0.9995])
    slo, shi = locate.quantile("sigma", [0.0005, 0.9995])
    mpad = 0.3 * (mhi - mlo)
    bounds = ((mlo - mpad, mhi + mpad), (max(slo * 0.6, s / 50.0), shi * 1.6))
    gp, meta = converge_grid(lambda sh: GridPosterior(loglik, logprior, bounds, sh), s, shape=shape)

    mu, sigma = gp.sample(draws, seed=seed)
    pp_d, ppu_d, ppl_d, ppk_d = _index_draws(mu, sigma, spec)
    ppm_d = _ppm_draws(mu, sigma, spec)
    draws_dict = {"mu": mu, "sigma": sigma, "ppk": ppk_d, "ppm": ppm_d}
    if pp_d is not None:
        draws_dict["pp"] = pp_d

    def med(a):
        return None if a is None else float(np.median(a))

    checks = [_censoring_fraction_check(n_censored, n_total)]

    step = Step(
        operation="bayes.capability_censored",
        params={
            "prior": None if prior is None else prior.to_params(),
            "data_sha256": data_digest(values),
            "seed": int(seed), "draws": int(draws),
            "grid": {"shape": list(meta["shape"]), "refinements": meta["refinements"],
                     "converged": meta["converged"]},
            "method": "grid",
            "truncation": None if truncation is None else [float(truncation[0]), float(truncation[1])],
            "censoring": None if censoring is None else censoring.to_params(),
            "cred_level": float(cred_level),
            "spec": {"lower": lower, "upper": upper, "target": target},
        },
        n_affected=n_total,
        timestamp=_now(),
    )
    history = tuple(base_history) + (step,) + tuple(_assumption_step(a) for a in checks)

    return BayesCensoredCapabilityResult(
        n_total=n_total, n_obs=n_obs, n_censored=n_censored, n_left=n_left, n_right=n_right,
        spec=spec, truncation=truncation,
        censoring=None if censoring is None else censoring.to_params(),
        seed=int(seed), draws=int(draws), cred_level=float(cred_level), grid=meta,
        pp=med(pp_d), ppu=med(ppu_d), ppl=med(ppl_d), ppk=med(ppk_d), ppm=med(ppm_d),
        prior_family=prior_family, _grid=gp, _draws=draws_dict,
        assumptions=checks, history=history,
    )
