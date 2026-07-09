"""Bayesian process capability (spec Algorithm C; Hoff sec 4.2).

Pipeline: validate spec -> sufficient statistics -> closed-form posterior ->
seeded Monte Carlo push-through -> capability indices. The overall-sigma path
follows BDA3 sec 3.2 for the noninformative default and the Normal-Inverse-chi2
conjugate update for an informative NormalPrior. Index definitions match the
classical capability module: Pp/Ppk use the overall sigma; ppm is the per-draw
normal tail probability. The sampling call order (chisquare then normal) is
normative so the T2.5 worked example reproduces bit for bit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from mfgqc import assumptions as _assume
from mfgqc._result import QCResult
from mfgqc.capability import _indices as _classical_indices
from mfgqc.data import Step, _Limits
from mfgqc.errors import MissingPrerequisiteError

from ._results import _assumption_step, _now, data_digest
from .conjugate import mu_marginal, sigma2_marginal, update
from .guardrails import (
    prior_conflict_check,
    prior_weight_check,
    require_min_n,
    small_sample_check,
)


def _index_draws(mu, sigma, spec: _Limits):
    """Vectorized (pp, ppu, ppl, ppk) over draws; None where a limit is absent.
    Same definitions as capability._indices (which drives the point estimate)."""
    pp = ppu = ppl = None
    if spec.lower is not None and spec.upper is not None:
        pp = (spec.upper - spec.lower) / (6.0 * sigma)
    if spec.upper is not None:
        ppu = (spec.upper - mu) / (3.0 * sigma)
    if spec.lower is not None:
        ppl = (mu - spec.lower) / (3.0 * sigma)
    avail = [v for v in (ppu, ppl) if v is not None]
    ppk = np.minimum(avail[0], avail[1]) if len(avail) == 2 else avail[0]
    return pp, ppu, ppl, ppk


def _ppm_draws(mu, sigma, spec: _Limits):
    """Vectorized parts-per-million nonconforming per draw, overall sigma:
    1e6 * [Phi((LSL-mu)/sigma) + Phi(-(USL-mu)/sigma)], one-sided drops a term."""
    total = np.zeros_like(mu)
    if spec.lower is not None:
        total = total + stats.norm.cdf((spec.lower - mu) / sigma)
    if spec.upper is not None:
        total = total + stats.norm.sf((spec.upper - mu) / sigma)
    return 1e6 * total


def _ppm_point(mu: float, sigma: float, spec: _Limits) -> float:
    total = 0.0
    if spec.lower is not None:
        total += float(stats.norm.cdf((spec.lower - mu) / sigma))
    if spec.upper is not None:
        total += float(stats.norm.sf((spec.upper - mu) / sigma))
    return 1e6 * total


@dataclass(frozen=True, repr=False)
class BayesCapabilityResult(QCResult):
    """Posterior capability analysis (immutable). Derived-index posterior draws
    are held under the private ``_draws`` dict and reached via interval()/prob()/
    quantiles(); only scalar summaries are public."""

    n: int
    mean: float
    s: float
    spec: _Limits
    path: str
    seed: int
    draws: int
    cred_level: float
    mun: float
    kn: float
    nun: float
    sn2: float
    pp: float | None
    ppu: float | None
    ppl: float | None
    ppk: float | None
    ppm: float
    prior_family: str | None = None
    _draws: dict = field(default_factory=dict, repr=False)
    assumptions: list = field(default_factory=list)
    history: tuple = field(default_factory=tuple)

    # ---- accessors -------------------------------------------------------
    def interval(self, quantity: str, level: float | None = None) -> tuple:
        """Credible interval. mean/sd use closed-form posterior quantiles; derived
        quantities use equal-tailed draw quantiles (linear interpolation)."""
        level = self.cred_level if level is None else level
        lo_p, hi_p = (1.0 - level) / 2.0, (1.0 + level) / 2.0
        if quantity in ("mu", "mean"):
            d = mu_marginal(self.mun, self.kn, self.nun, self.sn2)
            return float(d.ppf(lo_p)), float(d.ppf(hi_p))
        if quantity in ("sd", "sigma"):
            d = sigma2_marginal(self.nun, self.sn2)
            return math.sqrt(float(d.ppf(lo_p))), math.sqrt(float(d.ppf(hi_p)))
        arr = self._draws[quantity]
        lo, hi = np.quantile(arr, [lo_p, hi_p])
        return float(lo), float(hi)

    def quantiles(self, quantity: str, probs) -> list:
        return [float(v) for v in np.quantile(self._draws[quantity], probs)]

    def prob(self, quantity: str, threshold: float, direction: str = ">=") -> tuple:
        """(p_hat, mcse) for P(quantity >= threshold), mcse = sqrt(p(1-p)/draws)."""
        arr = self._draws[quantity]
        p = float((arr >= threshold).mean()) if direction == ">=" else float((arr <= threshold).mean())
        mcse = math.sqrt(p * (1.0 - p) / arr.size)
        return p, mcse

    # ---- reporting -------------------------------------------------------
    def _title(self) -> str:
        return f"Bayesian Capability ({self.path})"

    def _summary_lines(self) -> list[str]:
        conf = round(self.cred_level * 100)

        def fmt(v):
            return "  n/a" if v is None else f"{v:.4g}"

        mu_lo, mu_hi = self.interval("mu")
        p133, mcse = self.prob("ppk", 1.33)
        ppk_lo, ppk_hi = self.interval("ppk")
        lines = [
            f"n = {self.n}   mean = {self.mean:.5g}   s (overall) = {self.s:.4g}",
            f"mu {conf}% credible interval = ({mu_lo:.5g}, {mu_hi:.5g})",
            f"Pp  = {fmt(self.pp)}    Ppk point = {fmt(self.ppk)}",
            f"Ppk {conf}% credible interval = ({ppk_lo:.3g}, {ppk_hi:.3g})",
            f"P(Ppk >= 1.33) = {p133:.3g} +/- {mcse:.2g} (MC)",
            f"ppm point = {self.ppm:.0f}",
        ]
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
            raise ValueError(f"unknown capability view kind={kind!r}; use None, 'ppk', "
                             f"'mu', 'ppm', or 'predictive'.")


def capability_from_values(y, *, lower: float | None = None, upper: float | None = None,
                           target: float | None = None, prior=None, seed: int,
                           draws: int = 100_000, cred_level: float = 0.95,
                           base_history: tuple = ()) -> BayesCapabilityResult:
    """Bayesian capability from a raw measurement vector and spec limits."""
    values = np.asarray(y, dtype=float)
    values = values[~np.isnan(values)]
    n = values.size
    require_min_n(n)

    spec = _Limits(lower=lower, upper=upper, target=target)
    if not spec.has_any():
        raise MissingPrerequisiteError(
            "bayes capability requires at least one spec limit (lower or upper).",
            analysis="bayes_capability", missing=["spec"])

    ybar = float(values.mean())
    s2 = float(values.var(ddof=1))
    s = math.sqrt(s2)

    if prior is None:
        mun, kn, nun, sn2 = ybar, float(n), float(n - 1), s2  # noninformative, BDA3 sec 3.2
        path, prior_family = "noninformative", None
    else:
        mun, kn, nun, sn2 = update(prior.mu0, prior.k0, prior.nu0, prior.s20, n, ybar, s2)
        path, prior_family = "informative", prior.to_params()["family"]

    # Seeded push-through (normative call order: chisquare then normal).
    rng = np.random.default_rng(seed)
    sig2 = nun * sn2 / rng.chisquare(nun, draws)
    mu = rng.normal(mun, np.sqrt(sig2 / kn))
    sigma = np.sqrt(sig2)

    pp_d, _, _, ppk_d = _index_draws(mu, sigma, spec)
    ppm_d = _ppm_draws(mu, sigma, spec)
    draws_dict = {"mu": mu, "sigma": sigma, "ppk": ppk_d, "ppm": ppm_d}
    if pp_d is not None:
        draws_dict["pp"] = pp_d

    pp, ppu, ppl, ppk = _classical_indices(ybar, s, spec)
    ppm_point = _ppm_point(ybar, s, spec)

    checks = []
    if prior is not None:
        checks.append(prior_weight_check(prior.k0, n))
        checks.append(prior_conflict_check(prior, n, ybar))
    checks.append(small_sample_check(n))
    checks.append(_assume.check_normality(values, cpk_impact=None, context="capability"))

    step = Step(
        operation="bayes.capability",
        params={
            "prior": None if prior is None else prior.to_params(),
            "data_sha256": data_digest(values),
            "seed": int(seed),
            "draws": int(draws),
            "grid": None,
            "tests": [],
            "cred_level": float(cred_level),
            "spec": {"lower": lower, "upper": upper, "target": target},
        },
        n_affected=n,
        timestamp=_now(),
    )
    history = tuple(base_history) + (step,) + tuple(_assumption_step(a) for a in checks)

    return BayesCapabilityResult(
        n=n, mean=ybar, s=s, spec=spec, path=path,
        seed=int(seed), draws=int(draws), cred_level=float(cred_level),
        mun=mun, kn=kn, nun=nun, sn2=sn2,
        pp=pp, ppu=ppu, ppl=ppl, ppk=ppk, ppm=ppm_point,
        prior_family=prior_family, _draws=draws_dict,
        assumptions=checks, history=history,
    )


def compute(qc, *, prior=None, seed: int, draws: int = 100_000,
            cred_level: float = 0.95) -> BayesCapabilityResult:
    """Bayesian capability from a loaded QCData, using its attached spec.

    Mirrors :func:`mfgqc.capability.compute`: reads the measure and spec limits
    off ``qc`` and chains the result onto ``qc``'s provenance history.
    """
    spec = qc.meta.limits
    return capability_from_values(
        qc.values(), lower=spec.lower, upper=spec.upper, target=spec.target,
        prior=prior, seed=seed, draws=draws, cred_level=cred_level,
        base_history=qc.history)
