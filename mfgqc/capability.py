"""Process-capability analysis.

Correctness anchor (watch-list #1): Cp/Cpk use the WITHIN-subgroup sigma
(short-term, R-bar/d2 when subgroups exist) while Pp/Ppk ALWAYS use the overall
sigma (long-term, ordinary sample SD). The two are reported separately and never
conflated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from scipy import stats

from . import assumptions as _assume
from ._result import QCResult
from .assumptions import AssumptionCheck
from .constants import control_constant
from .data import QCData, Step, _Limits

_VALID_METHODS = ("normal", "boxcox", "clements", "percentile", "johnson")


@dataclass(frozen=True, repr=False)
class CapabilityResult(QCResult):
    """Result of a capability analysis (immutable)."""

    method: str
    spec: _Limits
    n: int
    mean: float
    sigma_within: float | None
    sigma_overall: float
    sigma_used: str  # "within (R-bar/d2)", "within (MR-bar/d2)", "within (pooled)", "overall"
    cp: float | None
    cpu: float | None
    cpl: float | None
    cpk: float | None
    pp: float | None
    ppu: float | None
    ppl: float | None
    ppk: float | None
    cpm: float | None
    alpha: float = 0.05
    cp_ci: tuple[float, float] | None = None
    cpk_ci: tuple[float, float] | None = None
    _values: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _transform: object = field(repr=False, default=None)  # callable for non-normal fits
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"Process Capability (method={self.method})"

    def _summary_lines(self) -> list[str]:
        def fmt(v: float | None) -> str:
            return "  n/a" if v is None else f"{v:.4g}"

        conf = round((1.0 - self.alpha) * 100)

        def ci(c: tuple[float, float] | None) -> str:
            if c is not None:
                return f"  {conf}% CI ({c[0]:.3g}, {c[1]:.3g})"
            # non-normal methods have no standard normal-theory CI; say so, don't fake one
            return "" if self.method == "normal" else "  CI: n/a (non-normal method)"

        lines = [
            f"n = {self.n}   mean = {self.mean:.5g}",
            f"sigma (within)  = {fmt(self.sigma_within)}",
            f"sigma (overall) = {self.sigma_overall:.5g}",
            f"Cp/Cpk sigma    = {self.sigma_used}",
            "",
            f"Cp  = {fmt(self.cp)}{ci(self.cp_ci)}",
            f"Cpk = {fmt(self.cpk)}{ci(self.cpk_ci)}   (Cpu={fmt(self.cpu)}, Cpl={fmt(self.cpl)})",
            f"Pp  = {fmt(self.pp)}    Ppk = {fmt(self.ppk)}   "
            f"(Ppu={fmt(self.ppu)}, Ppl={fmt(self.ppl)})",
            f"Cpm = {fmt(self.cpm)}",
        ]
        return lines

    def summary(self) -> dict:
        """Flat {label: value} dict of the headline numbers (dashboard-ready)."""
        normality = next((a for a in self.assumptions if a.name == "normality"), None)
        return {
            "method": self.method,
            "n": self.n,
            "mean": self.mean,
            "sigma_within": self.sigma_within,
            "sigma_overall": self.sigma_overall,
            "Cp": self.cp,
            "Cp_CI_low": None if self.cp_ci is None else self.cp_ci[0],
            "Cp_CI_high": None if self.cp_ci is None else self.cp_ci[1],
            "Cpk": self.cpk,
            "Cpk_CI_low": None if self.cpk_ci is None else self.cpk_ci[0],
            "Cpk_CI_high": None if self.cpk_ci is None else self.cpk_ci[1],
            "Pp": self.pp,
            "Ppk": self.ppk,
            "Cpm": self.cpm,
            "confidence": round((1.0 - self.alpha) * 100),
            "normality_passed": None if normality is None else normality.passed,
        }

    # ---- plotting --------------------------------------------------------
    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        self._render_axes(ax, kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        from . import plotting
        if kind in (None, "histogram", "capability"):
            plotting.capability_histogram(ax, self)
        elif kind in ("probability", "probplot"):
            plotting.capability_probplot(ax, self)
        else:
            raise ValueError(f"unknown capability view kind={kind!r}; use None or 'probability'.")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _assumption_step(a: AssumptionCheck) -> Step:
    return Step(
        operation=f"assumption:{a.name}",
        params={
            "test": a.test, "passed": a.passed, "magnitude": a.magnitude,
            "reliability": a.reliability, "p_value": a.p_value, "statistic": a.statistic,
        },
        n_affected=None,
        timestamp=_now(),
    )


def _within_sigma(qc: QCData) -> tuple[float | None, str]:
    """Estimate within-subgroup sigma. Returns (sigma, label) or (None, 'overall')."""
    try:
        sg = qc.subgroups()
    except ValueError:
        return None, "overall"
    groups = sg.groups
    if not groups:
        return None, "overall"
    if sg.equal_n and sg.n is not None and sg.n >= 2:
        n = sg.n
        ranges = np.array([g.max() - g.min() for g in groups], dtype=float)
        rbar = float(ranges.mean())
        d2 = control_constant("d2", n)
        return rbar / d2, "within (R-bar/d2)"
    if sg.equal_n and sg.n == 1:
        flat = np.concatenate(groups)
        mr = np.abs(np.diff(flat))
        if mr.size == 0:
            return None, "overall"
        return float(mr.mean()) / control_constant("d2", 2), "within (MR-bar/d2)"
    # Unequal subgroup sizes -> pooled within-subgroup standard deviation.
    num = 0.0
    den = 0.0
    for g in groups:
        if g.size >= 2:
            num += (g.size - 1) * float(np.var(g, ddof=1))
            den += g.size - 1
    if den <= 0:
        return None, "overall"
    return float(np.sqrt(num / den)), "within (pooled)"


def _indices(mu: float, sigma: float, spec: _Limits) -> tuple[float | None, float | None, float | None, float | None]:
    """Return (cp, cpu, cpl, cpk) for a given sigma. None where the limit is absent."""
    cp = cpu = cpl = None
    if spec.lower is not None and spec.upper is not None:
        cp = (spec.upper - spec.lower) / (6.0 * sigma)
    if spec.upper is not None:
        cpu = (spec.upper - mu) / (3.0 * sigma)
    if spec.lower is not None:
        cpl = (mu - spec.lower) / (3.0 * sigma)
    available = [v for v in (cpu, cpl) if v is not None]
    cpk = min(available) if available else None
    return cp, cpu, cpl, cpk


# Candidate distributions for the auto-fitting percentile (Clements / ISO 22514)
# method. Positive-support families are fit with floc=0 so they recover the
# data-generating percentiles (e.g. a 2-parameter lognormal) instead of drifting
# onto a spurious location shift.
_PERCENTILE_CANDIDATES = (
    ("normal", stats.norm, False),
    ("lognormal", stats.lognorm, True),
    ("gamma", stats.gamma, True),
    ("weibull", stats.weibull_min, True),
    ("exponential", stats.expon, True),
)


def _fit_best_distribution(values: np.ndarray):
    """Return (frozen_distribution, name) of the best fit by log-likelihood."""
    best = None
    positive_ok = float(np.min(values)) > 0
    for name, dist, positive in _PERCENTILE_CANDIDATES:
        if positive and not positive_ok:
            continue
        try:
            params = dist.fit(values, floc=0) if positive else dist.fit(values)
            ll = float(np.sum(dist.logpdf(values, *params)))
        except Exception:
            continue
        if np.isfinite(ll) and (best is None or ll > best[0]):
            best = (ll, name, dist(*params))
    if best is None:
        return stats.norm(*stats.norm.fit(values)), "normal"
    return best[2], best[1]


def _cpk_shift(values, mu, sigma_within, sigma_overall, spec) -> float | None:
    """Relative Cpk shift between the normal and the auto-fit non-normal method.

    This is the effect-size magnitude for the normality assumption: it measures
    how much the non-normality actually moves the reported index. Returns None if
    it cannot be computed (no usable index).
    """
    sigma = sigma_within if sigma_within is not None else sigma_overall
    _, _, _, cpk_norm = _indices(mu, sigma, spec)
    try:
        frozen, _ = _fit_best_distribution(values)
        lo, med, hi = (float(v) for v in frozen.ppf([0.00135, 0.5, 0.99865]))
        _, _, _, cpk_pct = _percentile_indices(lo, med, hi, spec)
    except Exception:
        return None
    if cpk_norm is None or cpk_pct is None or abs(cpk_pct) < 1e-9:
        return None
    return abs(cpk_norm - cpk_pct) / abs(cpk_pct)


def _percentile_indices(x_low: float, median: float, x_high: float, spec: _Limits):
    """Percentile-method (Clements / ISO 22514) capability indices.

    ``Cp=(USL-LSL)/(X.99865-X.00135)``, ``Cpu=(USL-M)/(X.99865-M)``,
    ``Cpl=(M-LSL)/(M-X.00135)``, ``Cpk=min(Cpu, Cpl)``.
    """
    cp = cpu = cpl = None
    if spec.lower is not None and spec.upper is not None:
        cp = (spec.upper - spec.lower) / (x_high - x_low)
    if spec.upper is not None:
        cpu = (spec.upper - median) / (x_high - median)
    if spec.lower is not None:
        cpl = (median - spec.lower) / (median - x_low)
    available = [v for v in (cpu, cpl) if v is not None]
    cpk = min(available) if available else None
    return cp, cpu, cpl, cpk


def _capability_cis(n: int, cp: float | None, cpk: float | None, alpha: float
                    ) -> tuple[tuple | None, tuple | None]:
    """Normal-theory CIs for Cp (chi-square, Montgomery eq 8.19) and Cpk
    (approx normal, eq 8.21). Returns (cp_ci, cpk_ci); each None if not computable."""
    if n < 2:
        return None, None
    cp_ci = None
    if cp is not None:
        lo = cp * float(np.sqrt(stats.chi2.ppf(alpha / 2, n - 1) / (n - 1)))
        hi = cp * float(np.sqrt(stats.chi2.ppf(1 - alpha / 2, n - 1) / (n - 1)))
        cp_ci = (lo, hi)
    cpk_ci = None
    if cpk is not None and cpk != 0:
        z = float(stats.norm.ppf(1 - alpha / 2))
        m = z * float(np.sqrt(1.0 / (9.0 * n * cpk**2) + 1.0 / (2.0 * (n - 1))))
        cpk_ci = (cpk * (1 - m), cpk * (1 + m))
    return cp_ci, cpk_ci


def compute(qc: QCData, *, method: str = "normal", alpha: float = 0.05) -> CapabilityResult:
    """Compute process-capability indices.

    Parameters
    ----------
    qc : QCData
        Source data. Requires at least one spec limit.
    method : str, optional
        ``"normal"`` (default) or a non-normal method (``"boxcox"``,
        ``"clements"``, ``"johnson"``). Non-normal methods do NOT run unless
        explicitly requested; the default never transforms.
    alpha : float, optional
        Significance level for the Cp/Cpk confidence intervals (default 0.05 ->
        95% CIs). CIs are reported only for the normal method; non-normal methods
        report ``CI: n/a`` rather than a faked normal-theory interval.

    Returns
    -------
    CapabilityResult
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}; got {method!r}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}.")
    spec = qc.meta.limits
    if not spec.has_any():
        from .errors import MissingPrerequisiteError
        raise MissingPrerequisiteError(
            "capability requires at least one spec limit (lower or upper); "
            "set it with .spec(lower=, upper=).",
            analysis="capability", missing=["spec"])

    values = qc.values()
    values = values[~np.isnan(values)]
    n = values.size
    mu = float(values.mean())
    sigma_overall = float(values.std(ddof=1))

    sigma_within, within_label = _within_sigma(qc)
    transform = None

    # Normality severity is driven by the EFFECT ON THE INDEX (Cpk shift between the
    # normal and the auto-fit non-normal method) - this answers "does the non-normality
    # move the number you care about" and self-corrects large-n over-detection.
    checks: list[AssumptionCheck] = []
    shift = _cpk_shift(values, mu, sigma_within, sigma_overall, spec)
    checks.append(_assume.check_normality(values, cpk_impact=shift, context="capability"))

    if method == "normal":
        cpk_sigma = sigma_within if sigma_within is not None else sigma_overall
        sigma_used = within_label if sigma_within is not None else "overall"
        cp, cpu, cpl, cpk = _indices(mu, cpk_sigma, spec)
        pp, ppu, ppl, ppk = _indices(mu, sigma_overall, spec)
        # subgroup sufficiency, only relevant when using within-sigma
        if sigma_within is not None:
            try:
                k = len(qc.subgroups().groups)
            except ValueError:
                k = 0
            checks.append(AssumptionCheck(
                name="subgroup_sufficiency", test="subgroup count >= 25",
                statistic=float(k), p_value=None, passed=k >= 25,
                magnitude=float(k), magnitude_label="subgroup count", reliability="ok", n=k,
                recommendation=None if k >= 25 else (
                    f"Only {k} subgroups; >=25 recommended for a stable within-sigma estimate."),
            ))
    else:
        # Non-normal capability.
        if method == "boxcox":
            shift = 0.0
            v = values
            if np.min(v) <= 0:
                shift = 1e-9 - float(np.min(v))
                v = v + shift
            tv, lam = stats.boxcox(v)
            tmu, tsigma = float(np.mean(tv)), float(np.std(tv, ddof=1))

            def _bc(x):
                xx = np.asarray(x, dtype=float) + shift
                return (np.power(xx, lam) - 1.0) / lam if lam != 0 else np.log(xx)

            tspec = _Limits(
                lower=None if spec.lower is None else float(_bc(spec.lower)),
                upper=None if spec.upper is None else float(_bc(spec.upper)),
                target=None if spec.target is None else float(_bc(spec.target)),
            )
            cp, cpu, cpl, cpk = _indices(tmu, tsigma, tspec)
            pp, ppu, ppl, ppk = cp, cpu, cpl, cpk  # boxcox is a long-term transform
            sigma_used = f"box-cox (lambda={lam:.3g})"
            transform = ("boxcox", lam, shift)
        else:
            # Percentile method (Clements / ISO 22514). 'clements'/'percentile'
            # auto-fit the best parametric distribution; 'johnson' uses Johnson-SU.
            if method in ("clements", "percentile"):
                frozen, dist_name = _fit_best_distribution(values)
            else:  # johnson
                frozen = stats.johnsonsu(*stats.johnsonsu.fit(values))
                dist_name = "johnson-su"
            x_low, median, x_high = (float(v) for v in frozen.ppf([0.00135, 0.5, 0.99865]))
            cp, cpu, cpl, cpk = _percentile_indices(x_low, median, x_high, spec)
            # Percentile method is a long-term (overall) view; Pp* mirror Cp*.
            pp, ppu, ppl, ppk = cp, cpu, cpl, cpk
            sigma_used = f"{method} percentile ({dist_name} fit)"
            transform = (method, dist_name)

    # Cpm (two-sided with target), overall-sigma based about the target.
    cpm = None
    if spec.lower is not None and spec.upper is not None and spec.target is not None:
        tau = np.sqrt(sigma_overall**2 + (mu - spec.target) ** 2)
        cpm = (spec.upper - spec.lower) / (6.0 * tau)

    analysis_step = Step(
        operation="capability",
        params={
            "method": method, "sigma_used": sigma_used,
            "cp": cp, "cpk": cpk, "pp": pp, "ppk": ppk, "cpm": cpm,
        },
        n_affected=n,
        timestamp=_now(),
    )
    # Confidence intervals: normal-theory only (Montgomery 8.3.5). Non-normal
    # methods get None -> reported as "CI: n/a (non-normal method)".
    if method == "normal":
        cp_ci, cpk_ci = _capability_cis(n, cp, cpk, alpha)
    else:
        cp_ci = cpk_ci = None

    history = qc.history + (analysis_step,) + tuple(_assumption_step(a) for a in checks)

    return CapabilityResult(
        method=method, spec=spec, n=n, mean=mu,
        sigma_within=sigma_within, sigma_overall=sigma_overall, sigma_used=sigma_used,
        cp=cp, cpu=cpu, cpl=cpl, cpk=cpk,
        pp=pp, ppu=ppu, ppl=ppl, ppk=ppk, cpm=cpm,
        alpha=alpha, cp_ci=cp_ci, cpk_ci=cpk_ci,
        _values=values, _transform=transform,
        assumptions=checks, history=history,
    )
