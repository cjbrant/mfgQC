"""MSA studies beyond gage R&R: bias, linearity, and stability.

These complete the measurement-systems-analysis story. All result objects are
immutable, carry their assumption checks and provenance history, and are
dashboard-ready (structured fields + ``.summary()``). Math follows AIAG MSA
4th ed.; the analysis numbers are never silently altered.
"""

from __future__ import annotations

from . import palette as _pal

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from . import assumptions as _assume
from ._result import QCResult
from .assumptions import AssumptionCheck
from .data import QCData, Step

if TYPE_CHECKING:  # pragma: no cover
    from .control_charts import ControlChartResult


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _assumption_step(a: AssumptionCheck) -> Step:
    return Step(
        operation=f"assumption:{a.name}",
        params={"test": a.test, "passed": a.passed, "magnitude": a.magnitude,
                "reliability": a.reliability, "p_value": a.p_value, "statistic": a.statistic},
        n_affected=None, timestamp=_now(),
    )


# =========================================================================== #
# Bias study
# =========================================================================== #
@dataclass(frozen=True, repr=False)
class BiasResult(QCResult):
    """Bias of repeated measurements of ONE part vs a known reference (AIAG)."""

    reference: float
    mean: float
    bias: float
    sigma_repeat: float
    t_stat: float
    p_value: float
    ci: tuple[float, float]
    df: int
    n: int
    verdict: str
    alpha: float
    _values: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return "MSA Bias Study"

    def _summary_lines(self) -> list[str]:
        conf = round((1.0 - self.alpha) * 100)
        return [
            f"n = {self.n}   reference = {self.reference:.5g}   mean = {self.mean:.5g}",
            f"bias = {self.bias:+.4g}   (sigma_repeat = {self.sigma_repeat:.4g})",
            f"t = {self.t_stat:.3f}   p = {self.p_value:.3g}   df = {self.df}",
            f"{conf}% CI on bias: ({self.ci[0]:+.4g}, {self.ci[1]:+.4g})",
            f"Verdict: bias is {self.verdict} "
            f"({'0 within the CI' if self.verdict == 'acceptable' else '0 outside the CI'}).",
        ]

    def summary(self) -> dict:
        return {
            "reference": self.reference, "mean": self.mean, "bias": self.bias,
            "sigma_repeat": self.sigma_repeat, "t": self.t_stat, "p_value": self.p_value,
            "CI_low": self.ci[0], "CI_high": self.ci[1], "df": self.df, "n": self.n,
            "verdict": self.verdict, "confidence": round((1.0 - self.alpha) * 100),
        }

    def _render_standalone(self, fig, kind, **kwargs):
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        ax.hist(self._values, bins="auto", color=_pal.active().data, edgecolor=_pal.active().bg)
        ax.axvline(self.reference, color=_pal.active().target, ls="--", lw=1.5, label=f"reference {self.reference:.4g}")
        ax.axvline(self.mean, color=_pal.active().ooc, ls="-", lw=1.5, label=f"mean {self.mean:.4g}")
        ax.set_title(f"Bias study (bias = {self.bias:+.4g}, {self.verdict})", fontsize=10)
        ax.legend(fontsize=8)
        return ax


def bias(qc: QCData, reference: float, *, alpha: float = 0.05) -> BiasResult:
    """Bias study: n repeated measurements of one part vs a known ``reference``.

    bias = mean - reference; t = bias / (sigma_repeat / sqrt(n)); bias is
    statistically zero (acceptable) at ``alpha`` if 0 lies within the t CI.

    Degrees of freedom: mfgQC uses the standard ``n - 1`` (a defensible, common
    choice). For the AIAG bias example this reproduces AIAG's published t = 0.537
    and its CI (-0.0299, 0.0519) exactly. (AIAG's manual prints a d2*-effective df
    of 1.993, but that value is internally inconsistent with its own stated CI -
    df=1.993 implies t_{.975} = 4.32 and a much wider CI; the n-1 df is what
    reproduces the printed interval, so mfgQC uses it.)
    """
    values = qc.values()
    values = values[~np.isnan(values)]
    n = values.size
    if n < 2:
        raise ValueError("bias study needs at least 2 measurements.")
    mean = float(values.mean())
    sigma = float(values.std(ddof=1))
    bias_val = mean - reference
    df = n - 1
    se = sigma / np.sqrt(n)
    t_stat = bias_val / se if se > 0 else float("inf")
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df))
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    ci = (bias_val - t_crit * se, bias_val + t_crit * se)
    verdict = "acceptable" if ci[0] <= 0 <= ci[1] else "not acceptable"

    checks = [_assume.check_normality(values)]
    step = Step(operation="bias_study",
                params={"reference": reference, "bias": bias_val, "t": t_stat,
                        "p_value": p_value, "verdict": verdict},
                n_affected=n, timestamp=_now())
    history = qc.history + (step,) + tuple(_assumption_step(a) for a in checks)
    return BiasResult(
        reference=float(reference), mean=mean, bias=bias_val, sigma_repeat=sigma,
        t_stat=float(t_stat), p_value=p_value, ci=(float(ci[0]), float(ci[1])),
        df=df, n=n, verdict=verdict, alpha=alpha, _values=values,
        assumptions=checks, history=history,
    )


# =========================================================================== #
# Linearity study
# =========================================================================== #
@dataclass(frozen=True, repr=False)
class LinearityResult(QCResult):
    """Linearity: regress bias on reference across the operating range (AIAG)."""

    slope: float
    intercept: float
    t_slope: float
    t_intercept: float
    p_slope: float
    p_intercept: float
    se_slope: float
    se_intercept: float
    r_squared: float
    df: int
    n: int
    references: tuple[float, ...]
    ref_bias: tuple[float, ...]  # mean bias at each distinct reference
    verdict: str
    alpha: float
    _x: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _bias: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return "MSA Linearity Study"

    def _summary_lines(self) -> list[str]:
        return [
            f"n = {self.n}   references = {len(self.references)}   df = {self.df}",
            f"bias = {self.slope:+.4g} * ref + {self.intercept:+.4g}   (R^2 = {self.r_squared:.4f})",
            f"slope:     {self.slope:+.4g}  (t = {self.t_slope:.3f}, p = {self.p_slope:.3g})",
            f"intercept: {self.intercept:+.4g}  (t = {self.t_intercept:.3f}, p = {self.p_intercept:.3g})",
            f"Verdict: linearity is {self.verdict}.",
            ("  (slope and intercept both ~0 -> bias=0 line within the bands)"
             if self.verdict == "acceptable"
             else "  (slope and/or intercept != 0 -> bias varies across the range)"),
        ]

    def summary(self) -> dict:
        return {
            "slope": self.slope, "intercept": self.intercept,
            "t_slope": self.t_slope, "t_intercept": self.t_intercept,
            "p_slope": self.p_slope, "p_intercept": self.p_intercept,
            "R_squared": self.r_squared, "df": self.df, "n": self.n,
            "verdict": self.verdict, "confidence": round((1.0 - self.alpha) * 100),
        }

    def _render_standalone(self, fig, kind, **kwargs):
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        x, y = self._x, self._bias
        ax.scatter(x, y, s=14, color=_pal.active().data, edgecolor=_pal.active().center, alpha=0.7, label="bias")
        xs = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(xs, self.slope * xs + self.intercept, color=_pal.active().ooc, lw=1.5, label="fitted")
        # confidence bands for the mean response
        xbar = float(x.mean())
        sxx = float(np.sum((x - xbar) ** 2))
        s = self.se_slope * np.sqrt(sxx) if sxx > 0 else 0.0
        tcrit = float(stats.t.ppf(1 - self.alpha / 2, self.df)) if self.df > 0 else 0.0
        band = tcrit * s * np.sqrt(1.0 / self.n + (xs - xbar) ** 2 / sxx) if sxx > 0 else np.zeros_like(xs)
        yhat = self.slope * xs + self.intercept
        ax.plot(xs, yhat + band, color=_pal.active().ooc, ls=":", lw=1, label="conf. band")
        ax.plot(xs, yhat - band, color=_pal.active().ooc, ls=":", lw=1)
        ax.axhline(0.0, color=_pal.active().target, ls="--", lw=1.3, label="bias = 0")
        ax.set_xlabel("reference"); ax.set_ylabel("bias")
        ax.set_title(f"Linearity ({self.verdict}, R^2={self.r_squared:.3f})", fontsize=10)
        ax.legend(fontsize=8)
        return ax


def _resolve_reference(qc: QCData, reference: "str | dict") -> np.ndarray:
    """Return a per-row reference vector from a column name or a {group: ref} mapping."""
    frame = qc.frame
    if isinstance(reference, str):
        if reference not in frame.columns:
            raise ValueError(f"reference column {reference!r} not found.")
        return frame[reference].to_numpy(dtype=float)
    if isinstance(reference, dict):
        # map via the 'part' or 'subgroup' role, else a same-named column
        for role in ("part", "subgroup"):
            if role in qc.meta.roles:
                key = qc.meta.roles[role]
                return frame[key].map(reference).to_numpy(dtype=float)
        raise ValueError("a dict reference needs a 'part' or 'subgroup' role to key on.")
    raise TypeError("reference must be a column name (str) or a {group: ref} dict.")


def linearity(qc: QCData, reference: "str | dict", *, alpha: float = 0.05) -> LinearityResult:
    """Linearity study: regress per-measurement bias on the reference value.

    Fits ``bias = slope*ref + intercept`` over all measurements; tests H0: slope=0
    and H0: intercept=0. Linearity is acceptable when neither is rejected (the
    bias=0 line lies within the regression's confidence bands).
    """
    ref = _resolve_reference(qc, reference)
    y_meas = qc.values()
    mask = ~(np.isnan(ref) | np.isnan(y_meas))
    ref, y_meas = ref[mask], y_meas[mask]
    bias_vec = y_meas - ref
    n = ref.size
    if n < 3 or np.unique(ref).size < 2:
        raise ValueError("linearity needs >=3 measurements across >=2 distinct reference values.")

    x = ref
    xbar = float(x.mean())
    sxx = float(np.sum((x - xbar) ** 2))
    slope, intercept = np.polyfit(x, bias_vec, 1)
    fitted = slope * x + intercept
    resid = bias_vec - fitted
    df = n - 2
    sse = float(np.sum(resid ** 2))
    s2 = sse / df
    s = np.sqrt(s2)
    se_slope = s / np.sqrt(sxx)
    se_intercept = s * np.sqrt(1.0 / n + xbar ** 2 / sxx)
    t_slope = slope / se_slope if se_slope > 0 else float("inf")
    t_intercept = intercept / se_intercept if se_intercept > 0 else float("inf")
    p_slope = float(2.0 * stats.t.sf(abs(t_slope), df))
    p_intercept = float(2.0 * stats.t.sf(abs(t_intercept), df))
    sst = float(np.sum((bias_vec - bias_vec.mean()) ** 2))
    r_squared = 1.0 - sse / sst if sst > 0 else 0.0

    verdict = "acceptable" if (p_slope >= alpha and p_intercept >= alpha) else "not acceptable"

    refs = tuple(sorted(float(v) for v in np.unique(x)))
    ref_bias = tuple(float(bias_vec[x == r].mean()) for r in refs)

    checks = [_assume.check_normality(resid)]
    step = Step(operation="linearity_study",
                params={"slope": float(slope), "intercept": float(intercept),
                        "t_slope": float(t_slope), "t_intercept": float(t_intercept),
                        "r_squared": r_squared, "verdict": verdict},
                n_affected=n, timestamp=_now())
    history = qc.history + (step,) + tuple(_assumption_step(a) for a in checks)
    return LinearityResult(
        slope=float(slope), intercept=float(intercept),
        t_slope=float(t_slope), t_intercept=float(t_intercept),
        p_slope=p_slope, p_intercept=p_intercept,
        se_slope=float(se_slope), se_intercept=float(se_intercept),
        r_squared=float(r_squared), df=df, n=n,
        references=refs, ref_bias=ref_bias, verdict=verdict, alpha=alpha,
        _x=x, _bias=bias_vec, assumptions=checks, history=history,
    )


# =========================================================================== #
# Stability study (control chart of a reference part over time)
# =========================================================================== #
@dataclass(frozen=True, repr=False)
class StabilityResult(QCResult):
    """Measurement-system stability: a control chart of a reference part over time.

    Stable iff the chart shows no out-of-control signals (special causes)."""

    chart: "ControlChartResult"
    n_signals: int
    stable: bool
    verdict: str
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return "MSA Stability Study"

    def _summary_lines(self) -> list[str]:
        lines = [
            f"Chart: {self.chart.kind}",
            f"Out-of-control signals: {self.n_signals}",
            f"Verdict: measurement system is {self.verdict}.",
        ]
        if not self.stable:
            for v in self.chart.violations[:8]:
                lines.append(f"  - {v}")
        return lines

    def summary(self) -> dict:
        return {
            "chart_kind": self.chart.kind,
            "n_signals": self.n_signals,
            "stable": self.stable,
            "verdict": self.verdict,
        }

    def _render_standalone(self, fig, kind, **kwargs):
        self.chart._render_standalone(fig, kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        self.chart._render_axes(ax, kind, **kwargs)


def stability(qc: QCData, *, kind: str | None = None, rules: str = "nelson") -> StabilityResult:
    """Stability study: put the reference-part-over-time measurements on a control
    chart and check for special causes. Thin specialization over ``control_chart``.

    When the data carries no ``subgroup``/``time`` role and no ``subgroup_size``
    (the common case: one reference part measured once per period), the study
    defaults to INDIVIDUALS - an I-MR chart over the row sequence - rather than
    erroring. Declare a subgroup role / subgroup_size to chart subgroups instead.
    """
    from dataclasses import replace as _replace
    meta = qc.meta
    has_structure = ("subgroup" in meta.roles or "time" in meta.roles
                     or meta.subgroup_size is not None)
    if not has_structure:
        qc = QCData(_frame=qc._frame, meta=_replace(meta, subgroup_size=1), history=qc.history)
    chart = qc.control_chart(kind=kind, rules=rules)
    n_signals = len(chart.violations)
    stable = n_signals == 0
    verdict = "stable" if stable else "unstable"
    step = Step(operation="stability_study",
                params={"chart": chart.kind, "n_signals": n_signals, "verdict": verdict},
                n_affected=None, timestamp=_now())
    history = qc.history + (step,)
    return StabilityResult(chart=chart, n_signals=n_signals, stable=stable,
                           verdict=verdict, assumptions=list(chart.assumptions), history=history)
