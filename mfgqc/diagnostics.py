"""Model-interpretive diagnostics: QC on a predictive model's residuals.

This module treats a model's prediction error as a *process characteristic* and
runs the same SPC machinery mfgQC applies to a manufacturing process. The whole
point is the HARD BOUNDARY below.

Framework-agnostic by construction
----------------------------------
mfgQC NEVER imports or branches on any ML framework. It does not know (and must
never need to know) whether the predictor is scikit-learn, PyTorch, TensorFlow,
XGBoost, a closed-form formula, or a lookup table. The only thing it ever takes
from a model is its predictions, via a single DUCK-TYPED ``model.predict(X)``
call. Arrays in, QC verdict out.

What you get
------------
* **Tier 1 (always):** residual assumption checks - normality, homoscedasticity,
  independence (Durbin-Watson + lag-1 autocorrelation) and zero-mean / bias.
* **Tier 2 (when ``tolerance`` is given):** the fraction of residuals inside the
  tolerance band and a *residual capability index* (a Cpk computed on the error
  distribution against the tolerance as if it were a spec).
* **Tier 3 (when ``order`` is given):** the residuals, re-sequenced by ``order``,
  on an individuals (I-MR) control chart, yielding a drift verdict.

The result, :class:`ModelDiagnosticResult`, is a frozen :class:`QCResult`: it
reports, summarises to a flat dict, and renders a residual diagnostic panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from scipy import stats

from . import assumptions as _assume
from . import palette as _pal
from ._result import QCResult
from .assumptions import AssumptionCheck, reliability
from .data import Step

ALPHA = _assume.ALPHA


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


# --------------------------------------------------------------------------- #
# Residual assumption checks (built in the AssumptionCheck v2 style:
# `passed` from the direct test; magnitude / reliability are adjacent context;
# recommend, don't switch).
# --------------------------------------------------------------------------- #
def _check_homoscedasticity(resid: np.ndarray, fitted: np.ndarray, *,
                            alpha: float = ALPHA) -> AssumptionCheck:
    """Constant residual variance vs the fitted value.

    A simple Breusch-Pagan-flavoured test: correlate ``|residual|`` with the
    fitted value and test that correlation against zero. ``passed`` is the direct
    test at alpha; the correlation magnitude is the practical-impact context.
    """
    r = np.asarray(resid, dtype=float)
    f = np.asarray(fitted, dtype=float)
    mask = ~(np.isnan(r) | np.isnan(f))
    r, f = r[mask], f[mask]
    n = r.size
    if n < 4 or float(np.std(f)) == 0.0:
        return AssumptionCheck("homoscedasticity", "|resid| vs fitted correlation",
                               float("nan"), None, True, None, "abs-resid corr",
                               "low_power", n, None)
    rho, p = stats.pearsonr(np.abs(r), f)
    rho, p = float(rho), float(p)
    passed = p >= alpha
    rec = None
    if not passed:
        rec = (f"Residual spread varies with the prediction (|resid|-vs-fitted r={rho:.2f}, "
               f"p={p:.3g}); the model's error is heteroscedastic - prediction intervals "
               "and any constant-tolerance capability are unreliable across the range.")
    return AssumptionCheck("homoscedasticity", "|resid| vs fitted correlation",
                           rho, p, passed, abs(rho), "abs-resid corr",
                           reliability(n), n, rec)


def _check_independence_dw(resid: np.ndarray, *, alpha: float = ALPHA) -> AssumptionCheck:
    """Residual independence via Durbin-Watson, with lag-1 autocorrelation context.

    ``passed`` is the direct serial-correlation test: the Durbin-Watson statistic
    lying in ``(1.5, 2.5)`` (the conventional "no meaningful serial correlation"
    band; DW=2 is the no-autocorrelation ideal). The lag-1 autocorrelation is the
    practical-impact context.
    """
    r = np.asarray(resid, dtype=float)
    r = r[~np.isnan(r)]
    n = r.size
    if n < 4:
        return AssumptionCheck("independence", "Durbin-Watson", float("nan"), None,
                               True, None, "lag-1 autocorr", "low_power", n, None)
    xc = r - r.mean()
    denom = float(np.sum(xc * xc))
    dw = float(np.sum(np.diff(r) ** 2) / denom) if denom > 0 else 2.0
    acf1 = float(np.sum(xc[1:] * xc[:-1]) / denom) if denom > 0 else 0.0
    passed = 1.5 < dw < 2.5
    rec = None
    if not passed:
        direction = "positive" if dw < 2.0 else "negative"
        rec = (f"Residuals are serially correlated (Durbin-Watson={dw:.2f}, lag-1 "
               f"autocorr={acf1:.2f}, {direction}); successive errors are not "
               "independent - the model is missing time/sequence structure (e.g. drift "
               "or an unmodelled trend).")
    return AssumptionCheck("independence", "Durbin-Watson", dw, None,
                           passed, abs(acf1), "lag-1 autocorr",
                           "low_power" if n < 30 else reliability(n), n, rec)


def _check_zero_mean(resid: np.ndarray, *, alpha: float = ALPHA) -> AssumptionCheck:
    """Zero-mean residuals (no systematic bias) via a one-sample t-test vs 0.

    ``passed`` is the direct test: the residual mean is NOT significantly
    different from zero. The standardised bias (mean / std) is the context.
    """
    r = np.asarray(resid, dtype=float)
    r = r[~np.isnan(r)]
    n = r.size
    sd = float(np.std(r, ddof=1)) if n > 1 else 0.0
    if n < 2 or sd == 0.0:
        return AssumptionCheck("zero_mean_bias", "one-sample t vs 0", float("nan"), None,
                               True, None, "std. bias", "low_power", n, None)
    t, p = stats.ttest_1samp(r, 0.0)
    t, p = float(t), float(p)
    mean = float(r.mean())
    std_bias = mean / sd
    passed = p >= alpha
    rec = None
    if not passed:
        sign = "over-predicting" if mean < 0 else "under-predicting"
        rec = (f"Residuals are not zero-mean (mean={mean:.3g}, t={t:.2f}, p={p:.3g}); the "
               f"model is systematically {sign} - recalibrate the intercept / bias.")
    return AssumptionCheck("zero_mean_bias", "one-sample t vs 0", t, p,
                           passed, abs(std_bias), "std. bias", reliability(n), n, rec)


def _normalize_tolerance(tolerance) -> tuple[float, float]:
    """Coerce a tolerance spec into a ``(lo, hi)`` band.

    Accepts a ``(lo, hi)`` pair or a single number ``v`` -> ``(-v, v)``.
    """
    if np.isscalar(tolerance):
        v = abs(float(tolerance))
        return -v, v
    lo, hi = tolerance
    lo, hi = float(lo), float(hi)
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class ModelDiagnosticResult(QCResult):
    """Result of a model residual diagnostic (immutable).

    The error distribution is summarised as a process: location/scale numbers,
    fit-quality numbers (RMSE/MAE), an optional tolerance-based capability view,
    and an optional drift verdict from a residual control chart.
    """

    n: int
    mean: float
    std: float
    rmse: float
    mae: float

    # Tier 2 (tolerance) - None when no tolerance was supplied.
    tolerance: tuple[float, float] | None = None
    pct_within: float | None = None
    residual_cpk: float | None = None
    residual_cpu: float | None = None
    residual_cpl: float | None = None

    # Tier 3 (order) - None / "n/a" when no order was supplied.
    drift: str = "n/a"
    n_signals: int | None = None
    chart: object = field(repr=False, default=None)  # ControlChartResult or None

    _resid: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _fitted: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    # ---- reporting -------------------------------------------------------
    def _title(self) -> str:
        return "Model Diagnostic (residuals as a process)"

    def _summary_lines(self) -> list[str]:
        def fmt(v):
            return "  n/a" if v is None else f"{v:.4g}"

        lines = [
            f"n = {self.n}   residual mean = {self.mean:.5g}   residual std = {self.std:.5g}",
            f"RMSE = {self.rmse:.5g}   MAE = {self.mae:.5g}",
        ]
        if self.tolerance is not None:
            lo, hi = self.tolerance
            lines += [
                "",
                f"Tolerance band   = ({lo:.4g}, {hi:.4g})",
                f"Within tolerance = {fmt(self.pct_within)}%",
                f"Residual Cpk     = {fmt(self.residual_cpk)}   "
                f"(Cpu={fmt(self.residual_cpu)}, Cpl={fmt(self.residual_cpl)})",
            ]
        if self.chart is not None:
            lines += [
                "",
                f"Drift (I-MR control chart): {self.drift.upper()}  "
                f"({self.n_signals} out-of-control signal"
                f"{'' if self.n_signals == 1 else 's'})",
            ]
        return lines

    def summary(self) -> dict:
        """Flat ``{label: value}`` dict of the headline numbers (dashboard-ready)."""
        lo = hi = None
        if self.tolerance is not None:
            lo, hi = self.tolerance
        return {
            "n": self.n,
            "residual_mean": self.mean,
            "residual_std": self.std,
            "rmse": self.rmse,
            "mae": self.mae,
            "tolerance_lo": lo,
            "tolerance_hi": hi,
            "pct_within": self.pct_within,
            "residual_cpk": self.residual_cpk,
            "residual_cpu": self.residual_cpu,
            "residual_cpl": self.residual_cpl,
            "drift": self.drift,
            "n_signals": self.n_signals,
        }

    # ---- visualization ---------------------------------------------------
    def _render_standalone(self, fig, kind, **kwargs):
        pal = _pal.active()
        r = np.asarray(self._resid, dtype=float)
        f = np.asarray(self._fitted, dtype=float)

        has_chart = self.chart is not None
        # Row 1: residual-vs-fitted scatter + residual histogram.
        # Row 2 (only with order): the residual I-MR control chart.
        nrows = 2 if has_chart else 1
        ax_sc = fig.add_subplot(nrows, 2, 1)
        ax_hist = fig.add_subplot(nrows, 2, 2)

        ax_sc.axhline(0.0, color=pal.target, ls="--", lw=1.5, zorder=1)
        ax_sc.scatter(f, r, s=14, color=pal.data, alpha=0.8, edgecolor=pal.bg,
                      linewidth=0.4, zorder=2)
        ax_sc.set_xlabel("fitted (prediction)")
        ax_sc.set_ylabel("residual")
        ax_sc.set_title("Residuals vs fitted")

        ax_hist.hist(r, bins="auto", density=True, color=pal.data,
                     edgecolor=pal.bg, alpha=0.9)
        if self.std > 0:
            xs = np.linspace(float(r.min()), float(r.max()), 200)
            ax_hist.plot(xs, stats.norm.pdf(xs, self.mean, self.std),
                         color=pal.center, lw=2, label="fitted normal")
        if self.tolerance is not None:
            lo, hi = self.tolerance
            for v in (lo, hi):
                ax_hist.axvline(v, color=pal.ooc, ls="--", lw=1.5)
        ax_hist.axvline(0.0, color=pal.target, ls=":", lw=1.2)
        ax_hist.set_xlabel("residual")
        ax_hist.set_ylabel("density")
        ax_hist.set_title("Residual distribution")

        if has_chart:
            ax_cc = fig.add_subplot(2, 1, 2)
            self.chart.view(ax=ax_cc)
            ax_cc.set_title("Residual control chart (I-MR, in order)")

    def _render_axes(self, ax, kind, **kwargs):
        pal = _pal.active()
        r = np.asarray(self._resid, dtype=float)
        f = np.asarray(self._fitted, dtype=float)
        ax.axhline(0.0, color=pal.target, ls="--", lw=1.5, zorder=1)
        ax.scatter(f, r, s=14, color=pal.data, alpha=0.8, edgecolor=pal.bg,
                   linewidth=0.4, zorder=2)
        ax.set_xlabel("fitted (prediction)")
        ax.set_ylabel("residual")
        ax.set_title("Residuals vs fitted")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def diagnose(y_true=None, y_pred=None, *, model=None, X=None, y=None,
             tolerance=None, order=None) -> ModelDiagnosticResult:
    """Run QC diagnostics on a model's residuals.

    Two calling conventions:

    * **Outputs-only:** ``diagnose(y_true, y_pred)`` - pass the ground truth and
      the predictions directly. This is the framework-agnostic core.
    * **Duck-typed model:** ``diagnose(model=m, X=X, y=y)`` - mfgQC calls
      ``m.predict(X)`` (and nothing else) to obtain ``y_pred``. It never imports
      or inspects the model's type; any object with a ``predict`` method works.

    Parameters
    ----------
    y_true, y_pred : array-like, optional
        Ground truth and predictions. Required unless ``model`` is given.
    model : object, optional
        Anything with a ``predict(X)`` method. DUCK-TYPED - no ML library is
        imported and the model's class is never inspected.
    X, y : array-like, optional
        Features and ground truth, used only on the ``model`` path.
    tolerance : (lo, hi) or float, optional
        Tolerance band for the residuals. A single number ``v`` means
        ``(-v, v)``. Enables Tier 2 (% within + residual capability).
    order : array-like, optional
        A sortable key (time, index, ...) used to re-sequence the residuals onto
        an I-MR control chart. Enables Tier 3 (drift verdict).

    Returns
    -------
    ModelDiagnosticResult
    """
    if model is not None:
        # The ONE thing mfgQC ever takes from a model: its predictions. Duck-typed.
        y_pred = np.asarray(model.predict(X), dtype=float)
        y_true = np.asarray(y, dtype=float)
    else:
        if y_true is None or y_pred is None:
            raise ValueError(
                "diagnose() requires either (y_true, y_pred) or model= with X/y.")
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got {y_true.shape} "
            f"and {y_pred.shape}.")

    resid = y_true - y_pred
    finite = ~(np.isnan(resid) | np.isnan(y_pred))
    r = resid[finite]
    fitted = y_pred[finite]
    n = int(r.size)
    if n == 0:
        raise ValueError("no finite residuals to diagnose.")

    mean = float(r.mean())
    std = float(np.std(r, ddof=1)) if n > 1 else 0.0
    rmse = float(np.sqrt(np.mean(r ** 2)))
    mae = float(np.mean(np.abs(r)))

    # ---- Tier 1: always ---------------------------------------------------
    checks: list[AssumptionCheck] = [
        _assume.check_normality(r),
        _check_homoscedasticity(r, fitted),
        _check_independence_dw(r),
        _check_zero_mean(r),
    ]

    # ---- Tier 2: tolerance -> % within + residual capability --------------
    tol = pct_within = res_cpk = res_cpu = res_cpl = None
    if tolerance is not None:
        lo, hi = _normalize_tolerance(tolerance)
        tol = (lo, hi)
        pct_within = float(100.0 * np.mean((r >= lo) & (r <= hi)))
        if std > 0:
            res_cpu = (hi - mean) / (3.0 * std)
            res_cpl = (mean - lo) / (3.0 * std)
            res_cpk = float(min(res_cpu, res_cpl))
            res_cpu, res_cpl = float(res_cpu), float(res_cpl)

    # ---- Tier 3: order -> residual control chart -> drift verdict ---------
    drift = "n/a"
    n_signals = None
    chart = None
    if order is not None:
        import pandas as pd
        order_arr = np.asarray(order)[finite]
        seq = np.argsort(order_arr, kind="stable")
        ordered = r[seq]
        qc = pd.DataFrame({"resid": ordered})
        chart = (
            __import__("mfgqc").load(qc, measure="resid").control_chart(kind="i_mr")
        )
        n_signals = len(chart.violations)
        drift = "degrading" if n_signals > 0 else "stable"

    analysis_step = Step(
        operation="model_diagnostic",
        params={
            "n": n, "rmse": rmse, "mae": mae,
            "tolerance": tol, "pct_within": pct_within, "residual_cpk": res_cpk,
            "drift": drift, "n_signals": n_signals,
        },
        n_affected=n,
        timestamp=_now(),
    )
    history = (analysis_step,) + tuple(_assumption_step(a) for a in checks)

    return ModelDiagnosticResult(
        n=n, mean=mean, std=std, rmse=rmse, mae=mae,
        tolerance=tol, pct_within=pct_within,
        residual_cpk=res_cpk, residual_cpu=res_cpu, residual_cpl=res_cpl,
        drift=drift, n_signals=n_signals, chart=chart,
        _resid=r, _fitted=fitted,
        assumptions=checks, history=history,
    )
