"""Regression / correlation and general (one-/two-way) ANOVA.

Three analyses, all surfacing their own assumptions (the wedge: report, never
silently switch):

- :func:`compute_regression` - OLS of the measure on one or more predictor
  columns via the normal equations (``numpy.linalg.lstsq``); reports coefficient
  inference (SE/t/p/CI), R^2/adj-R^2, the overall F-of-regression, and surfaces
  residual normality, homoscedasticity, and independence (Durbin-Watson).
- :func:`correlation` - correlation matrix + per-pair p-values (Pearson or
  Spearman).
- :func:`compute_anova` - one-way (1 factor) or two-way (2 factors + interaction)
  ANOVA on the measure; full SS/df/MS/F/p table plus eta^2 effect sizes, and
  surfaces residual normality + homogeneity of variance.

The OLS inference is pinned to ``scipy.stats.linregress`` for the simple case;
the ANOVA F/p is pinned to ``scipy.stats.f_oneway``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from . import assumptions as _assume
from . import palette as _pal
from ._result import QCResult
from .assumptions import AssumptionCheck
from .data import QCData, Step

_VALID_CORR_METHODS = ("pearson", "spearman")


# --------------------------------------------------------------------------- #
# History helpers (mirror capability.py / gage_rr.py exactly)
# --------------------------------------------------------------------------- #
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _assumption_step(a: AssumptionCheck) -> Step:
    return Step(
        operation=f"assumption:{a.name}",
        params={"test": a.test, "passed": a.passed, "magnitude": a.magnitude,
                "reliability": a.reliability, "p_value": a.p_value, "statistic": a.statistic},
        n_affected=None, timestamp=_now(),
    )


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class RegressionResult(QCResult):
    """Result of an OLS regression (immutable)."""

    terms: tuple[str, ...]              # incl. 'intercept' first
    coef: dict                          # name -> value
    se: dict                            # name -> std error
    t: dict                             # name -> t statistic
    p_value: dict                       # name -> coefficient p-value
    ci: dict                            # name -> (lo, hi) 95% CI
    r_squared: float
    adj_r_squared: float
    f_stat: float
    f_p_value: float
    resid_std_err: float
    df_resid: int
    n: int
    vif: dict = field(default_factory=dict)   # predictor -> variance inflation factor (multiple only)
    predictors: tuple[str, ...] = ()    # predictor column names (no intercept)
    response: str = "y"
    selection_path: tuple = ()          # steps of automated selection, if any
    selection_note: str | None = None
    _x: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _y: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _fitted: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _resid: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        kind = "simple" if len(self.predictors) == 1 else "multiple"
        return f"Regression ({kind} OLS): {self.response} ~ {' + '.join(self.predictors)}"

    def _summary_lines(self) -> list[str]:
        lines = [
            f"n = {self.n}   df(resid) = {self.df_resid}",
            f"R^2 = {self.r_squared:.4g}   adj R^2 = {self.adj_r_squared:.4g}   "
            f"resid std err = {self.resid_std_err:.4g}",
            f"F = {self.f_stat:.4g}   p = {self.f_p_value:.3g}",
            "",
            f"{'term':<16}{'coef':>12}{'std err':>12}{'t':>10}{'p':>10}"
            f"{'95% CI low':>14}{'95% CI high':>14}",
        ]
        for name in self.terms:
            lo, hi = self.ci[name]
            lines.append(
                f"{name:<16}{self.coef[name]:>12.5g}{self.se[name]:>12.4g}"
                f"{self.t[name]:>10.3g}{self.p_value[name]:>10.3g}"
                f"{lo:>14.5g}{hi:>14.5g}"
            )
        if self.vif:
            lines.append("")
            lines.append("VIF (multicollinearity): "
                         + ", ".join(f"{k}={v:.2f}" for k, v in self.vif.items()))
        if self.selection_path:
            lines.append("")
            lines.append("selection path: " + " ".join(self.selection_path))
            if self.selection_note:
                lines.append(f"NOTE: {self.selection_note}")
        return lines

    def summary(self) -> dict:
        """Flat {label: value} dict of the headline numbers (dashboard-ready)."""
        out: dict = {
            "response": self.response,
            "n": self.n,
            "df_resid": self.df_resid,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "f_stat": self.f_stat,
            "f_p_value": self.f_p_value,
            "resid_std_err": self.resid_std_err,
        }
        for name in self.terms:
            lo, hi = self.ci[name]
            out[f"coef[{name}]"] = self.coef[name]
            out[f"se[{name}]"] = self.se[name]
            out[f"t[{name}]"] = self.t[name]
            out[f"p[{name}]"] = self.p_value[name]
            out[f"ci_low[{name}]"] = lo
            out[f"ci_high[{name}]"] = hi
        for name, v in self.vif.items():
            out[f"vif[{name}]"] = v
        norm = next((a for a in self.assumptions if a.name == "normality"), None)
        out["residual_normality_passed"] = None if norm is None else norm.passed
        return out

    # ---- plotting --------------------------------------------------------
    def _render_standalone(self, fig, kind, **kwargs):
        simple = len(self.predictors) == 1
        # Simple regression: the fitted-line view. Multiple regression (or
        # kind='diagnostics'): a residual diagnostics panel (resid vs fitted, QQ,
        # scale-location) - there's no single predictor axis to plot against.
        if (simple and kind in (None, "fit")) or kind == "residuals":
            self._render_axes(fig.add_subplot(111), kind, **kwargs)
            return
        self._diagnostics_panel(fig)

    def _diagnostics_panel(self, fig):
        fitted, resid = self._fitted, self._resid
        axes = fig.subplots(1, 3)
        # 1. residuals vs fitted (structure / non-linearity)
        axes[0].scatter(fitted, resid, s=16, alpha=0.8, color=_pal.active().data)
        axes[0].axhline(0.0, color=_pal.active().ooc, lw=1, ls="--")
        axes[0].set_xlabel("fitted"); axes[0].set_ylabel("residual")
        axes[0].set_title("Residuals vs fitted", fontsize=9)
        # 2. normal QQ plot of residuals
        (osm, osr), (slope, inter, _r) = stats.probplot(resid, dist="norm")
        axes[1].scatter(osm, osr, s=16, alpha=0.8, color=_pal.active().data)
        axes[1].plot(osm, slope * osm + inter, color=_pal.active().center, lw=1.5)
        axes[1].set_xlabel("theoretical quantiles"); axes[1].set_ylabel("ordered residuals")
        axes[1].set_title("Normal QQ", fontsize=9)
        # 3. scale-location: sqrt|standardized residual| vs fitted (heteroscedasticity)
        s = resid.std(ddof=1) or 1.0
        axes[2].scatter(fitted, np.sqrt(np.abs(resid / s)), s=16, alpha=0.8, color=_pal.active().data)
        axes[2].set_xlabel("fitted"); axes[2].set_ylabel(r"$\sqrt{|std.\ resid|}$")
        axes[2].set_title("Scale-location", fontsize=9)
        fig.suptitle(self._title(), fontsize=10)

    def _render_axes(self, ax, kind, **kwargs):
        simple = len(self.predictors) == 1
        if kind is None:
            kind = "fit" if simple else "residuals"
        if kind == "fit" and simple:
            x = self._x[:, 1]
            order = np.argsort(x)
            xs = x[order]
            ax.scatter(x, self._y, s=20, alpha=0.8, label="data", color=_pal.active().data)
            ax.plot(xs, self._fitted[order], color=_pal.active().center, lw=2, label="fit")
            band = self._mean_response_band(xs)
            if band is not None:
                lo, hi = band
                ax.fill_between(xs, lo, hi, color=_pal.active().center, alpha=0.2,
                                label="95% mean CI")
            ax.set_xlabel(self.predictors[0])
            ax.set_ylabel(self.response)
            ax.set_title(self._title())
            ax.legend(loc="best", fontsize=8)
        elif kind in ("residuals", "fit"):
            ax.scatter(self._fitted, self._resid, s=20, alpha=0.8, color=_pal.active().data)
            ax.axhline(0.0, color=_pal.active().ooc, lw=1, ls="--")
            ax.set_xlabel("fitted values")
            ax.set_ylabel("residuals")
            ax.set_title("Residuals vs fitted")
        else:
            raise ValueError(f"unknown regression view kind={kind!r}; use None, 'fit', or 'residuals'.")
        return ax

    def _mean_response_band(self, xs_sorted: np.ndarray):
        """95% CI band for the mean response at the sorted predictor values
        (simple regression only)."""
        if len(self.predictors) != 1 or self.df_resid <= 0:
            return None
        X = self._x  # (n, 2) intercept + x
        try:
            xtx_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            return None
        s = self.resid_std_err
        tcrit = float(stats.t.ppf(0.975, self.df_resid))
        b0 = self.coef["intercept"]
        b1 = self.coef[self.predictors[0]]
        yhat = b0 + b1 * xs_sorted
        Xnew = np.column_stack([np.ones_like(xs_sorted), xs_sorted])
        leverage = np.einsum("ij,jk,ik->i", Xnew, xtx_inv, Xnew)
        half = tcrit * s * np.sqrt(np.clip(leverage, 0.0, None))
        return yhat - half, yhat + half


def _durbin_watson(resid: np.ndarray) -> float:
    d = np.diff(resid)
    denom = float(np.sum(resid * resid))
    if denom <= 0:
        return float("nan")
    return float(np.sum(d * d) / denom)


def _homoscedasticity_check(resid: np.ndarray, fitted: np.ndarray) -> AssumptionCheck:
    """Breusch-Pagan-style check: correlation of |resid| with fitted values.

    ``passed`` is the direct test that |resid| does not trend with the fitted
    value (Pearson correlation p >= alpha). The |r| is recorded as context.
    """
    n = resid.size
    abs_r = np.abs(resid)
    if n < 4 or np.std(fitted) == 0 or np.std(abs_r) == 0:
        return AssumptionCheck(
            "homoscedasticity", "corr(|resid|, fitted)", float("nan"), None,
            True, None, "abs-resid corr", "low_power", n, None)
    r, p = stats.pearsonr(abs_r, fitted)
    passed = float(p) >= _assume.ALPHA
    rel = _assume.reliability(n)
    rec = None
    if not passed:
        rec = (f"Residual spread trends with the fit (corr(|resid|, fitted) r={float(r):.2f}, "
               f"p={float(p):.3g}); heteroscedastic - consider a transform of the response "
               "or heteroscedasticity-robust (HC) standard errors.")
    return AssumptionCheck("homoscedasticity", "corr(|resid|, fitted)", float(r), float(p),
                           passed, float(abs(r)), "abs-resid corr", rel, n, rec)


def _independence_check(resid: np.ndarray) -> AssumptionCheck:
    """Durbin-Watson statistic. ``passed`` if roughly 1.5 < DW < 2.5 (DW is context)."""
    n = resid.size
    if n < 4:
        return AssumptionCheck(
            "independence", "Durbin-Watson", float("nan"), None,
            True, None, "Durbin-Watson", "low_power", n, None)
    dw = _durbin_watson(resid)
    passed = bool(np.isfinite(dw) and 1.5 < dw < 2.5)
    rel = _assume.reliability(n)
    rec = None
    if not passed:
        direction = "positive" if (np.isfinite(dw) and dw < 1.5) else "negative"
        rec = (f"Residuals show {direction} autocorrelation (Durbin-Watson DW={dw:.2f}, "
               "target ~2.0); inference assumes independent errors - consider a time-series "
               "model or check for an omitted ordering/trend term.")
    return AssumptionCheck("independence", "Durbin-Watson", dw, None,
                           passed, dw, "Durbin-Watson", rel, n, rec)


def _vif_values(predictor_cols: np.ndarray) -> list[float]:
    """Variance inflation factor for each predictor: ``1/(1 - R^2_j)`` where R^2_j
    is from regressing predictor j on the others. Quantifies multicollinearity."""
    n, k = predictor_cols.shape
    out = []
    for j in range(k):
        yj = predictor_cols[:, j]
        others = np.delete(predictor_cols, j, axis=1)
        A = np.column_stack([np.ones(n), others])
        b, _, _, _ = np.linalg.lstsq(A, yj, rcond=None)
        ss_tot = float(np.sum((yj - yj.mean()) ** 2))
        r2 = 1.0 - float(np.sum((yj - A @ b) ** 2)) / ss_tot if ss_tot > 0 else 0.0
        out.append(1.0 / (1.0 - r2) if r2 < 1.0 else float("inf"))
    return out


def _multicollinearity_check(vif: dict, n: int) -> AssumptionCheck:
    """``passed`` if every VIF < 5; max VIF is context. VIF>5 notable, >10 serious."""
    maxv = max(vif.values())
    worst = max(vif, key=vif.get)
    passed = bool(maxv < 5.0)
    rec = None
    if not passed:
        sev = "Severe" if maxv >= 10 else "Notable"
        rec = (f"{sev} multicollinearity (max VIF={maxv:.1f} for {worst!r}); collinear "
               "predictors inflate the coefficient standard errors - consider dropping or "
               "combining them. (VIF>5 notable, >10 serious.)")
    return AssumptionCheck("multicollinearity", "max VIF (<5)", float(maxv), None,
                           passed, float(maxv), "VIF", "ok", n, rec)


def compute_regression(qc: QCData, on) -> RegressionResult:
    """OLS of the measure (y) on one or more predictor columns, with an intercept.

    Parameters
    ----------
    qc : QCData
        The measure (``qc.meta.measure``) is the response.
    on : str or list of str
        Predictor column name (simple regression) or list of names (multiple).

    Returns
    -------
    RegressionResult
    """
    predictors = [on] if isinstance(on, str) else list(on)
    if not predictors:
        raise ValueError("compute_regression requires at least one predictor column in 'on'.")
    frame = qc.frame
    response = qc.meta.measure
    missing = [c for c in predictors if c not in frame.columns]
    if missing:
        raise ValueError(f"predictor column(s) {missing} not found in the frame "
                         f"(columns: {list(frame.columns)}).")
    if response in predictors:
        raise ValueError(f"the response {response!r} cannot also be a predictor.")

    cols = [response] + predictors
    sub = frame[cols].apply(pd.to_numeric, errors="coerce").dropna()
    y = sub[response].to_numpy(dtype=float)
    n = y.size
    k = len(predictors)            # predictors excluding intercept
    p_params = k + 1               # incl. intercept
    if n < p_params:
        raise ValueError(f"need at least as many observations as parameters: n={n}, parameters={p_params}.")
    # n == p_params is the exactly-determined (saturated) case: a unique fit with
    # zero residual df. The fit is valid; SE/t/p/F are NaN (no pure-error estimate),
    # which the DOE layer surfaces explicitly rather than fabricating an error term.

    X = np.column_stack([np.ones(n)] + [sub[c].to_numpy(dtype=float) for c in predictors])
    terms = ("intercept",) + tuple(predictors)

    beta, _resid_ss, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted

    ss_res = float(np.sum(resid * resid))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    df_resid = n - p_params
    df_model = k

    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj_r_squared = (1.0 - (1.0 - r_squared) * (n - 1) / df_resid
                     if (ss_tot > 0 and df_resid > 0) else float("nan"))

    mse = ss_res / df_resid if df_resid > 0 else float("nan")
    resid_std_err = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")

    # coefficient covariance = mse * (X'X)^-1
    xtx_inv = np.linalg.inv(X.T @ X)
    cov_beta = mse * xtx_inv
    se_vec = np.sqrt(np.clip(np.diag(cov_beta), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_vec = np.where(se_vec > 0, beta / se_vec, np.nan)
    p_vec = 2.0 * stats.t.sf(np.abs(t_vec), df_resid) if df_resid > 0 else np.full_like(beta, np.nan)
    tcrit = float(stats.t.ppf(0.975, df_resid)) if df_resid > 0 else float("nan")

    coef = {name: float(beta[i]) for i, name in enumerate(terms)}
    se = {name: float(se_vec[i]) for i, name in enumerate(terms)}
    t = {name: float(t_vec[i]) for i, name in enumerate(terms)}
    p_value = {name: float(p_vec[i]) for i, name in enumerate(terms)}
    ci = {name: (float(beta[i] - tcrit * se_vec[i]), float(beta[i] + tcrit * se_vec[i]))
          for i, name in enumerate(terms)}

    # Overall F-of-regression (ANOVA of regression).
    ss_model = ss_tot - ss_res
    if df_model > 0 and df_resid > 0 and mse > 0:
        f_stat = (ss_model / df_model) / mse
        f_p_value = float(stats.f.sf(f_stat, df_model, df_resid))
    else:
        f_stat = float("nan")
        f_p_value = float("nan")

    # Multicollinearity (multiple regression only): VIF per predictor.
    vif: dict = {}
    if k >= 2:
        vif = {name: float(v) for name, v in zip(predictors, _vif_values(X[:, 1:]))}

    # Assumption surfacing (the wedge).
    checks: list[AssumptionCheck] = [
        _assume.check_normality(resid, context="regression"),
        _homoscedasticity_check(resid, fitted),
        _independence_check(resid),
    ]
    if vif:
        checks.append(_multicollinearity_check(vif, n))

    step = Step(
        operation="regression",
        params={"on": predictors, "response": response,
                "r_squared": r_squared, "f_stat": f_stat, "f_p_value": f_p_value},
        n_affected=n, timestamp=_now(),
    )
    history = qc.history + (step,) + tuple(_assumption_step(a) for a in checks)

    return RegressionResult(
        terms=terms, coef=coef, se=se, t=t, p_value=p_value, ci=ci,
        r_squared=float(r_squared), adj_r_squared=float(adj_r_squared),
        f_stat=float(f_stat), f_p_value=float(f_p_value),
        resid_std_err=resid_std_err, df_resid=int(df_resid), n=int(n), vif=vif,
        predictors=tuple(predictors), response=response,
        _x=X, _y=y, _fitted=fitted, _resid=resid,
        assumptions=checks, history=history,
    )


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class CorrelationResult(QCResult):
    """Correlation matrix with per-pair p-values (immutable)."""

    method: str
    cols: tuple[str, ...]
    corr: dict                      # (a, b) -> r
    p_values: dict                  # (a, b) -> p
    n: int
    _matrix: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _pmatrix: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _pairs(self):
        return list(itertools.combinations(self.cols, 2))

    def _title(self) -> str:
        return f"Correlation ({self.method})"

    def _summary_lines(self) -> list[str]:
        lines = [f"n = {self.n}   variables: {', '.join(self.cols)}", "",
                 f"{'pair':<28}{'r':>10}{'p':>12}"]
        for a, b in self._pairs():
            r = self.corr[(a, b)]
            p = self.p_values[(a, b)]
            lines.append(f"{(a + ' ~ ' + b):<28}{r:>10.4g}{p:>12.3g}")
        return lines

    def summary(self) -> dict:
        """Flat dict of the notable (off-diagonal) pairs: r and p per pair."""
        out: dict = {"method": self.method, "n": self.n}
        for a, b in self._pairs():
            out[f"r[{a},{b}]"] = self.corr[(a, b)]
            out[f"p[{a},{b}]"] = self.p_values[(a, b)]
        return out

    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        self._render_axes(ax, kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        m = self._matrix
        im = ax.imshow(m, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_xticks(range(len(self.cols)))
        ax.set_yticks(range(len(self.cols)))
        ax.set_xticklabels(self.cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(self.cols, fontsize=8)
        for i in range(len(self.cols)):
            for j in range(len(self.cols)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
        ax.set_title(self._title())
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return ax


def correlation(df, cols=None, method: str = "pearson") -> CorrelationResult:
    """Correlation matrix + per-pair p-values for the given columns.

    Parameters
    ----------
    df : pandas.DataFrame or QCData
        Data; if a QCData is passed, its ``.frame`` is used.
    cols : list of str or None, optional
        Columns to correlate. Default: all numeric columns.
    method : str, optional
        ``"pearson"`` (default) or ``"spearman"``.

    Returns
    -------
    CorrelationResult
    """
    if method not in _VALID_CORR_METHODS:
        raise ValueError(f"method must be one of {_VALID_CORR_METHODS}; got {method!r}.")
    frame = df.frame if isinstance(df, QCData) else df
    if cols is None:
        cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    cols = list(cols)
    if len(cols) < 2:
        raise ValueError(f"correlation needs at least 2 numeric columns; got {cols}.")
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"column(s) {missing} not found in the frame.")

    sub = frame[cols].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(sub)
    if n < 3:
        raise ValueError(f"correlation needs at least 3 complete rows; got {n}.")
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr

    m = len(cols)
    rmat = np.eye(m)
    pmat = np.zeros((m, m))
    corr: dict = {}
    p_values: dict = {}
    for i in range(m):
        for j in range(m):
            if i == j:
                rmat[i, j] = 1.0
                pmat[i, j] = 0.0
                continue
            r, p = fn(sub[cols[i]].to_numpy(dtype=float), sub[cols[j]].to_numpy(dtype=float))
            rmat[i, j] = float(r)
            pmat[i, j] = float(p)
            if i < j:
                corr[(cols[i], cols[j])] = float(r)
                p_values[(cols[i], cols[j])] = float(p)

    step = Step(operation="correlation",
                params={"method": method, "cols": cols},
                n_affected=n, timestamp=_now())
    base_hist = df.history if isinstance(df, QCData) else ()
    history = base_hist + (step,)

    return CorrelationResult(
        method=method, cols=tuple(cols), corr=corr, p_values=p_values, n=int(n),
        _matrix=rmat, _pmatrix=pmat, assumptions=[], history=history,
    )


# --------------------------------------------------------------------------- #
# General ANOVA (one-way / two-way)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class AnovaResult(QCResult):
    """Result of a one-way or two-way ANOVA (immutable)."""

    table: dict                     # term -> {ss, df, ms, f, p_value, eta_sq}
    n: int
    factors: tuple[str, ...]
    response: str = "y"
    _group_means: dict = field(repr=False, default_factory=dict)
    _cell_means: dict = field(repr=False, default_factory=dict)
    _levels: dict = field(repr=False, default_factory=dict)
    _groups: dict = field(repr=False, default_factory=dict)
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def posthoc(self, method=None, control=None):
        """Routed pairwise multiple comparisons (one-way only). See
        :func:`mfgqc.posthoc.compute`."""
        if len(self.factors) != 1 or not self._groups:
            raise ValueError("posthoc is defined for a one-way ANOVA with retained groups.")
        from .posthoc import compute
        labels = list(self._levels[self.factors[0]])
        groups = [self._groups[lvl] for lvl in labels]
        return compute(groups, labels, self.assumptions, "anova",
                       method=method, control=control, base_history=self.history)

    @property
    def _term_order(self):
        order = list(self.factors)
        if len(self.factors) == 2:
            order.append(":".join(self.factors))
        order += ["residual", "total"]
        return [t for t in order if t in self.table]

    def _title(self) -> str:
        kind = "one-way" if len(self.factors) == 1 else "two-way"
        return f"ANOVA ({kind}): {self.response} ~ {' * '.join(self.factors)}"

    def _summary_lines(self) -> list[str]:
        lines = [f"n = {self.n}",
                 "",
                 f"{'source':<20}{'SS':>12}{'df':>6}{'MS':>12}{'F':>10}{'p':>10}{'eta^2':>10}"]
        for term in self._term_order:
            row = self.table[term]
            f_s = "" if row.get("f") is None else f"{row['f']:.4g}"
            p_s = "" if row.get("p_value") is None else f"{row['p_value']:.3g}"
            ms_s = "" if row.get("ms") is None else f"{row['ms']:.5g}"
            eta_s = "" if row.get("eta_sq") is None else f"{row['eta_sq']:.3g}"
            lines.append(
                f"{term:<20}{row['ss']:>12.5g}{row['df']:>6}{ms_s:>12}{f_s:>10}{p_s:>10}{eta_s:>10}"
            )
        return lines

    def summary(self) -> dict:
        """Flat dict of F/p/eta^2 per factor (and interaction for two-way)."""
        out: dict = {"response": self.response, "n": self.n}
        for term, row in self.table.items():
            if term in ("residual", "total"):
                continue
            out[f"F[{term}]"] = row.get("f")
            out[f"p[{term}]"] = row.get("p_value")
            out[f"eta_sq[{term}]"] = row.get("eta_sq")
        return out

    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        self._render_axes(ax, kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        if len(self.factors) == 1:
            f0 = self.factors[0]
            levels = self._levels[f0]
            means = [self._group_means[lvl] for lvl in levels]
            ax.bar([str(lvl) for lvl in levels], means, color="C0", alpha=0.85)
            ax.set_xlabel(f0)
            ax.set_ylabel(f"mean {self.response}")
            ax.set_title(self._title())
        else:
            f0, f1 = self.factors
            l0 = self._levels[f0]
            l1 = self._levels[f1]
            for lvl1 in l1:
                ys = [self._cell_means.get((lvl0, lvl1), np.nan) for lvl0 in l0]
                ax.plot([str(x) for x in l0], ys, marker="o", label=f"{f1}={lvl1}")
            ax.set_xlabel(f0)
            ax.set_ylabel(f"mean {self.response}")
            ax.set_title(self._title())
            ax.legend(loc="best", fontsize=8, title=f1)
        return ax


def compute_anova(qc: QCData, factors, interaction: bool = True) -> AnovaResult:
    """One-way (1 factor) or two-way (2 factors + interaction) ANOVA on the measure.

    Parameters
    ----------
    qc : QCData
    factors : list of str
        1 or 2 categorical column names in ``qc.frame``.
    interaction : bool, optional
        Two-way only. When True (default) the full model fits the interaction
        term. When False the additive model pools the interaction SS into the
        error term and the residuals are taken against the additive fit
        (row mean + column mean - grand mean). Ignored for one-way.

    Returns
    -------
    AnovaResult
    """
    factors = [factors] if isinstance(factors, str) else list(factors)
    if len(factors) not in (1, 2):
        raise ValueError(f"compute_anova supports 1 or 2 factors; got {len(factors)}.")
    frame = qc.frame
    response = qc.meta.measure
    missing = [c for c in factors if c not in frame.columns]
    if missing:
        raise ValueError(f"factor column(s) {missing} not found in the frame "
                         f"(columns: {list(frame.columns)}).")

    cols = [response] + factors
    sub = frame[cols].copy()
    sub[response] = pd.to_numeric(sub[response], errors="coerce")
    sub = sub.dropna(subset=cols)
    n = len(sub)
    if n < len(factors) + 2:
        raise ValueError(f"not enough complete observations for ANOVA: n={n}.")

    y = sub[response].to_numpy(dtype=float)
    grand = float(np.mean(y))
    ss_total = float(np.sum((y - grand) ** 2))
    df_total = n - 1

    # Track level orderings (first-appearance) for the views.
    levels = {f: list(pd.unique(sub[f])) for f in factors}

    if len(factors) == 1:
        f0 = factors[0]
        group_means = {}
        ss_factor = 0.0
        for lvl, idx in sub.groupby(f0, sort=False).groups.items():
            g = sub.loc[idx, response].to_numpy(dtype=float)
            m = float(np.mean(g))
            group_means[lvl] = m
            ss_factor += g.size * (m - grand) ** 2
        ss_factor = float(ss_factor)
        df_factor = len(levels[f0]) - 1
        ss_error = ss_total - ss_factor
        df_error = n - len(levels[f0])

        table = _one_way_table(f0, ss_factor, df_factor, ss_error, df_error,
                               ss_total, df_total)

        # residuals = y - its group mean
        resid = np.array([yi - group_means[lvl]
                          for yi, lvl in zip(y, sub[f0].to_numpy())])
        groups = [sub.loc[idx, response].to_numpy(dtype=float)
                  for _lvl, idx in sub.groupby(f0, sort=False).groups.items()]
        groups_by_level = {lvl: sub.loc[idx, response].to_numpy(dtype=float)
                           for lvl, idx in sub.groupby(f0, sort=False).groups.items()}
        cell_means: dict = {}
    else:
        f0, f1 = factors
        a = len(levels[f0])
        b = len(levels[f1])

        # Factor main effects (Type I/balanced-style on marginal means).
        def _marginal_ss(col):
            ss = 0.0
            for _lvl, idx in sub.groupby(col, sort=False).groups.items():
                g = sub.loc[idx, response].to_numpy(dtype=float)
                ss += g.size * (float(np.mean(g)) - grand) ** 2
            return float(ss)

        ss_a = _marginal_ss(f0)
        ss_b = _marginal_ss(f1)

        # Cells (combined a x b) SS, then interaction = cells - a - b.
        ss_cells = 0.0
        cell_means = {}
        for (lvl0, lvl1), idx in sub.groupby([f0, f1], sort=False).groups.items():
            g = sub.loc[idx, response].to_numpy(dtype=float)
            cm = float(np.mean(g))
            cell_means[(lvl0, lvl1)] = cm
            ss_cells += g.size * (cm - grand) ** 2
        ss_cells = float(ss_cells)
        ss_ab = ss_cells - ss_a - ss_b
        df_a = a - 1
        df_b = b - 1
        df_ab = (a - 1) * (b - 1)

        # marginal means for the additive fit / residuals
        row_means = {lvl: float(np.mean(sub.loc[idx, response].to_numpy(dtype=float)))
                     for lvl, idx in sub.groupby(f0, sort=False).groups.items()}
        col_means = {lvl: float(np.mean(sub.loc[idx, response].to_numpy(dtype=float)))
                     for lvl, idx in sub.groupby(f1, sort=False).groups.items()}

        if interaction:
            ss_error = ss_total - ss_cells
            df_error = n - a * b
            table = _two_way_table(f0, f1, ss_a, df_a, ss_b, df_b, ss_ab, df_ab,
                                   ss_error, df_error, ss_total, df_total)
            # residuals = y - its cell mean
            resid = np.array([yi - cell_means[(lvl0, lvl1)]
                              for yi, lvl0, lvl1 in
                              zip(y, sub[f0].to_numpy(), sub[f1].to_numpy())])
        else:
            # additive model: interaction SS pooled into error
            ss_error = (ss_total - ss_cells) + ss_ab
            df_error = (n - a * b) + df_ab
            table = _two_way_additive_table(f0, f1, ss_a, df_a, ss_b, df_b,
                                            ss_error, df_error, ss_total, df_total)
            # residuals against the additive fit (row + col - grand)
            resid = np.array([yi - (row_means[lvl0] + col_means[lvl1] - grand)
                              for yi, lvl0, lvl1 in
                              zip(y, sub[f0].to_numpy(), sub[f1].to_numpy())])
        group_means = row_means
        groups_by_level = {}        # posthoc is one-way only
        # homogeneity across factor-1 levels
        groups = [sub.loc[idx, response].to_numpy(dtype=float)
                  for _lvl, idx in sub.groupby(f0, sort=False).groups.items()]

    # Assumption surfacing.
    checks: list[AssumptionCheck] = [
        _assume.check_normality(resid),
        _assume.check_homogeneity(groups),
    ]
    # Sharpen recommendations toward Welch/Kruskal (don't auto-switch).
    checks = [_anova_recommendation(a) for a in checks]

    step = Step(
        operation="anova",
        params={"factors": factors, "response": response,
                "terms": {t: {"f": r.get("f"), "p_value": r.get("p_value"),
                              "eta_sq": r.get("eta_sq")}
                          for t, r in table.items() if t not in ("residual", "total")}},
        n_affected=n, timestamp=_now(),
    )
    history = qc.history + (step,) + tuple(_assumption_step(a) for a in checks)

    return AnovaResult(
        table=table, n=int(n), factors=tuple(factors), response=response,
        _group_means=group_means, _cell_means=cell_means, _levels=levels,
        _groups=groups_by_level,
        assumptions=checks, history=history,
    )


def _f_and_p(ss, df, ms_error, df_error):
    if df <= 0 or df_error <= 0 or ms_error <= 0:
        return None, None
    ms = ss / df
    f = ms / ms_error
    p = float(stats.f.sf(f, df, df_error))
    return float(f), p


def _row(ss, df, ss_total, *, ms_error=None, df_error=None, is_term=True):
    ms = (ss / df) if df > 0 else None
    f = p = None
    if is_term and ms_error is not None:
        f, p = _f_and_p(ss, df, ms_error, df_error)
    eta = (ss / ss_total) if (is_term and ss_total > 0) else None
    return {"ss": float(ss), "df": int(df),
            "ms": None if ms is None else float(ms),
            "f": f, "p_value": p, "eta_sq": eta}


def _one_way_table(factor, ss_f, df_f, ss_e, df_e, ss_t, df_t) -> dict:
    ms_e = (ss_e / df_e) if df_e > 0 else None
    table = {
        factor: _row(ss_f, df_f, ss_t, ms_error=ms_e, df_error=df_e),
        "residual": _row(ss_e, df_e, ss_t, is_term=False),
        "total": _row(ss_t, df_t, ss_t, is_term=False),
    }
    return table


def _two_way_table(f0, f1, ss_a, df_a, ss_b, df_b, ss_ab, df_ab,
                   ss_e, df_e, ss_t, df_t) -> dict:
    ms_e = (ss_e / df_e) if df_e > 0 else None
    inter = f"{f0}:{f1}"
    table = {
        f0: _row(ss_a, df_a, ss_t, ms_error=ms_e, df_error=df_e),
        f1: _row(ss_b, df_b, ss_t, ms_error=ms_e, df_error=df_e),
        inter: _row(ss_ab, df_ab, ss_t, ms_error=ms_e, df_error=df_e),
        "residual": _row(ss_e, df_e, ss_t, is_term=False),
        "total": _row(ss_t, df_t, ss_t, is_term=False),
    }
    return table


def _two_way_additive_table(f0, f1, ss_a, df_a, ss_b, df_b,
                            ss_e, df_e, ss_t, df_t) -> dict:
    """Two-way ANOVA table for the additive (no-interaction) model: both main
    effects tested against the pooled error (interaction SS folded into error)."""
    ms_e = (ss_e / df_e) if df_e > 0 else None
    table = {
        f0: _row(ss_a, df_a, ss_t, ms_error=ms_e, df_error=df_e),
        f1: _row(ss_b, df_b, ss_t, ms_error=ms_e, df_error=df_e),
        "residual": _row(ss_e, df_e, ss_t, is_term=False),
        "total": _row(ss_t, df_t, ss_t, is_term=False),
    }
    return table


def _anova_recommendation(a: AssumptionCheck) -> AssumptionCheck:
    """Sharpen the (already populated) recommendation toward ANOVA-specific advice."""
    if a.passed:
        return a
    from dataclasses import replace
    if a.name == "homogeneity_of_variance":
        rec = (f"Group variances differ (variance ratio {a.magnitude:.3g}); for unequal "
               "variances use Welch's ANOVA - do not auto-switch, decide in context.")
        return replace(a, recommendation=rec)
    if a.name == "normality":
        rec = (f"Residuals are not normal (AD={a.statistic:.3g}); consider a transform or a "
               "non-parametric Kruskal-Wallis test - recommendation, not a forced switch.")
        return replace(a, recommendation=rec)
    return a
