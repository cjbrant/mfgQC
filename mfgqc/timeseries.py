"""QC-scoped time-series characterization (Track 1C).

These tools answer "is my process stable / trending / cyclic over time?" - they
CHARACTERIZE and DETECT time structure. They are deliberately NOT forecasters:
there is no ARIMA/SARIMA/ML model here and nothing predicts future values.

Three analyses, each a module-level ``compute_*`` returning an immutable result:

- :func:`compute_trend` - regress the measure on time (the ``time`` role column
  if present, else the row index) via the existing OLS machinery; report the
  slope, its t/p, R^2, a stable/drifting verdict, and surface regression's own
  residual-assumption checks.
- :func:`compute_acf` - the sample autocorrelation function ACF(k) and partial
  autocorrelation PACF(k) (Durbin-Levinson recursion) with a +/-1.96/sqrt(n)
  confidence band; report which lags break the band.
- :func:`compute_decompose` - classical ADDITIVE decomposition (centered moving
  average trend, phase-averaged seasonal, observed-trend-seasonal residual).

numpy/scipy only (statsmodels is not a dependency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from ._result import QCResult
from . import palette as _pal
from .assumptions import AssumptionCheck
from .data import QCData, Step
from .regression import compute_regression


# --------------------------------------------------------------------------- #
# History helpers (mirror regression.py)
# --------------------------------------------------------------------------- #
def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Trend
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class TrendResult(QCResult):
    """Result of a trend (drift) characterization over time (immutable).

    The measure is regressed on a time index; a significant slope means the
    process is DRIFTING, not merely noisy. This detects time structure - it does
    not forecast the next value.
    """

    slope: float
    intercept: float
    t: float
    p_value: float
    r_squared: float
    verdict: str                # "drifting" | "stable"
    time_col: str               # name of the time axis ("index" when synthesized)
    n: int
    _time: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _y: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _fitted: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    response: str = "y"
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"Trend: {self.response} vs {self.time_col}"

    def _summary_lines(self) -> list[str]:
        return [
            f"n = {self.n}   time axis: {self.time_col}",
            f"slope = {self.slope:.5g}   intercept = {self.intercept:.5g}",
            f"t = {self.t:.4g}   p = {self.p_value:.3g}   R^2 = {self.r_squared:.4g}",
            f"verdict: {self.verdict.upper()}",
        ]

    def summary(self) -> dict:
        """Flat {label: value} dict of the headline numbers (dashboard-ready)."""
        return {
            "response": self.response,
            "time_col": self.time_col,
            "n": self.n,
            "slope": self.slope,
            "intercept": self.intercept,
            "t": self.t,
            "p_value": self.p_value,
            "r_squared": self.r_squared,
            "verdict": self.verdict,
        }

    # ---- plotting --------------------------------------------------------
    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        self._render_axes(ax, kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        pal = _pal.active()
        t = self._time
        order = np.argsort(t)
        ts = t[order]
        ax.plot(ts, self._y[order], marker="o", ms=3, lw=1, color=pal.data,
                alpha=0.85, label="observed")
        ax.plot(ts, self._fitted[order], lw=2, color=pal.center,
                label=f"trend (slope={self.slope:.3g})")
        ax.set_xlabel(self.time_col)
        ax.set_ylabel(self.response)
        ax.set_title(self._title())
        ax.legend(loc="best", fontsize=8)
        return ax


def compute_trend(qc: QCData) -> TrendResult:
    """Characterize drift: regress the measure on time and verdict stable/drifting.

    Time is the ``time`` role column if one is bound (``qc.meta.roles['time']``),
    otherwise a synthesized row index ``0..n-1``. A slope significant at 0.05 is
    reported as ``"drifting"``; otherwise ``"stable"``. The regression's own
    residual-assumption checks (normality / homoscedasticity / independence) are
    surfaced unchanged.

    Parameters
    ----------
    qc : QCData

    Returns
    -------
    TrendResult
    """
    response = qc.meta.measure
    time_role = qc.meta.roles.get("time")

    if time_role is not None:
        reg = compute_regression(qc, time_role)
        time_col = time_role
        t_axis = np.asarray(reg._x[:, 1], dtype=float)
    else:
        # Synthesize a 0..n-1 index column and regress on it.
        frame = qc.frame
        idx_name = "_t_index"
        while idx_name in frame.columns:
            idx_name += "_"
        frame[idx_name] = np.arange(len(frame), dtype=float)
        qc_idx = QCData(_frame=frame, meta=qc.meta, history=qc.history)
        reg = compute_regression(qc_idx, idx_name)
        time_col = "index"
        t_axis = np.asarray(reg._x[:, 1], dtype=float)

    pred = reg.predictors[0]
    slope = float(reg.coef[pred])
    intercept = float(reg.coef["intercept"])
    t_stat = float(reg.t[pred])
    p_value = float(reg.p_value[pred])
    r_squared = float(reg.r_squared)
    verdict = "drifting" if (np.isfinite(p_value) and p_value < 0.05) else "stable"

    step = Step(
        operation="trend",
        params={"response": response, "time_col": time_col,
                "slope": slope, "p_value": p_value, "verdict": verdict},
        n_affected=reg.n, timestamp=_now(),
    )
    history = qc.history + (step,)

    return TrendResult(
        slope=slope, intercept=intercept, t=t_stat, p_value=p_value,
        r_squared=r_squared, verdict=verdict, time_col=time_col, n=int(reg.n),
        _time=t_axis, _y=np.asarray(reg._y, dtype=float),
        _fitted=np.asarray(reg._fitted, dtype=float),
        response=response, assumptions=list(reg.assumptions), history=history,
    )


# --------------------------------------------------------------------------- #
# Autocorrelation (ACF / PACF)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class AutocorrelationResult(QCResult):
    """Sample ACF and PACF with a confidence band (immutable).

    Significant autocorrelation means consecutive measurements are NOT
    independent - the process carries memory. This detects that structure; it
    does not fit a model to it.
    """

    lags: np.ndarray            # 1..L
    acf: np.ndarray             # ACF(k) for k in lags
    pacf: np.ndarray            # PACF(k) for k in lags
    conf: float                 # +/- band half-width (1.96/sqrt(n))
    significant_lags: tuple[int, ...]
    n: int
    response: str = "y"
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"Autocorrelation: {self.response}"

    def _summary_lines(self) -> list[str]:
        lines = [
            f"n = {self.n}   max lag = {int(self.lags[-1]) if self.lags.size else 0}",
            f"95% band = +/-{self.conf:.4g}",
            f"significant lags: "
            f"{', '.join(str(k) for k in self.significant_lags) or '(none)'}",
            "",
            f"{'lag':>5}{'ACF':>12}{'PACF':>12}",
        ]
        for k, a, p in zip(self.lags, self.acf, self.pacf):
            mark = " *" if int(k) in self.significant_lags else ""
            lines.append(f"{int(k):>5}{a:>12.4g}{p:>12.4g}{mark}")
        return lines

    def summary(self) -> dict:
        """Flat dict: band, lag-1 ACF/PACF, and significant-lag count."""
        return {
            "response": self.response,
            "n": self.n,
            "conf": self.conf,
            "acf_lag1": float(self.acf[0]) if self.acf.size else float("nan"),
            "pacf_lag1": float(self.pacf[0]) if self.pacf.size else float("nan"),
            "n_significant": len(self.significant_lags),
        }

    # ---- plotting --------------------------------------------------------
    def _render_standalone(self, fig, kind, **kwargs):
        ax_acf = fig.add_subplot(211)
        ax_pacf = fig.add_subplot(212)
        self._stem(ax_acf, self.acf, "ACF")
        self._stem(ax_pacf, self.pacf, "PACF")
        ax_pacf.set_xlabel("lag")

    def _render_axes(self, ax, kind, **kwargs):
        which = (kind or "acf").lower()
        if which == "acf":
            self._stem(ax, self.acf, "ACF")
        elif which == "pacf":
            self._stem(ax, self.pacf, "PACF")
        else:
            raise ValueError(
                f"unknown autocorrelation view kind={kind!r}; use None, 'acf', or 'pacf'.")
        ax.set_xlabel("lag")
        return ax

    def _stem(self, ax, vals, label):
        pal = _pal.active()
        lags = self.lags
        ax.axhline(0.0, color=pal.axis, lw=1)
        markerline, stemlines, baseline = ax.stem(lags, vals)
        markerline.set_color(pal.center)
        markerline.set_markersize(4)
        stemlines.set_color(pal.data)
        baseline.set_visible(False)
        ax.axhline(self.conf, color=pal.limit, lw=1, ls="--")
        ax.axhline(-self.conf, color=pal.limit, lw=1, ls="--")
        ax.set_ylabel(label)
        ax.set_title(f"{label}: {self.response}")
        return ax


def _acf(x_centered: np.ndarray, lags: int) -> np.ndarray:
    """Sample ACF(k) for k=1..lags: sum(x_t*x_{t+k}) / sum(x_t^2)."""
    denom = float(np.sum(x_centered * x_centered))
    out = np.empty(lags, dtype=float)
    for k in range(1, lags + 1):
        if denom <= 0:
            out[k - 1] = 0.0
        else:
            out[k - 1] = float(np.sum(x_centered[:-k] * x_centered[k:]) / denom)
    return out


def _pacf_durbin_levinson(acf_full: np.ndarray, lags: int) -> np.ndarray:
    """PACF(k) for k=1..lags via the Durbin-Levinson recursion.

    ``acf_full`` is [r0=1, r1, r2, ...] (ACF including lag 0). Returns the
    partial autocorrelations phi_{k,k}.
    """
    pacf = np.zeros(lags, dtype=float)
    phi = np.zeros((lags + 1, lags + 1), dtype=float)
    if lags >= 1:
        phi[1, 1] = acf_full[1]
        pacf[0] = phi[1, 1]
    for k in range(2, lags + 1):
        num = acf_full[k] - sum(phi[k - 1, j] * acf_full[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * acf_full[j] for j in range(1, k))
        phi_kk = num / den if den != 0 else 0.0
        phi[k, k] = phi_kk
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi_kk * phi[k - 1, k - j]
        pacf[k - 1] = phi_kk
    return pacf


def compute_acf(qc: QCData, lags: int = 20) -> AutocorrelationResult:
    """Sample ACF and PACF of the measure, with a +/-1.96/sqrt(n) band.

    ACF(k) = sum(x_t * x_{t+k}) / sum(x_t^2) on the mean-centered series; PACF via
    the Durbin-Levinson recursion. A lag whose |ACF| exceeds the band is flagged
    as significant - evidence the process is not independent over time.

    Parameters
    ----------
    qc : QCData
    lags : int, optional
        Maximum lag (default 20). Clamped to ``n - 1``.

    Returns
    -------
    AutocorrelationResult
    """
    if lags < 1:
        raise ValueError(f"lags must be >= 1; got {lags}.")
    y = qc.values()
    y = y[np.isfinite(y)]
    n = y.size
    if n < 3:
        raise ValueError(f"autocorrelation needs at least 3 observations; got {n}.")
    max_lag = min(lags, n - 1)

    xc = y - float(np.mean(y))
    acf_vals = _acf(xc, max_lag)
    acf_full = np.concatenate(([1.0], acf_vals))
    pacf_vals = _pacf_durbin_levinson(acf_full, max_lag)

    conf = 1.96 / np.sqrt(n)
    lag_idx = np.arange(1, max_lag + 1)
    sig = tuple(int(k) for k, a in zip(lag_idx, acf_vals) if abs(a) > conf)

    step = Step(
        operation="autocorrelation",
        params={"response": qc.meta.measure, "lags": max_lag,
                "conf": float(conf), "n_significant": len(sig)},
        n_affected=n, timestamp=_now(),
    )
    history = qc.history + (step,)

    return AutocorrelationResult(
        lags=lag_idx, acf=acf_vals, pacf=pacf_vals, conf=float(conf),
        significant_lags=sig, n=int(n), response=qc.meta.measure,
        assumptions=[], history=history,
    )


# --------------------------------------------------------------------------- #
# Classical additive decomposition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class DecompositionResult(QCResult):
    """Classical additive decomposition observed = trend + seasonal + residual.

    Separates a slow trend and a repeating seasonal cycle from the residual
    noise so each can be judged on its own. This characterizes the cyclic
    structure - it does not extrapolate it forward.
    """

    period: int
    observed: np.ndarray
    trend: np.ndarray           # NaN where the centered MA is undefined
    seasonal: np.ndarray
    resid: np.ndarray           # NaN where trend is undefined
    seasonal_amplitude: float   # peak-to-trough of one seasonal cycle
    resid_std: float
    n: int
    response: str = "y"
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"Decomposition (additive, period={self.period}): {self.response}"

    def _summary_lines(self) -> list[str]:
        return [
            f"n = {self.n}   period = {self.period}",
            f"seasonal amplitude (peak-to-trough) = {self.seasonal_amplitude:.4g}",
            f"residual std = {self.resid_std:.4g}",
        ]

    def summary(self) -> dict:
        """Flat dict: period, seasonal amplitude, residual std."""
        return {
            "response": self.response,
            "n": self.n,
            "period": self.period,
            "seasonal_amplitude": self.seasonal_amplitude,
            "resid_std": self.resid_std,
        }

    # ---- plotting --------------------------------------------------------
    def _render_standalone(self, fig, kind, **kwargs):
        pal = _pal.active()
        x = np.arange(self.n)
        panels = [
            ("observed", self.observed, pal.data),
            ("trend", self.trend, pal.center),
            ("seasonal", self.seasonal, pal.target),
            ("residual", self.resid, pal.limit),
        ]
        axes = fig.subplots(4, 1, sharex=True)
        for ax, (label, series, color) in zip(np.atleast_1d(axes), panels):
            ax.plot(x, series, lw=1, color=color)
            ax.set_ylabel(label, fontsize=8)
            if label in ("seasonal", "residual"):
                ax.axhline(0.0, color=pal.axis, lw=1)
        np.atleast_1d(axes)[0].set_title(self._title())
        np.atleast_1d(axes)[-1].set_xlabel("index")

    def _render_axes(self, ax, kind, **kwargs):
        pal = _pal.active()
        x = np.arange(self.n)
        ax.plot(x, self.observed, lw=1, color=pal.data, label="observed")
        ax.plot(x, self.trend, lw=2, color=pal.center, label="trend")
        ax.set_xlabel("index")
        ax.set_ylabel(self.response)
        ax.set_title(self._title())
        ax.legend(loc="best", fontsize=8)
        return ax


def _centered_moving_average(y: np.ndarray, period: int) -> np.ndarray:
    """Centered moving average of window ``period``; NaN-padded at the ends.

    For an even period the standard 2xperiod centered average (half-weighted
    endpoints) is used so the window is centered on an integer index.
    """
    n = y.size
    trend = np.full(n, np.nan, dtype=float)
    half = period // 2
    if period % 2 == 1:
        for i in range(half, n - half):
            trend[i] = float(np.mean(y[i - half:i + half + 1]))
    else:
        # 2xm moving average: weights 0.5 at the two ends, 1.0 in between.
        w = np.ones(period + 1, dtype=float)
        w[0] = w[-1] = 0.5
        w = w / w.sum()
        for i in range(half, n - half):
            trend[i] = float(np.sum(y[i - half:i + half + 1] * w))
    return trend


def compute_decompose(qc: QCData, period: int) -> DecompositionResult:
    """Classical ADDITIVE decomposition into trend + seasonal + residual.

    trend = centered moving average of window ``period`` (NaN at the ends where
    undefined); detrended = observed - trend; seasonal = average of the detrended
    series within each phase 0..period-1, tiled and centered to sum ~0 over a
    period; residual = observed - trend - seasonal.

    Parameters
    ----------
    qc : QCData
    period : int
        Seasonal period (>= 2).

    Returns
    -------
    DecompositionResult
    """
    if period < 2:
        raise ValueError(f"period must be >= 2; got {period}.")
    y = qc.values()
    n = y.size
    if n < 2 * period:
        raise ValueError(
            f"decomposition needs at least 2 full periods (2*{period}={2 * period}); "
            f"got n={n}.")

    observed = y.astype(float).copy()
    trend = _centered_moving_average(observed, period)
    detrended = observed - trend

    # Phase means of the detrended series (ignoring NaN), then center to sum ~0.
    phase = np.arange(n) % period
    phase_means = np.full(period, np.nan, dtype=float)
    for ph in range(period):
        vals = detrended[(phase == ph) & np.isfinite(detrended)]
        phase_means[ph] = float(np.mean(vals)) if vals.size else 0.0
    phase_means = phase_means - float(np.mean(phase_means))  # center to sum ~0

    seasonal = phase_means[phase]
    resid = observed - trend - seasonal  # NaN where trend is NaN

    seasonal_amplitude = float(np.max(phase_means) - np.min(phase_means))
    finite_resid = resid[np.isfinite(resid)]
    resid_std = float(np.std(finite_resid)) if finite_resid.size else float("nan")

    step = Step(
        operation="decompose",
        params={"response": qc.meta.measure, "period": period,
                "seasonal_amplitude": seasonal_amplitude, "resid_std": resid_std},
        n_affected=n, timestamp=_now(),
    )
    history = qc.history + (step,)

    return DecompositionResult(
        period=int(period), observed=observed, trend=trend, seasonal=seasonal,
        resid=resid, seasonal_amplitude=seasonal_amplitude, resid_std=resid_std,
        n=int(n), response=qc.meta.measure, assumptions=[], history=history,
    )


# --------------------------------------------------------------------------- #
# Combined time-series screen: trend (linear + Mann-Kendall) + autocorrelation
# --------------------------------------------------------------------------- #
def _mann_kendall(y: np.ndarray):
    """Mann-Kendall trend test: returns (S, z, p, tau)."""
    from scipy import stats as _st
    n = y.size
    s = 0.0
    for i in range(n - 1):
        s += float(np.sum(np.sign(y[i + 1:] - y[i])))
    _, counts = np.unique(y, return_counts=True)
    tie = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    var_s = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    if var_s <= 0:
        return s, 0.0, 1.0, 0.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = float(2 * _st.norm.sf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    return float(s), float(z), p, float(tau)


@dataclass(frozen=True, repr=False)
class TimeSeriesResult(QCResult):
    """Combined time-series screen (immutable): trend + autocorrelation flags."""

    slope: float
    slope_p: float
    mk_tau: float
    mk_p: float
    mk_z: float
    direction: str
    acf_lags: np.ndarray
    acf_values: np.ndarray
    acf_bound: float
    sig_lags: tuple
    n: int
    _y: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return "Time-series screen: trend & autocorrelation"

    def _summary_lines(self) -> list[str]:
        lines = [
            f"n = {self.n}",
            f"linear trend: slope = {self.slope:.4g}, p = {self.slope_p:.3g}",
            f"Mann-Kendall: tau = {self.mk_tau:.3g}, p = {self.mk_p:.3g} "
            f"({self.direction})",
            f"autocorrelation: {len(self.sig_lags)} lag(s) beyond +/-{self.acf_bound:.3f}"
            + (f" (lags {', '.join(map(str, self.sig_lags))})" if self.sig_lags else ""),
            "",
            "this is an exploratory screen; it complements, and does not replace, the CUSUM "
            "and EWMA charts for monitoring.",
        ]
        return lines

    def summary(self) -> dict:
        return {"n": self.n, "slope": self.slope, "slope_p": self.slope_p,
                "mk_tau": self.mk_tau, "mk_p": self.mk_p, "direction": self.direction,
                "n_sig_lags": len(self.sig_lags), "acf_bound": self.acf_bound}

    def _render_standalone(self, fig, kind, **kwargs):
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        pal = _pal.active()
        kind = kind or "trend"
        if kind == "trend":
            t = np.arange(self.n)
            ax.plot(t, self._y, color=pal.data, lw=1, marker="o", ms=3)
            fit = self.slope * t + (self._y.mean() - self.slope * t.mean())
            ax.plot(t, fit, color=pal.center, lw=2, label=f"trend (p={self.slope_p:.3g})")
            ax.set_xlabel("time order"); ax.set_ylabel("measure")
            ax.set_title("Trend"); ax.legend(fontsize=8)
        elif kind == "acf":
            ax.stem(self.acf_lags, self.acf_values)
            ax.axhline(self.acf_bound, color=pal.ooc, ls="--", lw=1)
            ax.axhline(-self.acf_bound, color=pal.ooc, ls="--", lw=1)
            ax.axhline(0, color=pal.limit, lw=0.8)
            ax.set_xlabel("lag"); ax.set_ylabel("ACF"); ax.set_title("Autocorrelation")
        else:
            raise ValueError(f"unknown timeseries view kind={kind!r}; use 'trend' or 'acf'.")
        return ax


def compute_timeseries(qc: QCData, *, lags: int = 20, alpha: float = 0.05) -> TimeSeriesResult:
    """Trend (linear regression on time + Mann-Kendall) and autocorrelation (ACF
    with confidence bounds), surfaced as flags. See module docstring."""
    from scipy import stats as _st
    y = qc.values()
    y = y[~np.isnan(y)]
    n = y.size
    if n < 4:
        raise ValueError("timeseries needs at least 4 observations.")
    t = np.arange(n, dtype=float)
    lin = _st.linregress(t, y)
    s, z, mk_p, tau = _mann_kendall(y)
    direction = "no trend" if mk_p >= alpha else ("increasing" if tau > 0 else "decreasing")

    yc = y - y.mean()
    denom = float(np.sum(yc * yc))
    L = min(lags, n - 1)
    acf = np.array([1.0] + [float(np.sum(yc[k:] * yc[:-k]) / denom) for k in range(1, L + 1)])
    acf_lags = np.arange(L + 1)
    bound = float(_st.norm.ppf(1 - alpha / 2) / np.sqrt(n))
    sig = tuple(int(k) for k in acf_lags[1:] if abs(acf[k]) > bound)

    trend_present = bool(lin.pvalue < alpha or mk_p < alpha)
    trend_flag = AssumptionCheck("trend", "linear + Mann-Kendall", float(lin.slope),
                                 float(min(lin.pvalue, mk_p)), not trend_present, float(tau),
                                 "Kendall tau", "ok", n,
                                 None if not trend_present else
                                 (f"A trend is present (slope p={lin.pvalue:.3g}, MK p={mk_p:.3g}); the "
                                  "series is not stationary - detrend before control charting or model "
                                  "the trend."))
    autocorr_present = bool(len(sig) > 0)
    acf_flag = AssumptionCheck("autocorrelation", "ACF beyond bound", float(acf[1]),
                               None, not autocorr_present, float(acf[1]), "lag-1 ACF",
                               "ok", n,
                               None if not autocorr_present else
                               (f"Autocorrelation at lag(s) {', '.join(map(str, sig))}; observations are "
                                "not independent - Shewhart limits are unreliable, prefer EWMA/CUSUM or a "
                                "time-series model."))
    step = Step(operation="timeseries", params={"slope_p": float(lin.pvalue), "mk_p": mk_p,
                                                "sig_lags": list(sig)}, n_affected=n, timestamp=_now())
    return TimeSeriesResult(
        slope=float(lin.slope), slope_p=float(lin.pvalue), mk_tau=tau, mk_p=mk_p, mk_z=z,
        direction=direction, acf_lags=acf_lags, acf_values=acf, acf_bound=bound, sig_lags=sig,
        n=n, _y=y, assumptions=[trend_flag, acf_flag], history=qc.history + (step,))
