"""QC-scoped time-series characterization (Track 1C) oracle tests.

Pins trend / ACF-PACF / additive-decomposition against numpy/scipy ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from scipy import stats

import mfgqc
from mfgqc.timeseries import compute_acf, compute_decompose, compute_trend


# --------------------------------------------------------------------------- #
# Trend
# --------------------------------------------------------------------------- #
def _trending_qc(with_time_role: bool):
    rng = np.random.default_rng(0)
    t = np.arange(50, dtype=float)
    y = 0.5 * t + rng.standard_normal(50)
    df = pd.DataFrame({"y": y, "t": t})
    if with_time_role:
        return mfgqc.load(df, measure="y", roles={"time": "t"}), t, y
    return mfgqc.load(df, measure="y"), t, y


def test_trend_slope_matches_linregress_with_time_role():
    qc, t, y = _trending_qc(with_time_role=True)
    res = compute_trend(qc)
    lr = stats.linregress(t, y)
    assert res.slope == pytest.approx(0.5, abs=0.05)
    assert res.slope == pytest.approx(lr.slope, rel=1e-6)
    assert res.p_value == pytest.approx(lr.pvalue, rel=1e-6)
    assert res.verdict == "drifting"
    assert res.time_col == "t"


def test_trend_index_fallback_matches_linregress():
    qc, t, y = _trending_qc(with_time_role=False)
    res = compute_trend(qc)
    lr = stats.linregress(t, y)  # index is 0..49 == t
    assert res.slope == pytest.approx(lr.slope, rel=1e-6)
    assert res.p_value == pytest.approx(lr.pvalue, rel=1e-6)
    assert res.verdict == "drifting"
    assert res.time_col == "index"


def test_trend_flat_series_is_stable():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(60)  # no slope component
    df = pd.DataFrame({"y": y})
    qc = mfgqc.load(df, measure="y")
    res = compute_trend(qc)
    assert res.verdict == "stable"
    assert res.p_value >= 0.05


def test_trend_surfaces_regression_assumptions():
    qc, _t, _y = _trending_qc(with_time_role=True)
    res = compute_trend(qc)
    names = {a.name for a in res.assumptions}
    assert {"normality", "homoscedasticity", "independence"} <= names


def test_trend_summary_flat_and_view_figure():
    qc, _t, _y = _trending_qc(with_time_role=True)
    res = compute_trend(qc)
    s = res.summary()
    assert set(["slope", "p_value", "r_squared", "verdict", "n"]) <= set(s)
    assert all(not isinstance(v, dict) for v in s.values())
    fig = res.view()
    assert isinstance(fig, Figure)


# --------------------------------------------------------------------------- #
# Autocorrelation (ACF / PACF)
# --------------------------------------------------------------------------- #
def _ar1_qc(phi=0.8, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    df = pd.DataFrame({"y": x})
    return mfgqc.load(df, measure="y"), x


def test_acf_ar1_lag1():
    qc, _x = _ar1_qc()
    res = compute_acf(qc, lags=20)
    assert res.acf[0] == pytest.approx(0.8, abs=0.05)


def test_pacf_ar1_cutoff_after_lag1():
    qc, _x = _ar1_qc()
    res = compute_acf(qc, lags=20)
    assert res.pacf[0] == pytest.approx(0.8, abs=0.08)
    assert res.pacf[1] == pytest.approx(0.0, abs=0.08)


def test_acf_confidence_band_and_significant_lags():
    qc, _x = _ar1_qc()
    res = compute_acf(qc, lags=20)
    assert res.conf == pytest.approx(1.96 / np.sqrt(2000), rel=1e-9)
    assert 1 in res.significant_lags  # lag-1 clearly breaks the band


def test_acf_formula_matches_manual():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(200)
    df = pd.DataFrame({"y": x})
    qc = mfgqc.load(df, measure="y")
    res = compute_acf(qc, lags=5)
    xc = x - x.mean()
    denom = np.sum(xc * xc)
    for k in range(1, 6):
        manual = np.sum(xc[:-k] * xc[k:]) / denom
        assert res.acf[k - 1] == pytest.approx(manual, rel=1e-9, abs=1e-12)


def test_acf_summary_flat_and_view_figure():
    qc, _x = _ar1_qc()
    res = compute_acf(qc, lags=20)
    s = res.summary()
    assert {"acf_lag1", "pacf_lag1", "n_significant", "conf", "n"} <= set(s)
    assert res.assumptions == []
    fig = res.view()
    assert isinstance(fig, Figure)


# --------------------------------------------------------------------------- #
# Classical additive decomposition
# --------------------------------------------------------------------------- #
def _seasonal_qc(period=12, amplitude=5.0, n=120, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    trend = 0.1 * t
    seasonal = amplitude * np.sin(2 * np.pi * t / period)
    noise = 0.01 * rng.standard_normal(n)
    observed = trend + seasonal + noise
    df = pd.DataFrame({"y": observed})
    return mfgqc.load(df, measure="y"), observed


def test_decompose_reconstructs_observed():
    qc, observed = _seasonal_qc()
    res = compute_decompose(qc, period=12)
    recon = res.trend + res.seasonal + res.resid
    defined = np.isfinite(res.trend)
    assert np.allclose(recon[defined], observed[defined], atol=1e-9)


def test_decompose_recovers_seasonal_amplitude():
    qc, _observed = _seasonal_qc(amplitude=5.0)
    res = compute_decompose(qc, period=12)
    # sine amplitude 5 -> peak-to-trough ~10
    assert res.seasonal_amplitude == pytest.approx(10.0, abs=1.5)


def test_decompose_seasonal_sums_to_zero_over_period():
    qc, _observed = _seasonal_qc()
    res = compute_decompose(qc, period=12)
    one_cycle = res.seasonal[:12]
    assert np.sum(one_cycle) == pytest.approx(0.0, abs=1e-9)


def test_decompose_trend_nan_padded_at_ends():
    qc, _observed = _seasonal_qc()
    res = compute_decompose(qc, period=12)
    assert np.isnan(res.trend[0])
    assert np.isnan(res.trend[-1])
    assert np.isfinite(res.trend[res.n // 2])


def test_decompose_summary_flat_and_view_figure():
    qc, _observed = _seasonal_qc()
    res = compute_decompose(qc, period=12)
    s = res.summary()
    assert {"period", "seasonal_amplitude", "resid_std", "n"} <= set(s)
    assert res.assumptions == []
    fig = res.view()
    assert isinstance(fig, Figure)


def test_decompose_rejects_too_short_series():
    df = pd.DataFrame({"y": np.arange(10.0)})
    qc = mfgqc.load(df, measure="y")
    with pytest.raises(ValueError):
        compute_decompose(qc, period=12)
