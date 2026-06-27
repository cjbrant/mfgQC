"""Slice 7: time-series screen and multi-vari. Trend/ACF pinned to scipy +
negative controls (a stationary white series shows no trend and a flat ACF);
multi-vari variance components verified by construction (the seeded dominant
family is recovered as the largest component).
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

import mfgqc

matplotlib.use("Agg")


# --- time-series screen ------------------------------------------------------
def test_white_noise_shows_no_trend_flat_acf():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, 200)})
    res = mfgqc.load(df, measure="x").timeseries()
    assert res.direction == "no trend"
    assert res.mk_p > 0.05 and res.slope_p > 0.05
    assert len(res.sig_lags) <= 2                      # ~alpha false positives only
    trend = next(a for a in res.assumptions if a.name == "trend")
    assert trend.passed is True


def test_linear_trend_detected_both_ways():
    rng = np.random.default_rng(1)
    n = 100
    df = pd.DataFrame({"x": 0.05 * np.arange(n) + rng.normal(0, 0.5, n)})
    res = mfgqc.load(df, measure="x").timeseries()
    assert res.direction == "increasing"
    assert res.slope_p < 0.001 and res.mk_p < 0.001
    assert res.assumptions[0].passed is False          # trend flag fires


def test_mann_kendall_matches_manual_and_scipy_tau():
    from scipy import stats
    rng = np.random.default_rng(2)
    y = np.sort(rng.normal(0, 1, 40)) + rng.normal(0, 0.3, 40)
    res = mfgqc.load(pd.DataFrame({"x": y}), measure="x").timeseries()
    tau, p = stats.kendalltau(np.arange(y.size), y)
    assert abs(res.mk_tau - tau) < 1e-9                # MK tau == Kendall tau vs time


def test_autocorrelation_flagged_on_ar1():
    rng = np.random.default_rng(3)
    n = 300
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + rng.normal(0, 1)
    res = mfgqc.load(pd.DataFrame({"x": x}), measure="x").timeseries()
    assert 1 in res.sig_lags
    acf = next(a for a in res.assumptions if a.name == "autocorrelation")
    assert acf.passed is False and abs(res.acf_values[1] - 0.7) < 0.1


def test_timeseries_views():
    df = pd.DataFrame({"x": np.random.default_rng(0).normal(0, 1, 100)})
    res = mfgqc.load(df, measure="x").timeseries()
    assert res.view(kind="trend") is not None
    assert res.view(kind="acf") is not None


# --- multi-vari --------------------------------------------------------------
def _multivari_data(rng, temporal_sd, cyclical_sd, positional_sd, n_t=4, n_c=5, n_p=4):
    rows = []
    for t in range(n_t):
        t_eff = rng.normal(0, temporal_sd)
        for c in range(n_c):
            c_eff = rng.normal(0, cyclical_sd)
            for p in range(n_p):
                y = 100 + t_eff + c_eff + rng.normal(0, positional_sd)
                rows.append({"shift": f"T{t}", "part": f"T{t}P{c}", "pos": p, "y": y})
    return pd.DataFrame(rows)


def test_multivari_recovers_dominant_temporal():
    df = _multivari_data(np.random.default_rng(4), temporal_sd=8, cyclical_sd=1, positional_sd=1)
    res = mfgqc.load(df, measure="y").multivari(factors=["shift", "part"])
    assert res.families["temporal"] == "shift"
    dom = max(res.components, key=res.components.get)
    assert dom == "shift"                              # temporal family dominates
    assert res.percents["shift"] > 60


def test_multivari_recovers_dominant_positional():
    df = _multivari_data(np.random.default_rng(5), temporal_sd=1, cyclical_sd=1, positional_sd=8)
    res = mfgqc.load(df, measure="y").multivari(factors=["shift", "part"])
    dom = max(res.components, key=res.components.get)
    assert dom == "within"                             # within-piece (positional) dominates


def test_multivari_three_factors_and_view():
    df = _multivari_data(np.random.default_rng(6), 5, 2, 1)
    res = mfgqc.load(df, measure="y").multivari(factors=["shift", "part", "pos"])
    assert set(res.families) >= {"temporal", "cyclical", "positional"}
    assert abs(sum(res.percents.values()) - 100.0) < 1e-6
    assert res.view() is not None


def test_multivari_requires_2_or_3_factors():
    df = _multivari_data(np.random.default_rng(7), 5, 2, 1)
    with pytest.raises(ValueError, match="2 or 3"):
        mfgqc.load(df, measure="y").multivari(factors=["shift"])
