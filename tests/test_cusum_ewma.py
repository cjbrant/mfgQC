"""Tests for EWMA and CUSUM time-series control charts (mfgqc.timeseries_charts)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import mfgqc
from mfgqc.timeseries_charts import (
    CUSUMResult,
    EWMAResult,
    compute_cusum,
    compute_ewma,
)


def _qc(values, **kwargs):
    df = pd.DataFrame({"x": np.asarray(values, dtype=float)})
    return mfgqc.load(df, measure="x", **kwargs)


# --------------------------------------------------------------------------- #
# EWMA oracles
# --------------------------------------------------------------------------- #
def test_ewma_steady_state_multiplier():
    lam = 0.1
    mult = math.sqrt(lam / (2.0 - lam))
    assert abs(mult - 0.22942) < 1e-5


def test_ewma_last_point_limit_matches_steady_state():
    # sigma=1, large n -> last-point half-width ~ L*sigma*sqrt(lam/(2-lam)).
    lam, L = 0.1, 2.7
    res = compute_ewma(_qc([10.0] * 200), lam=lam, L=L, mu0=10.0, sigma=1.0)
    halfwidth = res.ucl[-1] - res.center
    expected = L * 1.0 * math.sqrt(lam / (2.0 - lam))
    assert abs(halfwidth - expected) < 1e-3


def test_ewma_flat_input_no_signal():
    res = compute_ewma(_qc([10.0] * 50), lam=0.1, L=2.7, mu0=10.0, sigma=1.0)
    assert np.allclose(res.z, 10.0)
    assert len(res.violations) == 0
    assert res.summary()["n_signals"] == 0


def test_ewma_sustained_shift_signals():
    x = [10.0] * 20 + [13.0] * 20
    res = compute_ewma(_qc(x), lam=0.1, L=2.7, mu0=10.0, sigma=1.0)
    assert len(res.violations) >= 1
    # The EWMA should drift toward the new level.
    assert res.z[-1] > res.z[19]


def test_ewma_incontrol_normal_few_signals():
    rng = np.random.default_rng(0)
    x = rng.normal(10.0, 1.0, 50)
    res = compute_ewma(_qc(x), lam=0.1, L=2.7, mu0=10.0, sigma=1.0)
    assert len(res.violations) <= 2


def test_ewma_summary_flat():
    res = compute_ewma(_qc([10.0] * 30), lam=0.1, L=2.7, mu0=10.0, sigma=1.0)
    s = res.summary()
    assert set(s) == {"lam", "L", "mu0", "sigma", "n_signals"}
    for v in s.values():
        assert isinstance(v, (int, float))
        assert not isinstance(v, dict)


def test_ewma_view_returns_figure():
    res = compute_ewma(_qc([10.0] * 30), mu0=10.0, sigma=1.0)
    fig = res.view()
    assert isinstance(fig, Figure)


def test_ewma_defaults_target_then_mean():
    # spec target used for mu0 when set.
    qc = _qc([10.0] * 10).spec(target=12.0)
    res = compute_ewma(qc, sigma=1.0)
    assert res.mu0 == 12.0
    # no target -> sample mean.
    res2 = compute_ewma(_qc([5.0, 7.0, 9.0]), sigma=1.0)
    assert abs(res2.mu0 - 7.0) < 1e-9


# --------------------------------------------------------------------------- #
# CUSUM oracles
# --------------------------------------------------------------------------- #
def test_cusum_design_constants():
    res = compute_cusum(_qc([10.0] * 7), k=0.5, h=5, mu0=10.0, sigma=1.0)
    assert abs(res.K - 0.5) < 1e-12
    assert abs(res.H - 5.0) < 1e-12


def test_cusum_flat_input_no_signal():
    res = compute_cusum(_qc([10.0] * 50), k=0.5, h=5, mu0=10.0, sigma=1.0)
    assert np.allclose(res.c_plus, 0.0)
    assert np.allclose(res.c_minus, 0.0)
    assert len(res.violations) == 0


def test_cusum_hand_computed_series():
    x = [10, 10, 12, 12, 12, 12, 12]
    res = compute_cusum(_qc(x), k=0.5, h=5, mu0=10.0, sigma=1.0)
    expected_cplus = np.array([0.0, 0.0, 1.5, 3.0, 4.5, 6.0, 7.5])
    assert np.allclose(res.c_plus, expected_cplus)
    assert np.allclose(res.c_minus, 0.0)
    # First signal where C+ first exceeds H=5 -> point 6 (index 5, C+=6.0).
    signal_points = sorted(v.point for v in res.violations)
    assert signal_points[0] == 6
    assert signal_points == [6, 7]


def test_cusum_sustained_shift_signals():
    x = [10.0] * 10 + [11.5] * 20
    res = compute_cusum(_qc(x), k=0.5, h=5, mu0=10.0, sigma=1.0)
    assert len(res.violations) >= 1
    assert all(v.rule == "cusum_upper" for v in res.violations)


def test_cusum_downward_shift_signals_lower():
    x = [10.0] * 10 + [8.5] * 20
    res = compute_cusum(_qc(x), k=0.5, h=5, mu0=10.0, sigma=1.0)
    assert len(res.violations) >= 1
    assert all(v.rule == "cusum_lower" for v in res.violations)


def test_cusum_incontrol_normal_few_signals():
    rng = np.random.default_rng(0)
    x = rng.normal(10.0, 1.0, 50)
    res = compute_cusum(_qc(x), k=0.5, h=5, mu0=10.0, sigma=1.0)
    assert len(res.violations) <= 2


def test_cusum_summary_flat():
    res = compute_cusum(_qc([10.0] * 30), k=0.5, h=5, mu0=10.0, sigma=1.0)
    s = res.summary()
    assert set(s) == {"k", "h", "K", "H", "mu0", "sigma", "n_signals"}
    for v in s.values():
        assert isinstance(v, (int, float))
        assert not isinstance(v, dict)


def test_cusum_view_returns_figure():
    res = compute_cusum(_qc([10.0] * 30), mu0=10.0, sigma=1.0)
    fig = res.view()
    assert isinstance(fig, Figure)


# --------------------------------------------------------------------------- #
# Shared behavior
# --------------------------------------------------------------------------- #
def test_default_sigma_from_moving_range():
    # Constant data -> MR-bar=0 -> sigma estimate 0; user can still pass sigma.
    res = compute_cusum(_qc([1.0, 3.0, 1.0, 3.0, 1.0]))
    # MR all = 2 -> MR-bar=2 -> sigma = 2/1.128.
    assert abs(res.sigma - (2.0 / 1.128)) < 1e-9


def test_results_are_correct_types():
    e = compute_ewma(_qc([10.0] * 5), mu0=10.0, sigma=1.0)
    c = compute_cusum(_qc([10.0] * 5), mu0=10.0, sigma=1.0)
    assert isinstance(e, EWMAResult)
    assert isinstance(c, CUSUMResult)
    # report() works (inherited from QCResult).
    assert isinstance(e.report(), str)
    assert isinstance(c.report(), str)
