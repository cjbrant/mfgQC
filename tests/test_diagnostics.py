"""Tests for the model-interpretive diagnostic layer (mfgqc.diagnostics).

Oracles are cross-checked against numpy/scipy. The headline contract here is the
HARD BOUNDARY: mfgQC interprets model OUTPUTS only and never imports or inspects
any ML framework - it takes predictions through a duck-typed ``model.predict``
and nothing else. The fake-model test below proves that with no ML library
present, and ``test_no_ml_framework_import`` asserts the source stays clean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from mfgqc.diagnostics import ModelDiagnosticResult, diagnose


# --------------------------------------------------------------------------- #
# Fixtures: a "good" model (clean N(0,1) residuals) and a "degrading" one
# (AR(1) + ramp -> serial correlation and drift).
# --------------------------------------------------------------------------- #
@pytest.fixture
def good_resid() -> np.ndarray:
    return np.random.default_rng(20260618).normal(0, 1, 500)


@pytest.fixture
def degrading_resid() -> np.ndarray:
    rng = np.random.default_rng(20260618)
    n = 200
    phi = 0.6
    e = rng.normal(0, 1, n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = phi * ar[i - 1] + e[i]
    return ar + np.linspace(0, 4, n)  # autocorrelation + upward drift


def _check(result, name):
    return next(a for a in result.assumptions if a.name == name)


# --------------------------------------------------------------------------- #
# Good model: residuals ~ N(0,1)
# --------------------------------------------------------------------------- #
def test_good_model_normality_passes(good_resid):
    res = diagnose(good_resid, np.zeros_like(good_resid))
    normality = _check(res, "normality")
    assert normality.passed is True
    assert normality.statistic < 1.0  # AD small for clean normal data


def test_good_model_residual_stats_match_numpy(good_resid):
    r = good_resid
    res = diagnose(r, np.zeros_like(r))
    assert res.n == 500
    assert res.mean == pytest.approx(float(r.mean()), abs=1e-12)
    assert res.std == pytest.approx(float(np.std(r, ddof=1)), abs=1e-12)
    assert res.rmse == pytest.approx(float(np.sqrt(np.mean(r ** 2))), abs=1e-12)
    assert res.mae == pytest.approx(float(np.mean(np.abs(r))), abs=1e-12)


def test_good_model_pct_within_and_cpk(good_resid):
    res = diagnose(good_resid, np.zeros_like(good_resid), tolerance=2)
    assert res.tolerance == (-2.0, 2.0)
    assert res.pct_within == pytest.approx(95.6, abs=0.01)
    assert res.residual_cpk == pytest.approx(0.682, abs=0.01)


def test_good_model_independence_and_bias_pass(good_resid):
    res = diagnose(good_resid, np.zeros_like(good_resid))
    assert _check(res, "independence").passed is True
    assert _check(res, "zero_mean_bias").passed is True
    assert _check(res, "homoscedasticity").passed is True


# --------------------------------------------------------------------------- #
# Degrading model: AR(1) + ramp -> independence flags
# --------------------------------------------------------------------------- #
def test_degrading_model_independence_fails(degrading_resid):
    res = diagnose(degrading_resid, np.zeros_like(degrading_resid))
    indep = _check(res, "independence")
    assert indep.passed is False
    # Durbin-Watson far from 2 (strong positive serial correlation).
    assert indep.statistic < 1.5
    assert indep.magnitude > 0.3  # high |lag-1 autocorr| context


def test_degrading_model_drift_verdict(degrading_resid):
    r = degrading_resid
    res = diagnose(r, np.zeros_like(r), order=np.arange(r.size))
    assert res.drift == "degrading"
    assert res.n_signals is not None and res.n_signals > 0
    assert res.chart is not None


# --------------------------------------------------------------------------- #
# Duck-typed model path: framework-agnostic contract, no ML library involved.
# --------------------------------------------------------------------------- #
def test_duck_typed_model_path_matches_outputs_path():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 3))
    w = np.array([1.5, -2.0, 0.5])
    y = X @ w + rng.normal(0, 0.1, size=120)

    class FakeModel:
        """A model defined entirely in the test - it has ONLY .predict().

        No sklearn / torch / tensorflow anywhere. If mfgQC needed to know the
        model's *kind*, this would be impossible.
        """

        def __init__(self, coef):
            self._coef = coef

        def predict(self, X):
            return X @ self._coef

    fake = FakeModel(w)
    via_model = diagnose(model=fake, X=X, y=y)
    via_outputs = diagnose(y, fake.predict(X))

    assert isinstance(via_model, ModelDiagnosticResult)
    assert via_model.n == via_outputs.n
    assert via_model.mean == pytest.approx(via_outputs.mean, abs=1e-12)
    assert via_model.std == pytest.approx(via_outputs.std, abs=1e-12)
    assert via_model.rmse == pytest.approx(via_outputs.rmse, abs=1e-12)


# --------------------------------------------------------------------------- #
# Tiers are optional
# --------------------------------------------------------------------------- #
def test_tier2_and_tier3_absent_by_default(good_resid):
    res = diagnose(good_resid, np.zeros_like(good_resid))
    # Tier 2 absent
    assert res.tolerance is None
    assert res.pct_within is None
    assert res.residual_cpk is None
    # Tier 3 absent
    assert res.chart is None
    assert res.n_signals is None
    assert res.drift == "n/a"


def test_tolerance_single_number_is_symmetric(good_resid):
    res = diagnose(good_resid, np.zeros_like(good_resid), tolerance=1.5)
    assert res.tolerance == (-1.5, 1.5)


def test_tolerance_pair_accepted(good_resid):
    res = diagnose(good_resid, np.zeros_like(good_resid), tolerance=(-3, 3))
    assert res.tolerance == (-3.0, 3.0)


# --------------------------------------------------------------------------- #
# Result surface: flat summary dict + Figure from view()
# --------------------------------------------------------------------------- #
def test_summary_is_flat_dict(good_resid):
    res = diagnose(good_resid, np.zeros_like(good_resid), tolerance=2)
    summary = res.summary()
    assert isinstance(summary, dict)
    # No nested containers - dashboard-ready flat scalars/None.
    for k, v in summary.items():
        assert not isinstance(v, (dict, list, tuple, np.ndarray)), f"{k} is nested"


def test_view_returns_figure(good_resid):
    res = diagnose(good_resid, np.zeros_like(good_resid))
    assert isinstance(res.view(), Figure)


def test_view_with_order_includes_control_chart(degrading_resid):
    r = degrading_resid
    res = diagnose(r, np.zeros_like(r), order=np.arange(r.size))
    fig = res.view()
    assert isinstance(fig, Figure)
    # residual-vs-fitted + histogram + control chart -> more than 2 axes.
    assert len(fig.axes) >= 3


def test_report_renders(good_resid):
    res = diagnose(good_resid, np.zeros_like(good_resid), tolerance=2)
    text = res.report()
    assert "Model Diagnostic" in text
    assert "Assumption checks:" in text


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #
def test_missing_arrays_raises():
    with pytest.raises(ValueError):
        diagnose()


# --------------------------------------------------------------------------- #
# THE BOUNDARY: diagnostics.py must not import any ML framework.
# --------------------------------------------------------------------------- #
def test_no_ml_framework_import():
    src = Path(__file__).resolve().parent.parent / "mfgqc" / "diagnostics.py"
    text = src.read_text()
    for forbidden in ("import sklearn", "import torch", "import tensorflow",
                      "from sklearn", "from torch", "from tensorflow",
                      "import xgboost", "import lightgbm"):
        assert forbidden not in text, f"diagnostics.py must not contain {forbidden!r}"
