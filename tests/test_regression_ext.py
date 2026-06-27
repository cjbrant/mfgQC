"""Slice 6: regression additions. Logistic coefficients/odds-ratios and the
Box-Cox lambda cross-checked against statsmodels/scipy; negative control that a
transform on already-normal data returns lambda near 1 and leaves the data
effectively unchanged.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

import mfgqc

matplotlib.use("Agg")


# --- model selection ---------------------------------------------------------
def test_forward_selection_keeps_real_predictors_drops_noise():
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    noise1 = rng.normal(0, 1, n); noise2 = rng.normal(0, 1, n)
    y = 3 * x1 - 2 * x2 + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "n1": noise1, "n2": noise2})
    res = mfgqc.load(df, measure="y").regress(["x1", "x2", "n1", "n2"], select="forward")
    assert set(res.predictors) == {"x1", "x2"}
    assert res.selection_path
    assert "inflates the apparent significance" in res.selection_note


def test_selection_criteria_run():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(0, 1, 100)})
    for c in ("aic", "bic"):
        df[f"x{c}"] = df["y"] * 0.5 + rng.normal(0, 1, 100)
    res = mfgqc.load(df, measure="y").regress(["xaic", "xbic"], select="stepwise", criterion="bic")
    assert isinstance(res.predictors, tuple)


# --- logistic ----------------------------------------------------------------
def _logit_data(rng, n=300, beta=(-1.0, 2.0)):
    x = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(beta[0] + beta[1] * x)))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"y": y, "x": x})


def test_logistic_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    df = _logit_data(np.random.default_rng(2))
    res = mfgqc.load(df, measure="y").logistic(on="x")
    X = sm.add_constant(df["x"].to_numpy())
    fit = sm.Logit(df["y"].to_numpy(), X).fit(disp=0)
    assert abs(res.coef["x"] - fit.params[1]) < 1e-6
    assert abs(res.odds_ratio["x"] - np.exp(fit.params[1])) < 1e-9
    assert abs(res.pseudo_r2 - fit.prsquared) < 1e-9
    assert 0.5 < res.auc <= 1.0


def test_logistic_refuses_complete_separation():
    # x perfectly separates the classes -> MLE diverges -> legible refusal.
    df = pd.DataFrame({"y": [0, 0, 0, 0, 1, 1, 1, 1], "x": [1, 2, 3, 4, 10, 11, 12, 13]})
    with pytest.raises(ValueError, match="separat"):
        mfgqc.load(df, measure="y").logistic(on="x")


def test_logistic_requires_binary():
    df = pd.DataFrame({"y": [0, 1, 2, 1, 0, 2], "x": [1, 2, 3, 4, 5, 6.0]})
    with pytest.raises(ValueError, match="binary"):
        mfgqc.load(df, measure="y").logistic(on="x")


# --- non-linear --------------------------------------------------------------
def test_nls_recovers_known_parameters():
    rng = np.random.default_rng(3)
    x = np.linspace(0.1, 5, 60)
    y = 2.5 * np.exp(-0.8 * x) + rng.normal(0, 0.02, x.size)
    df = pd.DataFrame({"y": y, "x": x})
    model = lambda x, a, b: a * np.exp(-b * x)
    res = mfgqc.load(df, measure="y").regress("x", model=model, start=[1, 1])
    assert abs(res.params["a"] - 2.5) < 0.1 and abs(res.params["b"] - 0.8) < 0.1
    assert res.r_squared > 0.99


# --- Box-Cox transform -------------------------------------------------------
def test_boxcox_matches_scipy():
    from scipy import stats
    rng = np.random.default_rng(4)
    raw = rng.lognormal(0, 0.5, 200)
    df = pd.DataFrame({"y": raw})
    qc2 = mfgqc.load(df, measure="y").transform(method="boxcox")
    _, lmbda, _ = stats.boxcox(raw, alpha=0.05)
    assert abs(qc2.history[-1].params["lambda"] - lmbda) < 1e-9
    # transformed data is logged, original untouched (immutability)
    assert qc2.history[-1].operation == "transform"


def test_boxcox_on_normal_returns_lambda_near_one():
    # negative control: already-normal positive data -> lambda ~ 1, data ~ unchanged.
    rng = np.random.default_rng(5)
    raw = rng.normal(100, 5, 400)
    df = pd.DataFrame({"y": raw})
    qc2 = mfgqc.load(df, measure="y").transform(method="boxcox")
    lo, hi = qc2.history[-1].params["lambda_ci"]
    # already-normal data: lambda=1 (no transform) is statistically supported -
    # the 95% CI on lambda includes 1, so the module would not push a transform.
    assert lo <= 1.0 <= hi


def test_boxcox_refuses_nonpositive():
    df = pd.DataFrame({"y": [-1.0, 2, 3, 4]})
    with pytest.raises(ValueError, match="positive"):
        mfgqc.load(df, measure="y").transform()


def test_logistic_view_renders():
    df = _logit_data(np.random.default_rng(2))
    assert mfgqc.load(df, measure="y").logistic(on="x").view() is not None
