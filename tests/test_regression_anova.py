"""Regression / correlation / general ANOVA tests.

Oracles are pinned to scipy's standard implementations:
- simple OLS  -> scipy.stats.linregress
- one-way ANOVA -> scipy.stats.f_oneway
- correlation -> scipy.stats.pearsonr
The two-way ANOVA is cross-checked by the SS decomposition identity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from scipy import stats

import mfgqc
from mfgqc.assumptions import AssumptionCheck
from mfgqc.regression import (
    AnovaResult,
    CorrelationResult,
    RegressionResult,
    compute_anova,
    compute_regression,
    correlation,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def simple_qc():
    """y = 2*x + 0.05 + tiny noise on x = 1..20 (seed 0, sd ~0.1)."""
    rng = np.random.default_rng(0)
    x = np.arange(1, 21, dtype=float)
    y = 2.0 * x + 0.05 + rng.normal(0.0, 0.1, size=x.size)
    df = pd.DataFrame({"x": x, "y": y})
    return mfgqc.load(df, measure="y"), x, y


@pytest.fixture
def multi_qc():
    rng = np.random.default_rng(7)
    n = 60
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    return mfgqc.load(df, measure="y")


# Three groups for one-way ANOVA.
G1 = [3, 4, 5, 6, 7]
G2 = [6, 7, 8, 9, 10]
G3 = [2, 3, 4, 5, 6]


@pytest.fixture
def oneway_qc():
    rows = []
    for name, g in (("g1", G1), ("g2", G2), ("g3", G3)):
        for v in g:
            rows.append({"grp": name, "y": float(v)})
    df = pd.DataFrame(rows)
    return mfgqc.load(df, measure="y")


@pytest.fixture
def twoway_qc():
    """Balanced 2x2 with 4 replicates each."""
    rng = np.random.default_rng(3)
    rows = []
    effects = {("lo", "x"): 10.0, ("lo", "y"): 12.0,
               ("hi", "x"): 14.0, ("hi", "y"): 20.0}
    for a in ("lo", "hi"):
        for b in ("x", "y"):
            for _ in range(4):
                rows.append({"A": a, "B": b,
                             "resp": effects[(a, b)] + rng.normal(0, 0.5)})
    df = pd.DataFrame(rows)
    return mfgqc.load(df, measure="resp")


# --------------------------------------------------------------------------- #
# Simple regression - pinned to scipy.stats.linregress
# --------------------------------------------------------------------------- #
def test_simple_regression_recovers_truth(simple_qc):
    qc, x, y = simple_qc
    res = compute_regression(qc, "x")
    assert isinstance(res, RegressionResult)
    assert res.coef["x"] == pytest.approx(2.0, abs=0.1)
    assert res.coef["intercept"] == pytest.approx(0.0, abs=0.2)
    assert res.r_squared > 0.99


def test_simple_regression_matches_linregress(simple_qc):
    qc, x, y = simple_qc
    res = compute_regression(qc, "x")
    lr = stats.linregress(x, y)
    assert res.coef["x"] == pytest.approx(lr.slope, rel=1e-6)
    assert res.coef["intercept"] == pytest.approx(lr.intercept, rel=1e-6)
    assert res.se["x"] == pytest.approx(lr.stderr, rel=1e-6)
    assert res.r_squared == pytest.approx(lr.rvalue ** 2, rel=1e-6)
    # linregress two-sided p for the slope == our slope coefficient p
    assert res.p_value["x"] == pytest.approx(lr.pvalue, rel=1e-6)
    # F-of-regression p (1 predictor) == slope p
    assert res.f_p_value == pytest.approx(lr.pvalue, rel=1e-6)


def test_regression_ci_brackets_estimate(simple_qc):
    qc, _x, _y = simple_qc
    res = compute_regression(qc, "x")
    for name in res.terms:
        lo, hi = res.ci[name]
        assert lo <= res.coef[name] <= hi


def test_regression_r_squared_in_unit_interval(multi_qc):
    res = compute_regression(multi_qc, ["x1", "x2"])
    assert 0.0 <= res.r_squared <= 1.0
    assert res.adj_r_squared <= res.r_squared


def test_multiple_regression_recovers_coefficients(multi_qc):
    res = compute_regression(multi_qc, ["x1", "x2"])
    assert set(res.terms) == {"intercept", "x1", "x2"}
    assert res.coef["x1"] == pytest.approx(2.0, abs=0.2)
    assert res.coef["x2"] == pytest.approx(-0.5, abs=0.2)
    assert res.df_resid == res.n - 3


def test_regression_assumptions_present(simple_qc):
    qc, _x, _y = simple_qc
    res = compute_regression(qc, "x")
    names = {a.name for a in res.assumptions}
    assert {"normality", "homoscedasticity", "independence"} <= names
    for a in res.assumptions:
        assert isinstance(a, AssumptionCheck)
        assert isinstance(a.passed, bool)


def test_regression_summary_is_flat(simple_qc):
    qc, _x, _y = simple_qc
    summ = compute_regression(qc, "x").summary()
    assert isinstance(summ, dict)
    for v in summ.values():
        assert not isinstance(v, (dict, list, tuple))


def test_regression_view_returns_figure(simple_qc, multi_qc):
    qc, _x, _y = simple_qc
    fig = compute_regression(qc, "x").view()
    assert isinstance(fig, Figure)
    fig2 = compute_regression(multi_qc, ["x1", "x2"]).view()
    assert isinstance(fig2, Figure)


def test_regression_report_renders(simple_qc):
    qc, _x, _y = simple_qc
    txt = compute_regression(qc, "x").report()
    assert "Regression" in txt
    assert "Assumption checks:" in txt


# --------------------------------------------------------------------------- #
# Correlation - pinned to scipy.stats.pearsonr
# --------------------------------------------------------------------------- #
def test_correlation_near_one_for_linear():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 50)
    y = 2.0 * x + rng.normal(0, 0.01, x.size)
    df = pd.DataFrame({"x": x, "y": y})
    res = correlation(df, ["x", "y"])
    assert isinstance(res, CorrelationResult)
    assert res.corr[("x", "y")] == pytest.approx(1.0, abs=1e-3)


def test_correlation_matches_pearsonr():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 80)
    y = 0.7 * x + rng.normal(0, 1, 80)
    df = pd.DataFrame({"x": x, "y": y})
    res = correlation(df, ["x", "y"])
    r, p = stats.pearsonr(x, y)
    assert res.corr[("x", "y")] == pytest.approx(r, rel=1e-6)
    assert res.p_values[("x", "y")] == pytest.approx(p, rel=1e-6)


def test_correlation_spearman_option():
    rng = np.random.default_rng(4)
    x = np.arange(40, dtype=float)
    y = x ** 2 + rng.normal(0, 1, 40)  # monotone -> spearman ~1
    df = pd.DataFrame({"x": x, "y": y})
    res = correlation(df, ["x", "y"], method="spearman")
    r, p = stats.spearmanr(x, y)
    assert res.corr[("x", "y")] == pytest.approx(r, rel=1e-6)
    assert res.p_values[("x", "y")] == pytest.approx(p, rel=1e-6)


def test_correlation_default_all_numeric():
    df = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 4, 6, 8],
                       "label": ["w", "x", "y", "z"]})
    res = correlation(df)
    assert set(res.cols) == {"a", "b"}


def test_correlation_summary_flat_and_view():
    df = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) * -1.5 + 1,
                       "c": np.random.default_rng(9).normal(size=20)})
    res = correlation(df)
    summ = res.summary()
    for v in summ.values():
        assert not isinstance(v, (dict, list, tuple))
    assert isinstance(res.view(), Figure)


# --------------------------------------------------------------------------- #
# One-way ANOVA - pinned to scipy.stats.f_oneway
# --------------------------------------------------------------------------- #
def test_oneway_matches_f_oneway(oneway_qc):
    res = compute_anova(oneway_qc, "grp")
    assert isinstance(res, AnovaResult)
    f, p = stats.f_oneway(G1, G2, G3)
    assert res.table["grp"]["f"] == pytest.approx(f, rel=1e-6)
    assert res.table["grp"]["p_value"] == pytest.approx(p, rel=1e-6)


def test_oneway_ss_rows_sum_to_total(oneway_qc):
    res = compute_anova(oneway_qc, "grp")
    ss_factor = res.table["grp"]["ss"]
    ss_error = res.table["residual"]["ss"]
    ss_total = res.table["total"]["ss"]
    assert ss_factor + ss_error == pytest.approx(ss_total, rel=1e-9)
    # df also add up
    assert (res.table["grp"]["df"] + res.table["residual"]["df"]
            == res.table["total"]["df"])


def test_oneway_eta_squared(oneway_qc):
    res = compute_anova(oneway_qc, "grp")
    eta = res.table["grp"]["eta_sq"]
    assert 0.0 <= eta <= 1.0
    assert eta == pytest.approx(res.table["grp"]["ss"] / res.table["total"]["ss"])


def test_oneway_assumptions_present(oneway_qc):
    res = compute_anova(oneway_qc, "grp")
    names = {a.name for a in res.assumptions}
    assert "normality" in names
    assert "homogeneity_of_variance" in names
    for a in res.assumptions:
        assert isinstance(a, AssumptionCheck)
        assert isinstance(a.passed, bool)


def test_oneway_summary_flat_and_view(oneway_qc):
    res = compute_anova(oneway_qc, "grp")
    summ = res.summary()
    for v in summ.values():
        assert not isinstance(v, (dict, list, tuple))
    assert isinstance(res.view(), Figure)


# --------------------------------------------------------------------------- #
# Two-way ANOVA - SS decomposition + scipy cross-check
# --------------------------------------------------------------------------- #
def test_twoway_ss_decomposition(twoway_qc):
    res = compute_anova(twoway_qc, ["A", "B"])
    ss_a = res.table["A"]["ss"]
    ss_b = res.table["B"]["ss"]
    ss_ab = res.table["A:B"]["ss"]
    ss_err = res.table["residual"]["ss"]
    ss_tot = res.table["total"]["ss"]
    assert ss_a + ss_b + ss_ab + ss_err == pytest.approx(ss_tot, rel=1e-9)


def test_twoway_df_decomposition(twoway_qc):
    res = compute_anova(twoway_qc, ["A", "B"])
    df_sum = (res.table["A"]["df"] + res.table["B"]["df"]
              + res.table["A:B"]["df"] + res.table["residual"]["df"])
    assert df_sum == res.table["total"]["df"]


def test_twoway_main_effect_matches_oneway_f(twoway_qc):
    """For a balanced design, factor-A main-effect SS equals the between-group SS
    of a one-way grouping on A; cross-check the A-margin SS against an explicit
    f_oneway-style between SS computation."""
    res = compute_anova(twoway_qc, ["A", "B"])
    df = twoway_qc.frame
    grand = df["resp"].mean()
    ss_a_expected = sum(
        len(g) * (g["resp"].mean() - grand) ** 2
        for _lvl, g in df.groupby("A")
    )
    assert res.table["A"]["ss"] == pytest.approx(ss_a_expected, rel=1e-9)


def test_twoway_eta_sq_present(twoway_qc):
    res = compute_anova(twoway_qc, ["A", "B"])
    for term in ("A", "B", "A:B"):
        assert 0.0 <= res.table[term]["eta_sq"] <= 1.0


def test_twoway_view_is_interaction_plot(twoway_qc):
    res = compute_anova(twoway_qc, ["A", "B"])
    assert isinstance(res.view(), Figure)


def test_twoway_assumptions_present(twoway_qc):
    res = compute_anova(twoway_qc, ["A", "B"])
    names = {a.name for a in res.assumptions}
    assert {"normality", "homogeneity_of_variance"} <= names


# --------------------------------------------------------------------------- #
# QCData wiring expectation (parent adds .regress/.anova; we use compute_*)
# --------------------------------------------------------------------------- #
def test_history_propagates(oneway_qc):
    res = compute_anova(oneway_qc, "grp")
    ops = [s.operation for s in res.history]
    assert "anova" in ops
    assert any(op.startswith("assumption:") for op in ops)
