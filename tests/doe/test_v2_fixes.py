"""Build spec v2 deltas:

- 4.3 over-parameterization refusal: a model with more parameters than DISTINCT
  runs is refused before fitting, with a DOE-aware message (not a raw lstsq error).
- 4.2a fraction detection on the factors= path: a confounded external matrix
  surfaces the same generators / resolution / alias list as the generation path.
- 4.6 calibrated constant-variance test (Brown-Forsythe across factor levels) and
  reliability tempering on the residual checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mfgqc
from mfgqc.doe import generate as gen
from ._oracles import SOUP, SOUP_ALIAS, SOUP_COEF, VOLT


# --- 4.3 over-parameterization refusal ----------------------------------------
def test_over_parameterized_fraction_refused():
    # 2^(4-1): 8 distinct runs, order-2 model needs 11 params.
    m = gen.coded_full_matrix(3)
    D = m[:, 0] * m[:, 1] * m[:, 2]          # D = ABC -> a 2^(4-1) in 8 runs
    df = pd.DataFrame({"A": m[:, 0], "B": m[:, 1], "C": m[:, 2], "D": D,
                       "y": np.arange(8, dtype=float)})
    with pytest.raises(ValueError, match="parameters but only .* distinct runs"):
        mfgqc.load(df, measure="y").doe(factors=["A", "B", "C", "D"], order=2)


def test_replicated_rank_deficient_refused_not_silent():
    # replication gives enough rows to pass the engine's n<p guard, but only 8
    # distinct points -> an order-2 (11-param) model is still not estimable.
    m = gen.coded_full_matrix(3)
    D = m[:, 0] * m[:, 1] * m[:, 2]
    base = pd.DataFrame({"A": m[:, 0], "B": m[:, 1], "C": m[:, 2], "D": D})
    df = pd.concat([base, base], ignore_index=True)
    df["y"] = np.arange(16, dtype=float)
    with pytest.raises(ValueError, match="not estimable"):
        mfgqc.load(df, measure="y").doe(factors=["A", "B", "C", "D"], order=2)


# --- 4.2a fraction detection on factors= --------------------------------------
def test_factors_path_detects_oracle4_fraction():
    res = mfgqc.load(SOUP, measure="y").doe(factors=["A", "B", "C", "D", "E"], order=2)
    assert "fractional" in res._title()
    assert res.resolution == 5
    assert res.generators == ("E=ABCD",)
    assert list(res.aliases) == SOUP_ALIAS
    assert res.alias_of["D:E"] == "ABC"
    # the analysis numbers are unchanged by detection
    for term, exp in SOUP_COEF.items():
        got = res.intercept if term == "intercept" else res.coef[term]
        assert abs(got - exp) < 5e-6


def test_full_factorial_not_flagged_fractional():
    res = mfgqc.load(VOLT, measure="y").doe(factors=["A", "B", "C"])
    assert "full" in res._title()
    assert res.resolution is None
    assert res.generators == ()


def test_detected_fraction_surfaces_aliasing_flag():
    res = mfgqc.load(SOUP, measure="y").doe(factors=["A", "B", "C", "D", "E"], order=2)
    alias_flag = next(a for a in res.adequacy if a.name == "aliasing")
    assert alias_flag.passed is False


# --- 4.6 calibrated variance + reliability tempering --------------------------
def _replicated_clean():
    m = gen.coded_full_matrix(3)
    m2 = np.vstack([m, m])
    y = 100 + 5 * m2[:, 0] + np.random.default_rng(3).normal(0, 0.4, m2.shape[0])
    return pd.DataFrame({"A": m2[:, 0], "B": m2[:, 1], "C": m2[:, 2], "y": y})


def test_constant_variance_is_breusch_pagan_and_dispersion_is_separate():
    res = mfgqc.load(_replicated_clean(), measure="y").doe(factors=["A", "B", "C"])
    names = {a.name for a in res.assumptions}
    assert "homoscedasticity" not in names          # the corr(|resid|,fitted) check is gone
    cv = next(a for a in res.assumptions if a.name == "constant_variance")
    assert "Breusch-Pagan" in cv.test               # primary, mean-linked verdict
    assert cv.passed is True                         # clean data -> homoscedastic
    disp = next(a for a in res.assumptions if a.name == "dispersion_effect")
    assert "Brown-Forsythe" in disp.test             # separate, labeled dispersion check
    assert disp.passed is True                       # clean data -> no dispersion effect


def test_breusch_pagan_fires_on_mean_linked_heteroscedasticity():
    # spread grows with the response level -> BP must flag constant_variance.
    m = gen.coded_full_matrix(3)
    m4 = np.vstack([m] * 4)
    rng = np.random.default_rng(7)
    sd = np.where(m4[:, 0] > 0, 4.0, 0.5)            # 8x spread at the high-A (high-response) level
    y = 50 + 10 * m4[:, 0] + rng.normal(0, 1, m4.shape[0]) * sd
    df = pd.DataFrame({"A": m4[:, 0], "B": m4[:, 1], "C": m4[:, 2], "y": y})
    res = mfgqc.load(df, measure="y").doe(factors=["A", "B", "C"])
    cv = next(a for a in res.assumptions if a.name == "constant_variance")
    assert cv.passed is False                         # mean-linked heteroscedasticity detected


def test_dispersion_effect_without_mean_trend_separates_from_BP():
    # a factor drives the variance but NOT the mean: dispersion_effect fires, the
    # mean-linked Breusch-Pagan verdict does not (the two answer different questions).
    m = gen.coded_full_matrix(3)
    m4 = np.vstack([m] * 4)
    rng = np.random.default_rng(7)
    sd = np.where(m4[:, 0] > 0, 4.0, 0.5)            # A drives spread, not the mean
    y = 50 + rng.normal(0, 1, m4.shape[0]) * sd
    df = pd.DataFrame({"A": m4[:, 0], "B": m4[:, 1], "C": m4[:, 2], "y": y})
    res = mfgqc.load(df, measure="y").doe(factors=["A", "B", "C"])
    cv = next(a for a in res.assumptions if a.name == "constant_variance")
    disp = next(a for a in res.assumptions if a.name == "dispersion_effect")
    assert disp.passed is False                       # A drives the variance
    assert cv.passed is True                          # no mean-variance trend


def test_small_design_residual_checks_are_tempered():
    # unreplicated full 2^3: 1 residual df at order 2 -> checks flagged low power,
    # not rendered as confident PASS/FAIL.
    m = gen.coded_full_matrix(3)
    y = 10 + 3 * m[:, 0] + 2 * m[:, 1]
    df = pd.DataFrame({"A": m[:, 0], "B": m[:, 1], "C": m[:, 2], "y": y})
    res = mfgqc.load(df, measure="y").doe(factors=["A", "B", "C"], order=2)
    assert res.df_resid > 0 and res.df_resid < 8
    rels = {a.name: a.reliability for a in res.assumptions}
    assert rels.get("normality") == "low_power"
    assert rels.get("constant_variance") == "low_power"
