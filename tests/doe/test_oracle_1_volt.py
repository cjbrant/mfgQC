"""Slice 1 gate: replicated 2^3 (Lawson volt). Coefficient table to 4 dp, the
replicated t/F path, and the significant set {A, A:C}."""

from __future__ import annotations

import numpy as np

import mfgqc
from ._oracles import VOLT, VOLT_COEF, VOLT_SE, VOLT_T


def _res():
    return mfgqc.load(VOLT, measure="y").doe(factors=["A", "B", "C"])


def test_replicated_fit_has_pure_error():
    res = _res()
    assert res.fit_kind == "replicated"
    assert res.df_resid == 8


def test_coefficients_match_to_4dp():
    res = _res()
    assert abs(res.intercept - VOLT_COEF["intercept"]) < 5e-5
    for term, exp in VOLT_COEF.items():
        if term == "intercept":
            continue
        assert abs(res.coef[term] - exp) < 5e-5, (term, res.coef[term], exp)


def test_effect_is_twice_coefficient():
    res = _res()
    assert abs(res.effect["A"] - 2 * VOLT_COEF["A"]) < 5e-5
    assert abs(res.effect["A"] - (-33.625)) < 5e-4


def test_standard_errors_and_t_match():
    res = _res()
    for term, t_exp in VOLT_T.items():
        assert abs(res.se[term] - VOLT_SE) < 5e-4, (term, res.se[term])
        assert abs(res.t[term] - t_exp) < 5e-3, (term, res.t[term], t_exp)


def test_significant_set_is_A_and_AC():
    res = _res()
    assert set(res.significant) == {"A", "A:C"}


def test_view_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    fig = _res().view()
    assert fig is not None
    fig2 = _res().view(kind="main_effects")
    assert fig2 is not None
