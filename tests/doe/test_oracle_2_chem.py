"""Slice 2 gate: unreplicated 2^4 (Lawson chem). 16 coefficients to 4 dp, the
no-pure-error surface, and the Lenth-active set {A, B, A:B}.

ME/SME are Tier-2 (cross-checked against BsMD::LenthPlot separately). The binary
active flag is keyed to SME (experiment-wise); A:C:D is the one borderline effect
that falls in the ME..SME 'possibly active' band, which is surfaced not dropped.
"""

from __future__ import annotations

import numpy as np

import mfgqc
from ._oracles import CHEM, CHEM_COEF


def _res():
    return mfgqc.load(CHEM, measure="y").doe(factors=["A", "B", "C", "D"])


def test_saturated_no_pure_error():
    res = _res()
    assert res.fit_kind == "saturated"
    assert res.df_resid == 0
    assert abs(res.r_squared - 1.0) < 1e-9
    # SE/t/p are NaN (no error term fabricated)
    assert not np.isfinite(res.se["A"])
    assert not np.isfinite(res.p_value["A"])
    pe = next(a for a in res.adequacy if a.name == "pure_error")
    assert pe.passed is False and "Lenth" in pe.recommendation


def test_sixteen_coefficients_match_to_4dp():
    res = _res()
    assert abs(res.intercept - CHEM_COEF["intercept"]) < 5e-5
    for term, exp in CHEM_COEF.items():
        if term == "intercept":
            continue
        assert abs(res.coef[term] - exp) < 5e-5, (term, res.coef[term], exp)


def test_active_set_is_A_B_AB():
    res = _res()
    assert set(res.active) == {"A", "B", "A:B"}


def test_borderline_effect_surfaced_as_possibly_active():
    res = _res()
    # A:C:D clears ME but not SME: surfaced rather than silently promoted/dropped.
    assert "A:C:D" in res.possibly_active
    assert "A:C:D" not in res.active


def test_lenth_quantities_present():
    res = _res()
    assert res.lenth is not None
    assert res.lenth.pse > 0 and res.lenth.me < res.lenth.sme
