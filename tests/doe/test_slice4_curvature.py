"""Slice 4: center points, curvature / lack-of-fit, residual diagnostics. No
Lawson oracle; pinned to a statsmodels cross-check (Tier 2, secondary) and to
negative controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mfgqc
from mfgqc.doe import generate as gen


def _design_with_center(curve_shift, seed):
    rng = np.random.default_rng(seed)
    m = gen.coded_full_matrix(2)
    rows = []
    for _ in range(2):                       # replicate factorial corners
        for r in m:
            rows.append((r[0], r[1], 50 + 3 * r[0] + 2 * r[1] + rng.normal(0, 0.3)))
    for _ in range(4):                       # replicated center points
        rows.append((0.0, 0.0, 50 + curve_shift + rng.normal(0, 0.3)))
    return pd.DataFrame(rows, columns=["A", "B", "y"])


def test_curvature_detected_when_present():
    df = _design_with_center(curve_shift=10.0, seed=7)
    res = mfgqc.load(df, measure="y").doe(factors=["A", "B"])
    cv = res.curvature
    assert cv is not None and cv.passed is False
    assert "response-surface" in cv.recommendation


def test_no_curvature_when_planar():
    df = _design_with_center(curve_shift=0.0, seed=3)
    res = mfgqc.load(df, measure="y").doe(factors=["A", "B"])
    assert res.curvature.passed is True


def test_curvature_SS_matches_statsmodels_tier2():
    # Tier 2 (secondary): the center-point curvature SS equals the sequential SS
    # of a center indicator added to the factorial model in statsmodels OLS.
    sm = pytest.importorskip("statsmodels.api")
    df = _design_with_center(curve_shift=6.0, seed=21)
    res = mfgqc.load(df, measure="y").doe(factors=["A", "B"])
    ss_curv = res.curvature.magnitude

    y = df["y"].to_numpy()
    center = (df["A"] == 0) & (df["B"] == 0)
    # base model: factorial mains; augmented adds the center indicator
    Xb = sm.add_constant(np.column_stack([df["A"], df["B"]]))
    Xa = sm.add_constant(np.column_stack([df["A"], df["B"], center.astype(float)]))
    ssr_b = sm.OLS(y, Xb).fit().ssr
    ssr_a = sm.OLS(y, Xa).fit().ssr
    ss_indicator = ssr_b - ssr_a            # sequential SS of the curvature term
    assert abs(ss_curv - ss_indicator) < 1e-6


def test_center_points_give_pure_error_flag():
    df = _design_with_center(curve_shift=0.0, seed=9)
    res = mfgqc.load(df, measure="y").doe(factors=["A", "B"])
    cp = next(a for a in res.adequacy if a.name == "center_points")
    assert cp.passed is True                # curvature is checkable
