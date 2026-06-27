"""Revision 2 fixes (post blind-validation v1):

1. fractional_factorial infers `fraction` from `generators` (was wrongly required).
2. .doe() refuses an out-of-range / 3+-level factor instead of silently coding it.
3. every view plots its data (residuals panel is real; saturated states why).
4. DOE static views default to the light theme (white background).
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

import mfgqc
from mfgqc.doe import generate as gen

matplotlib.use("Agg")


# --- Fix 1: fraction inferred from generators ---------------------------------
def test_generators_without_fraction():
    d = mfgqc.design.fractional_factorial(
        factors=["A", "B", "C", "D", "E", "F"], generators=["E=ABC", "F=ABD"])
    assert d.kind == "fractional"
    assert d.generators == ("E=ABC", "F=ABD")
    assert d.resolution == 4               # 2^(6-2), shortest defining word ABCE/ABDF/CDEF


def test_fraction_only_picks_min_aberration():
    d = mfgqc.design.fractional_factorial(["A", "B", "C", "D", "E"], fraction="1/2")
    assert d.generators == ("E=ABCD",)
    assert d.resolution == 5


def test_fraction_contradicting_generators_raises():
    with pytest.raises(ValueError, match="contradict"):
        mfgqc.design.fractional_factorial(["A", "B", "C", "D", "E"],
                                         generators=["E=ABCD"], fraction="1/4")


def test_neither_generators_nor_fraction_raises():
    with pytest.raises(ValueError, match="either generators"):
        mfgqc.design.fractional_factorial(["A", "B", "C", "D"])


# --- Fix 2: factor-structure guard --------------------------------------------
def _design_df(k):
    m = gen.coded_full_matrix(k)
    cols = {chr(ord("A") + j): m[:, j] for j in range(k)}
    cols["y"] = np.arange(m.shape[0], dtype=float) + 1
    return pd.DataFrame(cols)


def test_out_of_range_factor_is_refused():
    df = _design_df(3)
    df.loc[2, "A"] = 2                       # contaminate A: levels become {-1, 1, 2}
    with pytest.raises(ValueError, match="not a clean two-level factor"):
        mfgqc.load(df, measure="y").doe(factors=["A", "B", "C"])


def test_clean_two_level_passes_guard():
    df = _design_df(3)
    res = mfgqc.load(df, measure="y").doe(factors=["A", "B", "C"], order=2)
    assert res.n == 8


def test_center_points_pass_guard():
    # 2^2 plus center points: factors carry {-1, 0, 1}, 0 = exact midpoint -> allowed.
    m = gen.coded_full_matrix(2)
    rows = [(r[0], r[1], 10 + r[0]) for r in m] * 2 + [(0.0, 0.0, 10.0)] * 3
    df = pd.DataFrame(rows, columns=["A", "B", "y"])
    res = mfgqc.load(df, measure="y").doe(factors=["A", "B"])
    assert res.curvature is not None         # center points recognised, not refused


# --- Fix 3: views plot data ---------------------------------------------------
def test_residuals_view_plots_two_panels():
    m = gen.coded_full_matrix(3)
    m2 = np.vstack([m, m])
    y = 100 + 5 * m2[:, 0] + np.random.default_rng(0).normal(0, 0.5, m2.shape[0])
    df = pd.DataFrame({"A": m2[:, 0], "B": m2[:, 1], "C": m2[:, 2], "y": y})
    res = mfgqc.load(df, measure="y").doe(factors=["A", "B", "C"])
    fig = res.view(kind="residuals")
    assert len(fig.axes) == 2                # resid-vs-fitted AND normal QQ
    # the residual scatter actually has points plotted
    assert any(len(ax.collections) > 0 for ax in fig.axes)


def test_interaction_surface_view_renders():
    # Section 4.8 lists interaction_surface in the per-view draw contract; it must
    # draw (not error with 'unknown kind').
    d = mfgqc.design.fractional_factorial(["A", "B", "C", "D", "E"], fraction="1/2")
    m = d.matrix
    rng = np.random.default_rng(0)
    y = 5 + 2 * m[:, 2] + 3 * m[:, 4] + 2.5 * m[:, 2] * m[:, 4] + rng.normal(0, 0.2, m.shape[0])
    df = pd.DataFrame({f: m[:, i] for i, f in enumerate("ABCDE")}); df["y"] = y
    res = mfgqc.load(df, measure="y").doe(design=d, order=2)
    fig = res.view(kind="interaction_surface")
    assert fig.axes and min(fig.get_facecolor()[:3]) > 0.9


def test_saturated_residuals_states_why_not_empty():
    df = _design_df(4)
    res = mfgqc.load(df, measure="y").doe(factors=["A", "B", "C", "D"])
    assert res.fit_kind == "saturated"
    fig = res.view(kind="residuals")
    txt = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "residual degrees of freedom" in txt


# --- Fix 4: light default theme -----------------------------------------------
def test_doe_views_default_to_light():
    df = _design_df(3)
    res = mfgqc.load(df, measure="y").doe(factors=["A", "B", "C"], order=2)
    for kind in ("main_effects", "pareto", "halfnormal"):
        fig = res.view(kind=kind)
        assert min(fig.get_facecolor()[:3]) > 0.9, (kind, fig.get_facecolor())


def test_doe_light_does_not_change_global_default():
    df = _design_df(3)
    mfgqc.load(df, measure="y").doe(factors=["A", "B", "C"], order=2).view(kind="pareto")
    from mfgqc import palette
    assert palette.active().name == "phosphor"   # global default untouched
