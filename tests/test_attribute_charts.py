"""Attribute control charts (p, np, c, u) with the n= sample-size convenience."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mfgqc


def test_p_chart_constant_n_oracle():
    # pbar = 0.20, n = 50 -> UCL 0.3697, LCL 0.0303 (Montgomery attributes example).
    df = pd.DataFrame({"d": [10] * 8, "ins": [50] * 8})  # 10/50 = 0.20
    r = mfgqc.load(df, measure="d").control_chart(kind="p", n="ins")
    assert r.location_cl == pytest.approx(0.20, abs=1e-6)
    assert float(r.location_ucl[0]) == pytest.approx(0.3697, abs=1e-3)
    assert float(r.location_lcl[0]) == pytest.approx(0.0303, abs=1e-3)


def test_p_chart_variable_n_stepped_limits():
    df = pd.DataFrame({"d": [10, 10, 10], "ins": [50, 100, 200]})
    r = mfgqc.load(df, measure="d").control_chart(kind="p", n="ins")
    ucls = [float(x) for x in r.location_ucl]
    assert ucls[0] > ucls[1] > ucls[2]   # limits tighten as n grows (stepped)


def test_c_chart_oracle_and_lcl_floor():
    r = mfgqc.load(pd.DataFrame({"c": [19.85] * 4}), measure="c").control_chart(kind="c")
    assert r.location_cl == pytest.approx(19.85, abs=1e-6)
    assert float(r.location_ucl[0]) == pytest.approx(33.216, abs=1e-3)
    assert float(r.location_lcl[0]) == pytest.approx(6.484, abs=1e-3)
    # LCL floors at 0 when the formula goes negative
    r2 = mfgqc.load(pd.DataFrame({"c": [2.0] * 4}), measure="c").control_chart(kind="c")
    assert float(r2.location_lcl[0]) == 0.0


def test_np_requires_constant_n():
    df = pd.DataFrame({"d": [3, 5, 4], "ins": [50, 100, 50]})
    with pytest.raises(ValueError, match="constant"):
        mfgqc.load(df, measure="d").control_chart(kind="np", n="ins")


def test_u_chart_variable_area_runs():
    df = pd.DataFrame({"defects": [5, 8, 3, 6, 10], "units": [10, 12, 8, 9, 15]})
    r = mfgqc.load(df, measure="defects").control_chart(kind="u", n="units")
    assert r.kind == "u"
    assert len(r.location_ucl) == 5


def test_attribute_dispersion_assumption_present():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"d": rng.integers(5, 16, 25), "ins": [100] * 25})
    r = mfgqc.load(df, measure="d").control_chart(kind="p", n="ins")
    assert any(a.name == "dispersion" for a in r.assumptions)


def test_n_as_constant_int():
    df = pd.DataFrame({"d": [10] * 6})
    r = mfgqc.load(df, measure="d").control_chart(kind="p", n=50)
    assert float(r.location_ucl[0]) == pytest.approx(0.3697, abs=1e-3)
