"""Additional control-chart types: I-MR and attribute charts (p, np, c, u).

These are formula-reproduction tests: the expected limits are computed
independently in the test from the documented formulas/constants, so a wrong
constant or formula in the implementation is caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mfgqc


def test_i_mr_inferred_and_limits():
    x = np.array([10.1, 9.8, 10.3, 10.0, 9.7, 10.2, 9.9, 10.4, 10.1, 9.6,
                  10.0, 10.2, 9.8, 10.1, 9.9])
    qc = mfgqc.load(pd.DataFrame({"x": x}), measure="x", subgroup_size=1)
    cc = qc.control_chart()  # should infer i_mr (all subgroups size 1)

    assert cc.kind == "i_mr"
    assert cc.inferred is True

    mr = np.abs(np.diff(x))
    mrbar = mr.mean()
    d2 = 1.128  # n=2
    expected_ucl = x.mean() + 3 * mrbar / d2
    expected_lcl = x.mean() - 3 * mrbar / d2

    assert cc.location_cl == pytest.approx(x.mean())
    assert float(np.unique(cc.location_ucl)[0]) == pytest.approx(expected_ucl, rel=1e-6)
    assert float(np.unique(cc.location_lcl)[0]) == pytest.approx(expected_lcl, rel=1e-6)
    # MR chart: CL = MRbar, UCL = D4(2) * MRbar = 3.267 * MRbar, LCL = 0
    assert cc.disp_label == "MR"
    assert cc.disp_cl == pytest.approx(mrbar)
    assert float(np.unique(cc.disp_ucl)[0]) == pytest.approx(3.267 * mrbar, rel=1e-6)


def test_p_chart_variable_limits():
    defects = np.array([5, 8, 6, 7, 9, 4, 6, 10, 5, 7])
    n = np.array([100, 120, 100, 110, 130, 90, 100, 140, 100, 110])
    qc = mfgqc.load(
        pd.DataFrame({"defects": defects, "n": n}),
        measure="defects", roles={"size": "n"},
    )
    cc = qc.control_chart(kind="p")
    pbar = defects.sum() / n.sum()
    assert cc.location_cl == pytest.approx(pbar)
    assert cc.location_points == pytest.approx(defects / n)
    exp_ucl0 = pbar + 3 * np.sqrt(pbar * (1 - pbar) / n[0])
    assert cc.location_ucl[0] == pytest.approx(exp_ucl0)
    # varying n => varying limits
    assert len(np.unique(cc.location_ucl)) > 1


def test_np_chart_constant_n():
    defects = np.array([3, 5, 2, 4, 6, 3, 4, 5, 2, 4])
    n = np.full(10, 50)
    qc = mfgqc.load(
        pd.DataFrame({"d": defects, "n": n}), measure="d", roles={"size": "n"})
    cc = qc.control_chart(kind="np")
    pbar = defects.sum() / n.sum()
    assert cc.location_cl == pytest.approx(50 * pbar)
    assert cc.location_ucl[0] == pytest.approx(50 * pbar + 3 * np.sqrt(50 * pbar * (1 - pbar)))


def test_c_chart():
    counts = np.array([7, 10, 6, 8, 12, 5, 9, 11, 7, 8])
    qc = mfgqc.load(pd.DataFrame({"c": counts}), measure="c")
    cc = qc.control_chart(kind="c")
    cbar = counts.mean()
    assert cc.location_cl == pytest.approx(cbar)
    assert cc.location_ucl[0] == pytest.approx(cbar + 3 * np.sqrt(cbar))
    assert cc.location_lcl[0] == pytest.approx(max(0.0, cbar - 3 * np.sqrt(cbar)))


def test_u_chart_variable_limits():
    counts = np.array([12, 15, 9, 20, 14, 11, 18, 10])
    units = np.array([10, 12, 8, 15, 11, 9, 14, 10], dtype=float)
    qc = mfgqc.load(
        pd.DataFrame({"c": counts, "u": units}), measure="c", roles={"size": "u"})
    cc = qc.control_chart(kind="u")
    ubar = counts.sum() / units.sum()
    assert cc.location_cl == pytest.approx(ubar)
    assert cc.location_points == pytest.approx(counts / units)
    assert cc.location_ucl[0] == pytest.approx(ubar + 3 * np.sqrt(ubar / units[0]))


def test_within_subgroup_outlier_flags_on_range_chart():
    """Bug 1: a single inflated reading must trip the R chart even if the mean is in-limits."""
    base = np.random.default_rng(0).normal(10, 1, (25, 5))
    base[11, 0] = 16.0  # subgroup 12, reading 1
    df = pd.DataFrame([{"subgroup": i + 1, "x": v}
                       for i, row in enumerate(base) for v in row])
    qc = mfgqc.load(df, measure="x", roles={"subgroup": "subgroup"}, subgroup_size=5)
    cc = qc.control_chart()
    disp_violations = [v for v in cc.violations if v.chart == "dispersion"]
    assert any(v.point == 12 for v in disp_violations)


def test_np_requires_constant_n():
    df = pd.DataFrame({"d": [1, 2, 3], "n": [10, 20, 30]})
    qc = mfgqc.load(df, measure="d", roles={"size": "n"})
    with pytest.raises(ValueError, match="constant subgroup size"):
        qc.control_chart(kind="np")
