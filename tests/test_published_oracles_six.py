"""Published-authority oracles for the six new modules (TEST_ORACLE_SIX_MODULES).

[PUBLISHED] = an external authority (Montgomery / ANSI-ASQ Z1.9) states input->output.
[CROSS-CHECK] = pinned to scipy or a math identity. PUBLISHED preferred where available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mfgqc

# AIAG linearity raw data (doubles as the regression oracle).
_AIAG_LIN = {
    2.00: [2.70, 2.50, 2.40, 2.50, 2.70, 2.30, 2.50, 2.50, 2.40, 2.40, 2.60, 2.40],
    4.00: [5.10, 3.90, 4.20, 5.00, 3.80, 3.90, 3.90, 3.90, 3.90, 4.00, 4.10, 3.80],
    6.00: [5.80, 5.70, 5.90, 5.90, 6.00, 6.10, 6.00, 6.10, 6.40, 6.30, 6.00, 6.10],
    8.00: [7.60, 7.70, 7.80, 7.70, 7.80, 7.80, 7.80, 7.70, 7.80, 7.50, 7.60, 7.70],
    10.00: [9.10, 9.30, 9.50, 9.30, 9.40, 9.50, 9.50, 9.50, 9.60, 9.20, 9.30, 9.40],
}


# ===================== MODULE 1 — attributes charts ===================== #
def test_p_chart_montgomery_ex_7_1_formula():
    # [PUBLISHED] Montgomery Ex 7.1: pbar=0.2313, n=50 -> UCL=0.4102, LCL=0.0524.
    df = pd.DataFrame({"d": [11.565, 11.565], "i": [50, 50]})  # pbar = 11.565/50
    r = mfgqc.load(df, measure="d").control_chart(kind="p", n="i")
    assert r.location_cl == pytest.approx(0.2313, abs=1e-4)
    assert float(r.location_ucl[0]) == pytest.approx(0.4102, abs=1e-3)
    assert float(r.location_lcl[0]) == pytest.approx(0.0524, abs=1e-3)


def test_np_chart_formula_pin():
    # [CROSS-CHECK] n=50, pbar=0.20 -> CL=10, UCL=18.485, LCL=1.515.
    r = mfgqc.load(pd.DataFrame({"d": [10] * 6}), measure="d").control_chart(kind="np", n=50)
    assert r.location_cl == pytest.approx(10.0, abs=1e-6)
    assert float(r.location_ucl[0]) == pytest.approx(18.485, abs=1e-3)
    assert float(r.location_lcl[0]) == pytest.approx(1.515, abs=1e-3)


# ===================== MODULE 2 — Pareto + chi-square ==================== #
def test_pareto_identity():
    # [CROSS-CHECK] deterministic identity (no external published oracle exists).
    r = mfgqc.pareto(pd.Series({"A": 50, "B": 30, "C": 15, "D": 5}))
    assert list(r.cum_pct) == pytest.approx([50.0, 80.0, 95.0, 100.0])
    assert set(r.vital_few) == {"A", "B"}


def test_chi_square_cross_check():
    # [CROSS-CHECK vs scipy] 2x2 no Yates.
    r = mfgqc.contingency([[30, 20], [15, 35]])
    assert r.chi2 == pytest.approx(9.0909, abs=1e-3)
    assert r.p_value == pytest.approx(0.002569, abs=1e-4)
    assert r.dof == 1
    assert r.cramers_v == pytest.approx(0.3015, abs=1e-3)


# ===================== MODULE 3 — regression / ANOVA ==================== #
def test_regression_reproduces_aiag_linearity():
    # [PUBLISHED] bias ~ reference: slope -0.1317, intercept 0.7367, R^2 0.7143.
    rows = [{"ref": r, "bias": v - r} for r, vals in _AIAG_LIN.items() for v in vals]
    reg = mfgqc.load(pd.DataFrame(rows), measure="bias").regress(on="ref")
    assert reg.coef["ref"] == pytest.approx(-0.1317, abs=1e-4)
    assert reg.coef["intercept"] == pytest.approx(0.7367, abs=1e-4)
    assert reg.r_squared == pytest.approx(0.7143, abs=1e-3)
    assert reg.df_resid == 58


def test_ols_sanity():
    # [CROSS-CHECK] x=1..5, y -> slope 1.99, intercept 0.05, R^2 0.9973.
    reg = mfgqc.load(pd.DataFrame({"y": [2.1, 3.9, 6.2, 7.8, 10.1], "x": [1, 2, 3, 4, 5]}),
                    measure="y").regress(on="x")
    assert reg.coef["x"] == pytest.approx(1.99, abs=1e-2)
    assert reg.coef["intercept"] == pytest.approx(0.05, abs=1e-2)
    assert reg.r_squared == pytest.approx(0.9973, abs=1e-3)


def test_one_way_anova_cross_check():
    # [CROSS-CHECK vs scipy] groups -> F=10.400, p=0.00462.
    adf = pd.DataFrame({"y": [1, 2, 3, 4, 2, 3, 4, 5, 5, 6, 7, 8],
                        "g": ["a"] * 4 + ["b"] * 4 + ["c"] * 4})
    row = mfgqc.load(adf, measure="y").anova(factors=["g"]).table["g"]
    assert row["f"] == pytest.approx(10.400, abs=1e-2)
    assert row["p_value"] == pytest.approx(0.00462, abs=2e-4)


def test_anova_reproduces_gage_parts_ss(aiag_qc):
    # [PUBLISHED] one-way anova on 'part' reproduces the gage R&R Parts SS = 88.3619.
    g = aiag_qc.frame
    a = mfgqc.load(g, measure="y").anova(factors=["part"])
    assert a.table["part"]["ss"] == pytest.approx(88.3619, abs=1e-3)


# ===================== MODULE 4 — Z1.9 ================================== #
def test_z19_published_code_letter_and_n():
    # [PUBLISHED] lot 100, level II -> code F -> n=10.
    plan = mfgqc.z19_plan(lot_size=100, aql=1.0, level="II", severity="normal")
    assert plan.code_letter == "F"
    assert plan.n == 10


def test_z19_mechanics_and_normality_guard():
    # [CROSS-CHECK] xbar=50,s=2,LSL=44,USL=56 -> QL=QU=3.0, ~0.135% each tail.
    from scipy import stats
    plan = mfgqc.z19_plan(lot_size=100, aql=1.0)
    z = np.random.default_rng(0).standard_normal(plan.n)
    sample = (z - z.mean()) / z.std(ddof=1) * 2.0 + 50.0
    d = plan.inspect(sample, lower=44.0, upper=56.0)
    assert d.QL == pytest.approx(3.0, abs=1e-6) and d.QU == pytest.approx(3.0, abs=1e-6)
    assert d.est_pct_lower == pytest.approx(stats.norm.sf(3.0), abs=1e-9)
    # [PUBLISHED behavior] normality guard fires on non-normal data
    skew = np.random.default_rng(1).exponential(2.0, plan.n) + 45.0
    bad = plan.inspect(skew, lower=44.0, upper=56.0)
    assert next(a for a in bad.assumptions if a.name == "normality").passed is False


# ===================== MODULE 5 — CUSUM + EWMA (Table 9.1) ============== #
def test_ewma_montgomery_ex_9_2(table_9_1_qc):
    # [PUBLISHED] lambda=0.1, L=2.7, mu0=10, sigma=1 -> z-series + limits + OOC 29,30.
    ew = table_9_1_qc.ewma_chart(lam=0.1, L=2.7, mu0=10.0, sigma=1.0)
    assert float(ew.z[0]) == pytest.approx(9.945, abs=5e-4)
    assert float(ew.z[1]) == pytest.approx(9.7495, abs=5e-4)
    assert float(ew.z[2]) == pytest.approx(9.70355, abs=5e-4)
    assert float(ew.z[28]) == pytest.approx(10.6468, abs=5e-4)
    assert float(ew.z[29]) == pytest.approx(10.6341, abs=5e-4)
    assert float(ew.ucl[-1]) == pytest.approx(10.6189, abs=1e-3)
    assert float(ew.lcl[-1]) == pytest.approx(9.3811, abs=1e-3)
    assert {29, 30} <= {v.point for v in ew.violations}


def test_cusum_montgomery_table_9_1(table_9_1_qc):
    # [PUBLISHED] K=0.5, H=5, mu0=10, sigma=1 -> C+ signals at 29; C- never signals.
    cu = table_9_1_qc.cusum_chart(k=0.5, h=5, mu0=10.0, sigma=1.0)
    assert 29 in {v.point for v in cu.violations}
    # all signals are upper-arm (C+); the lower CUSUM never fires on this upward drift
    assert all(float(cu.c_minus[v.point - 1]) <= cu.H for v in cu.violations)
