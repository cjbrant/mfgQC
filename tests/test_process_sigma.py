"""Slice 2: attributes capability (process sigma). Oracle = the canonical
DPMO-to-sigma table (6210 DPMO at 4.0 sigma short-term, 3.4 at 6.0); exact CI
cross-checked against scipy/statsmodels; zero-defect negative control reports a
one-sided bound, never an infinite sigma.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

import mfgqc

matplotlib.use("Agg")


# --- DPMO-to-sigma anchors ----------------------------------------------------
@pytest.mark.parametrize("dpmo,sigma", [(6210, 4.0), (66807, 3.0), (308537, 2.0), (3.4, 6.0)])
def test_dpmo_to_sigma_table_anchors(dpmo, sigma):
    # a process producing `dpmo` defects per million sits at `sigma` short-term.
    res = mfgqc.process_sigma(defects=dpmo, units=1_000_000, kind="defectives")
    assert abs(res.dpmo - dpmo) < 1.0
    assert abs(res.z_st - sigma) < 0.01, (dpmo, res.z_st, sigma)
    assert abs(res.z_lt - (sigma - 1.5)) < 0.01      # the 1.5 shift is explicit


def test_shift_is_surfaced_not_hidden():
    res = mfgqc.process_sigma(defects=6210, units=1_000_000, kind="defectives")
    rep = res.report()
    assert "1.5 sigma shift is a CONVENTION" in rep
    assert "Z.lt" in rep and "Z.st" in rep
    assert res.shift == 1.5


# --- exact CI cross-checks ----------------------------------------------------
def test_binomial_ci_matches_statsmodels_beta():
    sm = pytest.importorskip("statsmodels.stats.proportion")
    res = mfgqc.process_sigma(defects=12, units=400, kind="defectives")
    lo, hi = sm.proportion_confint(12, 400, alpha=0.05, method="beta")
    assert abs(res.dpmo_ci[0] / 1e6 - lo) < 1e-9
    assert abs(res.dpmo_ci[1] / 1e6 - hi) < 1e-9


def test_poisson_defects_rate_and_yield():
    # 50 defects over 200 units, 5 opportunities each -> exposure 1000.
    res = mfgqc.process_sigma(defects=50, units=200, opportunities=5, kind="defects")
    assert abs(res.dpu - 0.25) < 1e-9
    assert abs(res.dpmo - 50_000) < 1e-6              # 50/1000 * 1e6
    assert abs(res.fty - np.exp(-0.25)) < 1e-9        # Poisson first-time yield
    # exact Poisson CI brackets the point rate
    assert res.dpmo_ci[0] < res.dpmo < res.dpmo_ci[1]


# --- zero-defect negative control --------------------------------------------
def test_zero_defects_reports_bound_not_infinity():
    res = mfgqc.process_sigma(defects=0, units=300, kind="defectives")
    assert res.zero_defect is True
    assert np.isfinite(res.z_st) and np.isfinite(res.z_st_ci[0])
    assert res.one_sided_bound is not None and res.one_sided_bound > 0
    assert "one-sided exact upper bound" in res.report()
    flag = res.assumptions[0]
    assert flag.passed is False                       # rate-stability flag fires


def test_small_sample_is_flagged():
    res = mfgqc.process_sigma(defects=1, units=20, kind="defectives")
    assert res.assumptions[0].reliability == "low_power"


# --- from loaded data ---------------------------------------------------------
def test_attribute_capability_from_dataframe():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"fail": rng.binomial(1, 0.05, 500)})
    res = mfgqc.load(df, measure="fail").attribute_capability()
    assert res.kind == "defectives"
    assert res.units == 500
    assert abs(res.p_hat - df["fail"].mean()) < 1e-9


def test_view_returns_figure():
    res = mfgqc.process_sigma(defects=6210, units=1_000_000, kind="defectives")
    assert res.view() is not None
