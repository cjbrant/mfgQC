"""Z1.9 variables acceptance sampling (standard-deviation method, sigma unknown)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import mfgqc


def _sample_with(xbar, s, n=20, seed=0):
    """A sample standardized to exactly the given mean and sample SD."""
    z = np.random.default_rng(seed).standard_normal(n)
    z = (z - z.mean()) / z.std(ddof=1)
    return z * s + xbar


def test_z19_mechanics_oracle():
    # xbar=50, s=2, LSL=44, USL=56 -> QL=QU=3.0; ~0.135% nonconforming each tail.
    plan = mfgqc.z19_plan(lot_size=200, aql=1.0)
    d = plan.inspect(_sample_with(50.0, 2.0, n=plan.n), lower=44.0, upper=56.0)
    assert d.QL == pytest.approx(3.0, abs=1e-6)
    assert d.QU == pytest.approx(3.0, abs=1e-6)
    assert d.est_pct_lower == pytest.approx(stats.norm.sf(3.0), abs=1e-9)  # ~0.00135
    assert d.est_pct_total == pytest.approx(2 * stats.norm.sf(3.0), abs=1e-9)


def test_z19_code_letter_and_sample_size():
    # [PUBLISHED] Z1.9 level II: lot 100 -> code F -> n=10.
    assert mfgqc.z19_plan(lot_size=100, aql=1.0).code_letter == "F"
    assert mfgqc.z19_plan(lot_size=100, aql=1.0).n == 10
    assert mfgqc.z19_plan(lot_size=200, aql=1.0).code_letter == "G"
    assert mfgqc.z19_plan(lot_size=200, aql=1.0).n == 15


def test_z19_k_matches_design_table_midrange():
    # k via the standard normal-approximation design reproduces published Z1.9
    # normal-inspection k for mid-range code letters (AQL 1.0%).
    f = mfgqc.z19_plan(lot_size=100, aql=1.0)    # code F, n=10
    assert f.n == 10 and f.k == pytest.approx(1.33, abs=0.03)
    h = mfgqc.z19_plan(lot_size=300, aql=1.0)    # code H, n=20
    assert h.n == 20 and h.k == pytest.approx(1.62, abs=0.03)
    ll = mfgqc.z19_plan(lot_size=5000, aql=1.0)  # code L, n=50
    assert ll.n == 50 and ll.k == pytest.approx(1.88, abs=0.03)


def test_z19_accept_reject_logic():
    plan = mfgqc.z19_plan(lot_size=200, aql=1.0)   # k ~ 1.62
    # well inside the limits -> Q large -> accept
    assert plan.inspect(_sample_with(50.0, 2.0, n=plan.n), lower=44.0, upper=56.0).decision == "accept"
    # mean near the upper limit -> QU small (< k) -> reject
    assert plan.inspect(_sample_with(55.0, 2.0, n=plan.n), lower=44.0, upper=56.0).decision == "reject"


def test_z19_limits_carried_on_plan():
    # Bug 2 fix: limits attach to the plan; inspect() falls back to them.
    plan = mfgqc.z19_plan(lot_size=100, aql=1.0, lower=44.0, upper=56.0)
    assert plan.lower == 44.0 and plan.upper == 56.0
    d = plan.inspect(_sample_with(50.0, 2.0, n=plan.n))   # no limits passed to inspect
    assert d.QL == pytest.approx(3.0, abs=1e-6) and d.QU == pytest.approx(3.0, abs=1e-6)
    # inspect() args override the plan's limits
    d2 = plan.inspect(_sample_with(50.0, 2.0, n=plan.n), lower=40.0)
    assert d2.QL == pytest.approx(5.0, abs=1e-6)


def test_z19_one_sided():
    plan = mfgqc.z19_plan(lot_size=200, aql=1.0)
    d = plan.inspect(_sample_with(50.0, 2.0, n=plan.n), lower=44.0)
    assert d.QU is None and d.QL == pytest.approx(3.0, abs=1e-6)


def test_z19_normality_flag_fires_on_nonnormal():
    plan = mfgqc.z19_plan(lot_size=200, aql=1.0)
    rng = np.random.default_rng(1)
    skewed = rng.exponential(2.0, plan.n) + 45.0    # clearly non-normal
    d = plan.inspect(skewed, lower=44.0, upper=56.0)
    norm = next(a for a in d.assumptions if a.name == "normality")
    assert norm.passed is False
    assert "normal" in (norm.recommendation or "").lower()


def test_z19_summary_flat_and_view():
    import matplotlib.figure as mfig
    plan = mfgqc.z19_plan(lot_size=200, aql=1.0)
    s = plan.summary()
    assert {"n", "code_letter", "k", "M", "aql"} <= set(s)
    assert all(not isinstance(v, (list, dict)) for v in s.values())
    d = plan.inspect(_sample_with(50.0, 2.0, n=plan.n), lower=44.0, upper=56.0)
    assert isinstance(d.view(), mfig.Figure)
    assert all(not isinstance(v, (list, dict)) for v in d.summary().values())


def test_z19_inspect_needs_a_limit():
    plan = mfgqc.z19_plan(lot_size=200, aql=1.0)
    with pytest.raises(ValueError, match="at least one spec limit"):
        plan.inspect(_sample_with(50.0, 2.0, n=plan.n))
