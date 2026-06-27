"""Slice 1: sample size and power. Oracle = canonical Cohen / Montgomery anchors;
statsmodels power as a secondary cross-check; negative control that solving for n
then back-solving for power returns the input.
"""

from __future__ import annotations

import math

import matplotlib
import numpy as np
import pytest

import mfgqc

matplotlib.use("Agg")
smp = pytest.importorskip("statsmodels.stats.power")


# --- t-test ------------------------------------------------------------------
def test_two_sample_n_matches_cohen_anchor():
    # Cohen / Montgomery anchor: d=0.5, alpha=0.05 two-sided, power=0.80 -> n~64/group.
    res = mfgqc.power.t_test(effect=0.5, power=0.80)
    assert res.solved_for == "n"
    assert math.ceil(res.n) == 64
    assert abs(res.n - 63.77) < 0.5            # secondary: statsmodels value


def test_two_sample_n_matches_statsmodels():
    for d, pw in [(0.3, 0.80), (0.8, 0.90), (0.5, 0.95)]:
        res = mfgqc.power.t_test(effect=d, power=pw)
        sm = smp.TTestIndPower().solve_power(effect_size=d, power=pw, alpha=0.05,
                                             alternative="two-sided")
        assert abs(res.n - sm) < 0.05, (d, pw, res.n, sm)


def test_one_sample_and_paired():
    r1 = mfgqc.power.t_test(effect=0.5, power=0.80, kind="one-sample")
    sm = smp.TTestPower().solve_power(effect_size=0.5, power=0.80, alpha=0.05,
                                      alternative="two-sided")
    assert abs(r1.n - sm) < 0.05
    rp = mfgqc.power.t_test(effect=0.5, power=0.80, kind="paired")
    assert abs(rp.n - r1.n) < 1e-6            # paired == one-sample on differences


def test_solve_for_power_and_effect():
    p = mfgqc.power.t_test(effect=0.5, n=64)
    assert p.solved_for == "power" and abs(p.power - 0.80) < 0.01
    e = mfgqc.power.t_test(n=64, power=0.80)
    assert e.solved_for == "effect" and abs(e.effect - 0.5) < 0.01


# --- ANOVA -------------------------------------------------------------------
def test_anova_n_matches_statsmodels_total():
    # f=0.25, k=4, power=0.80 -> total N ~ 178 (statsmodels nobs is total).
    res = mfgqc.power.anova(groups=4, effect=0.25, power=0.80)
    sm_total = smp.FTestAnovaPower().solve_power(effect_size=0.25, k_groups=4,
                                                 power=0.80, alpha=0.05)
    assert abs(res.n_total - sm_total) < 0.1
    assert abs(res.n - sm_total / 4) < 0.1


# --- proportion (approximation) ----------------------------------------------
def test_proportion_is_flagged_approximate():
    res = mfgqc.power.proportion(p1=0.10, p2=0.20, power=0.80)
    assert res.approximate is True
    from statsmodels.stats.proportion import proportion_effectsize
    sm = smp.NormalIndPower().solve_power(
        effect_size=proportion_effectsize(0.10, 0.20), power=0.80, alpha=0.05)
    assert abs(res.n - sm) / sm < 0.20         # same ballpark as the arcsine form


# --- variance ----------------------------------------------------------------
def test_variance_power_solves():
    res = mfgqc.power.variance(ratio=2.0, n=50)
    assert 0 < res.power < 1
    n = mfgqc.power.variance(ratio=2.0, power=res.power)
    assert abs(n.n - 50) < 1.0                 # round-trips


# --- negative controls / guards ----------------------------------------------
def test_solve_n_then_backsolve_power_roundtrips():
    a = mfgqc.power.t_test(effect=0.4, power=0.85)
    b = mfgqc.power.t_test(effect=0.4, n=a.n)
    assert abs(b.power - 0.85) < 1e-6


def test_zero_or_two_nones_raises():
    with pytest.raises(ValueError, match="exactly one"):
        mfgqc.power.t_test(effect=0.5)                      # two None (n, power)
    with pytest.raises(ValueError, match="exactly one"):
        mfgqc.power.t_test(effect=0.5, n=64, power=0.80)    # zero None


def test_unreachable_power_raises_legibly():
    with pytest.raises(ValueError, match="not reachable|cannot solve"):
        mfgqc.power.t_test(effect=0.5, power=0.02)           # below alpha floor


def test_report_and_view():
    res = mfgqc.power.t_test(effect=0.5, power=0.80)
    assert "minimum" not in res.report().lower()           # solved for n, not effect
    assert "solved for: n" in res.report()
    fig = res.view(kind="power_curve")
    assert fig is not None
    edge = mfgqc.power.t_test(n=64, power=0.80)
    assert "minimum detectable" in edge.report().lower()
