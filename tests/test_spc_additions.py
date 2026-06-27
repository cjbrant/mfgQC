"""Slice 5: SPC additions (Xbar-S inference, pre-control, short-run). Xbar-S
limits cross-checked against the control-constant math; pre-control pins to the
standard rule set and negative controls (a centered capable stream qualifies and
runs; an off-center stream is flagged)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

import mfgqc

matplotlib.use("Agg")


# --- Xbar-S inference (already in the engine; verify it selects + states) -----
def test_large_subgroups_infer_xbar_s():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(10, 1, 12 * 25),
                       "sg": np.repeat(np.arange(25), 12)})
    cc = mfgqc.load(df, measure="x", subgroup="sg").control_chart()
    assert cc.kind == "xbar_s" and cc.inferred is True


def test_small_subgroups_infer_xbar_r():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(10, 1, 5 * 25),
                       "sg": np.repeat(np.arange(25), 5)})
    cc = mfgqc.load(df, measure="x", subgroup="sg").control_chart()
    assert cc.kind == "xbar_r"


def test_xbar_s_limits_match_constant_math():
    from mfgqc.constants import control_constant
    rng = np.random.default_rng(1)
    n, ng = 12, 20
    df = pd.DataFrame({"x": rng.normal(50, 2, n * ng), "sg": np.repeat(np.arange(ng), n)})
    cc = mfgqc.load(df, measure="x", subgroup="sg").control_chart(kind="xbar_s")
    sub = df.groupby("sg")["x"]
    grand = sub.mean().mean()
    sbar = sub.std(ddof=1).mean()
    a3 = control_constant("A3", n)
    assert abs(cc.location_ucl[0] - (grand + a3 * sbar)) < 1e-6
    assert abs(cc.location_cl - grand) < 1e-9


# --- pre-control --------------------------------------------------------------
def test_precontrol_zones_and_qualification():
    # capable, centered stream: spec [40,60], target 50, sd 2 -> Cpk ~ 1.6, qualifies.
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.normal(50, 2, 40)})
    res = mfgqc.load(df, measure="x").spec(lower=40, upper=60, target=50).precontrol()
    assert abs(res.pc_lower - 45) < 1e-9 and abs(res.pc_upper - 55) < 1e-9   # central half
    assert res.qualified is True
    assert res.assumptions[0].passed is True                                  # capable
    assert res.summary()["n_green"] > res.summary()["n_yellow"]


def test_precontrol_flags_incapable_process():
    # wide spread vs spec -> Cpk < 1.33 -> capability-prerequisite flag fires.
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.normal(50, 7, 40)})
    res = mfgqc.load(df, measure="x").spec(lower=40, upper=60, target=50).precontrol()
    flag = res.assumptions[0]
    assert flag.passed is False and "capability is not established" in flag.recommendation


def test_precontrol_red_triggers_stop():
    vals = [50, 50, 50, 50, 50, 65, 50]          # one out-of-spec (red) after qualifying
    df = pd.DataFrame({"x": vals})
    res = mfgqc.load(df, measure="x").spec(lower=40, upper=60, target=50).precontrol()
    assert any(a == "STOP" for _, a, _ in res.dispositions)
    assert res.zones[5] == "red_high"


def test_precontrol_requires_spec():
    df = pd.DataFrame({"x": [1.0, 2, 3]})
    with pytest.raises(ValueError, match="spec limits"):
        mfgqc.load(df, measure="x").precontrol()


# --- short-run / standardized -------------------------------------------------
def test_short_run_pools_parts_on_one_scale():
    rng = np.random.default_rng(4)
    # three part numbers at very different levels; standardized they share a chart.
    df = pd.concat([
        pd.DataFrame({"y": rng.normal(100, 2, 20), "part": "P1"}),
        pd.DataFrame({"y": rng.normal(5, 0.1, 20), "part": "P2"}),
        pd.DataFrame({"y": rng.normal(2000, 50, 20), "part": "P3"})])
    cc = mfgqc.load(df, measure="y").short_run_chart(by="part")
    assert cc.kind == "short_run"
    assert abs(cc.location_cl) < 1e-9 and abs(cc.location_ucl[0] - 3.0) < 1e-9
    assert abs(np.mean(cc.location_points)) < 0.5         # standardized, centered ~0


def test_short_run_flags_heterogeneous_spread():
    rng = np.random.default_rng(5)
    df = pd.concat([
        pd.DataFrame({"y": rng.normal(100, 1, 30), "part": "P1"}),
        pd.DataFrame({"y": rng.normal(100, 12, 30), "part": "P2"})])
    cc = mfgqc.load(df, measure="y").short_run_chart(by="part")
    homo = next(a for a in cc.assumptions if a.name == "homogeneity_of_variance")
    assert homo.passed is False                           # spreads differ across parts


def test_view_renders():
    df = pd.DataFrame({"x": np.random.default_rng(2).normal(50, 2, 40)})
    res = mfgqc.load(df, measure="x").spec(lower=40, upper=60, target=50).precontrol()
    assert res.view() is not None
