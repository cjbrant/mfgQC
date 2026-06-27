"""Slice 4: multiple comparisons and remaining non-parametrics. Oracle =
statsmodels Tukey (which reproduces Montgomery) and scipy Friedman/Mood; negative
control that post-hoc on no-difference data finds no significant pairs.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

import mfgqc

matplotlib.use("Agg")


def _three_groups(rng, shifts=(0, 0, 0), sd=1.0, n=20):
    data = []
    for i, s in enumerate(shifts):
        for v in rng.normal(50 + s, sd, n):
            data.append({"y": v, "grp": f"L{i+1}"})
    return pd.DataFrame(data)


# --- routing -----------------------------------------------------------------
def test_routes_to_tukey_when_equal_variance_normal():
    df = _three_groups(np.random.default_rng(0), shifts=(0, 2, 4))
    res = mfgqc.load(df, measure="y").anova(factors=["grp"]).posthoc()
    assert res.method == "tukey"
    assert "Tukey" in res.family


def test_routes_to_games_howell_when_unequal_variance():
    rng = np.random.default_rng(1)
    df = pd.concat([
        pd.DataFrame({"y": rng.normal(50, 1, 30), "grp": "L1"}),
        pd.DataFrame({"y": rng.normal(52, 6, 30), "grp": "L2"}),
        pd.DataFrame({"y": rng.normal(48, 1, 30), "grp": "L3"})])
    res = mfgqc.load(df, measure="y").anova(factors=["grp"]).posthoc()
    assert res.method == "games-howell"


def test_kruskal_route_gives_dunn():
    rng = np.random.default_rng(2)
    a = mfgqc.test_anova(rng.lognormal(0, 0.6, 25), rng.lognormal(0.4, 0.6, 25),
                        rng.lognormal(0.1, 0.6, 25))
    assert "kruskal" in a.test_used.lower()
    res = a.posthoc()
    assert res.method == "dunn"


def test_control_gives_dunnett():
    df = _three_groups(np.random.default_rng(3), shifts=(0, 3, 5))
    res = mfgqc.load(df, measure="y").anova(factors=["grp"]).posthoc(control="L1")
    assert res.method == "dunnett"
    assert all(p.b == "L1" for p in res.pairs)


# --- correctness vs statsmodels ----------------------------------------------
def test_tukey_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.multicomp")
    df = _three_groups(np.random.default_rng(5), shifts=(0, 2, 5), n=25)
    res = mfgqc.load(df, measure="y").anova(factors=["grp"]).posthoc(method="tukey")
    tk = sm.pairwise_tukeyhsd(df["y"], df["grp"], alpha=0.05)
    sm_p = dict(zip([tuple(sorted(g)) for g in zip(tk.groupsunique[tk._multicomp.pairindices[0]],
                                                   tk.groupsunique[tk._multicomp.pairindices[1]])],
                    tk.pvalues))
    for p in res.pairs:
        key = tuple(sorted([p.a, p.b]))
        assert abs(p.p_adj - sm_p[key]) < 1e-6, (key, p.p_adj, sm_p[key])


def test_forced_method_surfaces_routing():
    df = _three_groups(np.random.default_rng(6), shifts=(0, 2, 4))
    res = mfgqc.load(df, measure="y").anova(factors=["grp"]).posthoc(method="dunn")
    assert res.method == "dunn" and res.routed is False
    assert "would have routed" in res.route_reason


# --- negative control --------------------------------------------------------
def test_no_difference_finds_no_significant_pairs():
    df = _three_groups(np.random.default_rng(11), shifts=(0, 0, 0))
    res = mfgqc.load(df, measure="y").anova(factors=["grp"]).posthoc()
    assert all(not p.significant for p in res.pairs)


# --- non-parametrics ---------------------------------------------------------
def test_mood_median_matches_scipy():
    from scipy import stats
    rng = np.random.default_rng(7)
    g = [rng.normal(s, 1, 30) for s in (0, 1, 2)]
    res = mfgqc.test_medians(*g)
    stat, p, _, _ = stats.median_test(*g)
    assert abs(res.statistic - stat) < 1e-9 and abs(res.p_value - p) < 1e-9
    assert "less powerful" in res.recommendation.lower()


def test_repeated_routes_and_matches_friedman():
    from scipy import stats
    rng = np.random.default_rng(8)
    # strongly skewed residuals with unequal condition spread -> Friedman route.
    rows = []
    for s in range(15):
        base = rng.normal(0, 1)
        for c, (shift, scale) in enumerate(((0, 0.3), (0.5, 0.3), (1.2, 4.0))):
            rows.append({"subj": s, "cond": f"c{c}", "y": base + shift + rng.exponential(scale)})
    df = pd.DataFrame(rows)
    res = mfgqc.test_repeated(df, subject="subj", within="cond", response="y")
    assert res.test_used == "friedman"
    assert not res.assumptions[0].passed or not res.assumptions[1].passed
    wide = df.pivot_table(index="subj", columns="cond", values="y")
    stat, p = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    assert abs(res.statistic - stat) < 1e-9


def test_repeated_rm_anova_matches_statsmodels():
    pytest.importorskip("statsmodels.stats.anova")
    from statsmodels.stats.anova import AnovaRM
    rng = np.random.default_rng(9)
    rows = []
    for s in range(20):
        base = rng.normal(0, 1)
        for c, shift in enumerate((0, 1.0, 2.0)):
            rows.append({"subj": s, "cond": f"c{c}", "y": base + shift + rng.normal(0, 1)})
    df = pd.DataFrame(rows)
    res = mfgqc.test_repeated(df, subject="subj", within="cond", response="y", method="rm_anova")
    aov = AnovaRM(df, depvar="y", subject="subj", within=["cond"]).fit()
    assert abs(res.statistic - float(aov.anova_table.iloc[0]["F Value"])) < 1e-6


def test_view_returns_figure():
    df = _three_groups(np.random.default_rng(0), shifts=(0, 2, 4))
    res = mfgqc.load(df, measure="y").anova(factors=["grp"]).posthoc()
    assert res.view() is not None
