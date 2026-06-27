"""Hypothesis-testing oracles (Montgomery Ch. 4) + routing/behavioral checks.

H1/H2 are pinned to published t0/p/CI values. Because Montgomery reports summary
statistics (x-bar, s, n), the data are synthesized to those EXACT statistics, so
the test reproduces the published numbers. H3/H4 are behavioral + scipy-pinned
(the right approach where no single textbook number applies).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import mfgqc


def make_exact(mean: float, sd: float, n: int, seed: int = 0) -> np.ndarray:
    """A near-normal sample with EXACTLY the given sample mean and sample sd."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    z = (z - z.mean()) / z.std(ddof=1)
    return z * sd + mean


# --------------------------------------------------------------------------- #
# H1 - one-sample t (Montgomery Example 4.3)
# --------------------------------------------------------------------------- #
def test_one_sample_t_montgomery_4_3():
    x = make_exact(3210.73, 117.61, 15)
    res = mfgqc.test_mean(x, target=3200)
    assert res.test_used == "one-sample t"
    assert res.statistic == pytest.approx(0.353, abs=2e-3)
    assert res.df == 14
    assert res.p_value == pytest.approx(0.729, abs=2e-3)
    assert res.ci[0] == pytest.approx(3145.6, abs=0.2)
    assert res.ci[1] == pytest.approx(3275.9, abs=0.2)


# --------------------------------------------------------------------------- #
# H2 - two-sample pooled t (Montgomery Example 4.9)
# --------------------------------------------------------------------------- #
def test_two_sample_pooled_t_montgomery_4_9():
    a = make_exact(92.255, 2.39, 8, seed=1)
    b = make_exact(92.733, 2.98, 8, seed=2)
    res = mfgqc.test_means(a, b)  # default: Student's pooled
    assert res.test_used == "Student's t (pooled)"
    assert res.statistic == pytest.approx(-0.354, abs=3e-3)
    assert res.df == 14
    assert res.p_value == pytest.approx(0.7287, abs=3e-3)


# --------------------------------------------------------------------------- #
# H3 - routing / consistency
# --------------------------------------------------------------------------- #
def test_unequal_variance_routes_to_welch():
    rng = np.random.default_rng(10)
    a = rng.normal(0.0, 1.0, 50)
    b = rng.normal(0.0, 5.0, 50)  # clearly larger variance

    # routing is the DEFAULT: unequal variance -> Welch automatically
    routed = mfgqc.test_means(a, b)
    assert routed.test_used == "Welch's t"
    assert routed.routed is True
    # Welch (Satterthwaite) df is fractional, unlike Student's integer df
    assert abs(routed.df - round(routed.df)) > 1e-6
    assert "welch" in routed.selection_reason.lower()
    lev = next(c for c in routed.assumptions if c.name == "homogeneity_of_variance")
    assert lev.passed is False

    # forcing pooled overrides, and warns it overrode the recommendation
    forced = mfgqc.test_means(a, b, method="pooled")
    assert forced.test_used == "Student's t (pooled)"
    assert forced.routed is False
    assert "welch" in (forced.recommendation or "").lower()


def test_nonnormal_routes_to_mannwhitney():
    rng = np.random.default_rng(11)
    a = rng.exponential(1.0, 60)
    b = rng.exponential(2.0, 60)

    routed = mfgqc.test_means(a, b)   # non-normal -> Mann-Whitney by default
    assert routed.test_used == "Mann-Whitney U"
    assert routed.routed is True
    assert "mann-whitney" in routed.selection_reason.lower()


def test_proportion_validity():
    # min expected count = 5 meets the >=5 rule -> passes; magnitude shown as context
    r = mfgqc.test_proportion(2, 10, 0.5)
    assert r.assumptions[0].magnitude == 5.0
    assert r.assumptions[0].passed is True
    # min expected count = 1 -> fails the direct rule, recommends exact test
    r2 = mfgqc.test_proportion(1, 20, 0.05)
    assert r2.assumptions[0].passed is False
    assert r2.recommendation is not None
    assert "binomial" in r2.recommendation.lower() or "exact" in r2.recommendation.lower()


def test_student_t_matches_scipy():
    rng = np.random.default_rng(12)
    a = rng.normal(10.0, 2.0, 40)
    b = rng.normal(11.0, 2.0, 40)
    res = mfgqc.test_means(a, b, test="student")
    ref = stats.ttest_ind(a, b, equal_var=True)
    assert res.statistic == pytest.approx(ref.statistic, abs=1e-9)
    assert res.p_value == pytest.approx(ref.pvalue, abs=1e-9)


# --------------------------------------------------------------------------- #
# H4 - variance tests
# --------------------------------------------------------------------------- #
def test_bartlett_levene_match_scipy():
    rng = np.random.default_rng(13)
    groups = [rng.normal(0, 1, 30), rng.normal(0, 1.2, 30), rng.normal(0, 0.9, 30)]
    bart = mfgqc.test_variance(*groups)  # normal -> Bartlett
    assert bart.test_used == "Bartlett"
    ref_b = stats.bartlett(*groups)
    assert bart.statistic == pytest.approx(ref_b.statistic, abs=1e-6)
    assert bart.p_value == pytest.approx(ref_b.pvalue, abs=1e-6)

    skew = [rng.exponential(1, 40), rng.exponential(1.5, 40), rng.exponential(2, 40)]
    lev = mfgqc.test_variance(*skew, auto=True)  # non-normal -> Levene
    assert lev.test_used == "Levene"
    ref_l = stats.levene(*skew, center="median")
    assert lev.statistic == pytest.approx(ref_l.statistic, abs=1e-6)
    assert lev.p_value == pytest.approx(ref_l.pvalue, abs=1e-6)


def test_normality_gates_bartlett_vs_levene():
    rng = np.random.default_rng(14)
    normal_groups = [rng.normal(0, 1, 40), rng.normal(0, 1, 40), rng.normal(0, 1, 40)]
    res_n = mfgqc.test_variance(*normal_groups)   # routes: normal -> Bartlett
    assert res_n.test_used == "Bartlett"
    assert res_n.routed is True

    skew_groups = [rng.exponential(1, 50), rng.exponential(1, 50), rng.exponential(1, 50)]
    res_s = mfgqc.test_variance(*skew_groups)     # routes: non-normal -> Levene
    assert res_s.test_used == "Levene"
    assert "levene" in res_s.selection_reason.lower()


# --------------------------------------------------------------------------- #
# Integration: QCData method + provenance
# --------------------------------------------------------------------------- #
def test_qcdata_test_mean_propagates_history():
    import pandas as pd
    x = make_exact(3210.73, 117.61, 15)
    qc = mfgqc.load(pd.DataFrame({"v": x}), measure="v")
    res = qc.test_mean(3200)
    assert res.statistic == pytest.approx(0.353, abs=2e-3)
    # history starts from the QCData ingestion step, then the test + assumptions
    assert res.history[0].operation == "load"
    assert any(s.operation == "test_mean" for s in res.history)


def test_repr_is_full_report():
    x = make_exact(3210.73, 117.61, 15)
    text = repr(mfgqc.test_mean(x, target=3200))
    assert "H0:" in text and "H1:" in text
    assert "one-sample t" in text
    assert "decision at alpha" in text
    assert not text.startswith("HypothesisResult(")
