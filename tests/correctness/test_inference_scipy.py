"""Correctness: inference modules vs scipy / statsmodels / scikit-learn in-test.

Each test computes the oracle with an independent engine on freshly seeded data
(no network, no transcribed constants) and checks mfgQC reproduces it. The expected
value is never taken from a prior mfgQC run -- it is computed here by a different
library. This is the companion's "strongest for the inference modules" route.

Modules covered: contingency (chi-square), correlation, process_sigma (DPMO<->Z),
power (t-test and ANOVA), posthoc (Tukey HSD), nonparametric (Mood's median test),
attribute_agreement (Cohen's kappa, incl. weighted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import mfgqc
from mfgqc.attribute_agreement import cohen_kappa


# --------------------------------------------------------------------------- #
# contingency / chi-square  vs  scipy.stats.chi2_contingency
# --------------------------------------------------------------------------- #
def test_contingency_vs_scipy():
    """chi-square statistic / dof / p vs scipy.chi2_contingency (no correction)."""
    table = np.array([[42, 30, 28], [55, 40, 25], [38, 52, 60]])
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    r = mfgqc.contingency(table.tolist())
    assert r.chi2 == pytest.approx(chi2, rel=1e-9)
    assert r.p_value == pytest.approx(p, rel=1e-9)
    assert r.dof == dof


# --------------------------------------------------------------------------- #
# correlation  vs  scipy.stats.pearsonr / spearmanr
# --------------------------------------------------------------------------- #
def test_pearson_correlation_vs_scipy():
    """Pearson r and its p-value vs scipy.stats.pearsonr on seeded data."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=80)
    y = 0.6 * x + rng.normal(size=80)
    pr = stats.pearsonr(x, y)
    r = mfgqc.correlation(pd.DataFrame({"x": x, "y": y}), method="pearson")
    assert r.corr[("x", "y")] == pytest.approx(pr.statistic, rel=1e-9)
    assert r.p_values[("x", "y")] == pytest.approx(pr.pvalue, rel=1e-6)


def test_spearman_correlation_vs_scipy():
    """Spearman rho vs scipy.stats.spearmanr on seeded data."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=60)
    y = np.exp(0.5 * x) + rng.normal(scale=0.3, size=60)  # monotone, non-linear
    sr = stats.spearmanr(x, y)
    r = mfgqc.correlation(pd.DataFrame({"x": x, "y": y}), method="spearman")
    assert r.corr[("x", "y")] == pytest.approx(sr.statistic, rel=1e-9)


# --------------------------------------------------------------------------- #
# process_sigma  DPMO <-> Z  vs  scipy.stats.norm (the defining identity)
# --------------------------------------------------------------------------- #
def test_process_sigma_dpmo_and_z_vs_norm():
    """DPMO and long/short-term Z vs the scipy.norm definition with a 1.5 shift."""
    defects, units, opps = 23.0, 1000.0, 5.0
    r = mfgqc.process_sigma(defects, units, opps, kind="defects")
    p_defect = defects / (units * opps)
    assert r.dpmo == pytest.approx(p_defect * 1e6, rel=1e-12)
    assert r.z_lt == pytest.approx(stats.norm.isf(p_defect), abs=1e-9)
    assert r.z_st == pytest.approx(stats.norm.isf(p_defect) + 1.5, abs=1e-9)


# --------------------------------------------------------------------------- #
# power  vs  statsmodels.stats.power
# --------------------------------------------------------------------------- #
def test_power_two_sample_t_vs_statsmodels():
    """Two-sample t-test power vs statsmodels.TTestIndPower."""
    from statsmodels.stats.power import TTestIndPower
    sm = TTestIndPower().power(effect_size=0.5, nobs1=64, alpha=0.05,
                               ratio=1.0, alternative="two-sided")
    r = mfgqc.power.t_test(effect=0.5, n=64, alpha=0.05, kind="two-sample")
    assert r.power == pytest.approx(sm, rel=1e-4)


def test_power_anova_vs_statsmodels():
    """One-way ANOVA power vs statsmodels.FTestAnovaPower."""
    from statsmodels.stats.power import FTestAnovaPower
    # mfgQC's n is per group (k=4 groups); statsmodels nobs is the total sample size.
    sm = FTestAnovaPower().power(effect_size=0.4, nobs=60 * 4, alpha=0.05, k_groups=4)
    r = mfgqc.power.anova(groups=4, effect=0.4, n=60, alpha=0.05)
    assert r.power == pytest.approx(sm, rel=1e-4)


# --------------------------------------------------------------------------- #
# posthoc Tukey HSD  vs  statsmodels.pairwise_tukeyhsd
# --------------------------------------------------------------------------- #
def test_tukey_vs_statsmodels():
    """Tukey HSD pairwise differences and adjusted p-values vs statsmodels."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    rng = np.random.default_rng(3)
    groups = [rng.normal(loc, 1.0, 12) for loc in (0.0, 0.5, 1.5)]
    data = np.concatenate(groups)
    labels = np.repeat(["g1", "g2", "g3"], 12)
    sm = pairwise_tukeyhsd(data, labels)
    # statsmodels rows are in (g1,g2),(g1,g3),(g2,g3) order with meandiff = b - a.
    sm_diffs = {tuple(sorted((str(a), str(b)))): d
                for a, b, d in zip(sm.groupsunique[sm._multicomp.pairindices[0]],
                                   sm.groupsunique[sm._multicomp.pairindices[1]],
                                   sm.meandiffs)}
    sm_padj = {tuple(sorted((str(a), str(b)))): p
               for a, b, p in zip(sm.groupsunique[sm._multicomp.pairindices[0]],
                                  sm.groupsunique[sm._multicomp.pairindices[1]],
                                  sm.pvalues)}
    ph = mfgqc.test_anova(*groups, labels=["g1", "g2", "g3"]).posthoc(method="tukey")
    for pair in ph.pairs:
        key = tuple(sorted((pair.a, pair.b)))
        assert abs(pair.diff) == pytest.approx(abs(sm_diffs[key]), abs=1e-9)
        assert pair.p_adj == pytest.approx(sm_padj[key], abs=1e-6)


# --------------------------------------------------------------------------- #
# nonparametric Mood's median test  vs  scipy.stats.median_test
# --------------------------------------------------------------------------- #
def test_median_test_vs_scipy():
    """test_medians (Mood's median test) statistic and p vs scipy.median_test."""
    rng = np.random.default_rng(5)
    groups = [rng.normal(loc, 1.0, 30) for loc in (0.0, 0.4, 0.9)]
    stat, p, *_ = stats.median_test(*groups)
    r = mfgqc.test_medians(*groups)
    assert r.statistic == pytest.approx(stat, rel=1e-9)
    assert r.p_value == pytest.approx(p, rel=1e-9)


# --------------------------------------------------------------------------- #
# attribute_agreement Cohen's kappa  vs  sklearn.metrics.cohen_kappa_score
# --------------------------------------------------------------------------- #
def test_cohen_kappa_vs_sklearn():
    """Unweighted and weighted Cohen's kappa vs scikit-learn."""
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(9)
    a = rng.integers(1, 6, size=50)
    b = a.copy()
    flip = rng.choice(50, size=12, replace=False)
    b[flip] = rng.integers(1, 6, size=12)
    cats = [1, 2, 3, 4, 5]
    assert cohen_kappa(a, b, categories=cats) == pytest.approx(
        cohen_kappa_score(a, b), abs=1e-9)
    assert cohen_kappa(a, b, categories=cats, weights="linear") == pytest.approx(
        cohen_kappa_score(a, b, weights="linear"), abs=1e-9)
    assert cohen_kappa(a, b, categories=cats, weights="quadratic") == pytest.approx(
        cohen_kappa_score(a, b, weights="quadratic"), abs=1e-9)
