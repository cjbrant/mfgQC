"""Slice 3: attribute agreement (kappa).

NOTE ON THE ORACLE: the spec names the AIAG MSA 4th-edition attribute study as
the oracle and says it is in the project uploads. It is NOT present (only the
gage-R&R and bias/linearity datasets are). So this slice pins the kappa MATH to
confident published anchors and library cross-checks (sklearn cohen_kappa_score,
statsmodels fleiss_kappa, statsmodels Wilson CI), pending the AIAG dataset to
close the AIAG-specific worked-example gate.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

import mfgqc
from mfgqc.attribute_agreement import cohen_kappa, fleiss_kappa

matplotlib.use("Agg")


# --- kappa math vs published anchor + libraries -------------------------------
def test_cohen_kappa_published_2x2_anchor():
    # classic 2x2: table [[20,5],[10,15]] -> kappa = 0.40 (po=0.70, pe=0.50).
    a = np.array([0] * 25 + [1] * 25)
    b = np.array([0] * 20 + [1] * 5 + [0] * 10 + [1] * 15)
    assert abs(cohen_kappa(a, b) - 0.40) < 1e-9


def test_cohen_kappa_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    a = rng.integers(0, 3, 60); b = a.copy()
    flip = rng.random(60) < 0.3
    b[flip] = rng.integers(0, 3, flip.sum())
    assert abs(cohen_kappa(a, b) - sk.cohen_kappa_score(a, b)) < 1e-9
    for w in ("linear", "quadratic"):
        assert abs(cohen_kappa(a, b, weights=w) - sk.cohen_kappa_score(a, b, weights=w)) < 1e-9


def test_fleiss_kappa_matches_statsmodels():
    ir = pytest.importorskip("statsmodels.stats.inter_rater")
    rng = np.random.default_rng(2)
    counts = rng.integers(0, 4, (20, 3))
    counts = (counts.T / counts.sum(axis=1) * 5).T.round()        # ~5 raters/item
    counts[counts.sum(axis=1) == 0] = np.array([5, 0, 0])
    # normalise rows to a constant rater count for the comparison
    counts = np.array([[3, 2, 0]] * 10 + [[5, 0, 0]] * 5 + [[1, 1, 3]] * 5, dtype=float)
    assert abs(fleiss_kappa(counts) - ir.fleiss_kappa(counts)) < 1e-9


def test_wilson_ci_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.proportion")
    from mfgqc.attribute_agreement import _wilson
    lo, hi = _wilson(45, 50)
    slo, shi = sm.proportion_confint(45, 50, method="wilson")
    assert abs(lo - slo) < 1e-9 and abs(hi - shi) < 1e-9


# --- end-to-end on a synthetic crossed study ---------------------------------
def _study(rng, p_flip=0.05, n_parts=30, appraisers=("A", "B", "C"), trials=3):
    truth = rng.integers(0, 2, n_parts)
    rows = []
    for p in range(n_parts):
        for a in appraisers:
            for t in range(trials):
                r = truth[p] if rng.random() > p_flip else 1 - truth[p]
                rows.append({"part": p, "appraiser": a, "trial": t, "y": int(r),
                             "ref": int(truth[p])})
    return pd.DataFrame(rows)


def test_high_agreement_study_passes():
    df = _study(np.random.default_rng(0), p_flip=0.02)
    res = mfgqc.load(df, measure="y").attribute_agreement(
        rating="y", part="part", appraiser="appraiser", reference="ref")
    assert res.n_appraisers == 3 and res.n_trials == 3
    assert res.between["method"] == "Fleiss"
    assert res.between["pct"] >= 0.90
    assert res.assumptions[0].passed is True             # agreement adequacy
    assert all(d["kappa"] > 0.6 for d in res.vs_reference.values())


def test_poor_agreement_flagged():
    df = _study(np.random.default_rng(3), p_flip=0.45)
    res = mfgqc.load(df, measure="y").attribute_agreement(
        rating="y", part="part", appraiser="appraiser")
    flag = next(a for a in res.assumptions if a.name == "agreement")
    assert flag.passed is False and "not reproducible" in flag.recommendation


def test_kappa_marginal_skew_paradox_flagged():
    # classic high-prevalence paradox: cross-table [[85,5],[5,5]] -> 90% agreement
    # but kappa ~ 0.44 because category 0 dominates. Flag it, don't silently fail it.
    a_ratings = [0] * 85 + [0] * 5 + [1] * 5 + [1] * 5     # appraiser A by part
    b_ratings = [0] * 85 + [1] * 5 + [0] * 5 + [1] * 5     # appraiser B by part
    rows = []
    for p in range(100):
        for a, ratings in (("A", a_ratings), ("B", b_ratings)):
            for t in range(2):                             # two identical trials
                rows.append({"part": p, "appraiser": a, "trial": t, "y": ratings[p]})
    df = pd.DataFrame(rows)
    res = mfgqc.load(df, measure="y").attribute_agreement(
        rating="y", part="part", appraiser="appraiser")
    skew_flag = next(a for a in res.assumptions if a.name == "kappa_marginal_skew")
    assert skew_flag.passed is False and "paradox" in skew_flag.recommendation


def test_report_and_view():
    df = _study(np.random.default_rng(1))
    res = mfgqc.load(df, measure="y").attribute_agreement(
        rating="y", part="part", appraiser="appraiser", reference="ref")
    assert "Landis-Koch" in res.report()
    assert res.view(kind="agreement") is not None
