"""Assumption flags v2: binary verdict + context (supersedes graded).

Verifies the v2 design and the specific regression it fixes: the DIRECT test
drives ``passed``; magnitude/reliability are adjacent context that never flip the
verdict. The geyser-class failure (a small Cpk shift issuing a false all-clear on
grossly non-normal data) is gone because passed is AD-driven.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mfgqc
from mfgqc import assumptions as A


def _qc(x, lo, hi):
    return mfgqc.load(pd.DataFrame({"y": np.asarray(x, dtype=float)}),
                     measure="y").spec(lower=lo, upper=hi)


def test_assumptioncheck_structure():
    c = A.check_normality(np.random.default_rng(0).normal(0, 1, 100))
    assert isinstance(c.passed, bool)
    assert c.reliability in ("ok", "low_power", "oversensitive")
    assert hasattr(c, "magnitude") and hasattr(c, "magnitude_label") and hasattr(c, "n")
    assert not hasattr(c, "severity")  # graded scale removed


def test_bimodal_is_fail_despite_small_magnitude():
    """The v1 regression: bimodal data (huge AD) must FAIL, magnitude doesn't override."""
    rng = np.random.default_rng(1)
    bimodal = np.concatenate([rng.normal(0, 1, 300), rng.normal(8, 1, 300)])  # ~0 skew
    c = A.check_normality(bimodal)  # no cpk_impact -> skew is the (small) magnitude
    assert c.passed is False                    # AD drives the verdict
    assert c.magnitude_label == "skew"
    assert abs(c.magnitude) < 0.5               # magnitude is small, yet verdict is FAIL


def test_geyser_capability_fails_with_small_cpk_context(montgomery_qc):
    """A bimodal sample with a spec: FAIL at the test level, small Cpk impact as context."""
    rng = np.random.default_rng(2)
    bimodal = np.concatenate([rng.normal(40, 3, 150), rng.normal(80, 3, 150)])
    cap = _qc(bimodal, 20, 100).capability()
    norm = next(a for a in cap.assumptions if a.name == "normality")
    assert norm.passed is False                     # not normal -> FAIL, no false all-clear
    assert norm.magnitude_label == "est. Cpk impact"  # impact still shown as context


def test_large_n_trivial_deviation_fails_but_flagged_oversensitive():
    # A mild skew (gamma) at huge n: AD detects it (FAIL, the honest verdict), but the
    # reliability caveat flags n is huge and the Cpk impact is small.
    rng = np.random.default_rng(3)
    x = rng.gamma(50, 2.0, 100_000)  # mean ~100, sd ~14, skew ~0.28
    cap = _qc(x, 40, 160).capability()
    norm = next(a for a in cap.assumptions if a.name == "normality")
    assert norm.passed is False
    assert norm.reliability == "oversensitive"
    assert norm.magnitude is not None and norm.magnitude < 0.2  # impact small


def test_small_n_clean_passes_with_low_power_caveat():
    rng = np.random.default_rng(4)
    c = A.check_normality(rng.normal(0, 1, 12))
    assert c.passed is True
    assert c.reliability == "low_power"
    assert c.n == 12


def test_variance_ratio_is_context_levene_drives_verdict():
    rng = np.random.default_rng(5)
    a = rng.normal(0, 1, 60); b = rng.normal(0, 1, 60)         # ratio ~1
    c_ok = A.check_homogeneity([a, b])
    assert c_ok.passed is True
    assert c_ok.magnitude_label == "variance ratio"

    a2 = rng.normal(0, 1, 60); b2 = rng.normal(0, 5, 60)       # ratio ~25
    c_fail = A.check_homogeneity([a2, b2])
    assert c_fail.passed is False
    assert c_fail.magnitude > 9


def test_report_renders_binary_plus_context():
    rng = np.random.default_rng(6)
    text = repr(_qc(rng.lognormal(1.0, 0.5, 2000), 0, 20).capability())
    assert "[FAIL]" in text or "[PASS]" in text   # binary bracket, not a graded scale
    assert "est. Cpk impact" in text              # magnitude context present
    assert "[ok]" not in text and "[severe]" not in text  # no graded severities
