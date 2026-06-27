"""Capability oracles (Montgomery)."""

from __future__ import annotations

import numpy as np
import pytest

from mfgqc import assumptions


def test_hardbake_cp_cpk_within_sigma(montgomery_qc):
    """Oracle 1: within-sigma = R-bar/d2 = 0.1398, Cp=1.192, Cpk=1.179."""
    cap = montgomery_qc.capability()  # default normal
    assert cap.sigma_within == pytest.approx(0.1398, abs=5e-4)
    assert "within (R-bar/d2)" == cap.sigma_used
    assert cap.cp == pytest.approx(1.192, abs=5e-3)
    assert cap.cpk == pytest.approx(1.179, abs=5e-3)
    assert cap.cpu == pytest.approx(1.179, abs=5e-3)
    assert cap.cpl == pytest.approx(1.205, abs=5e-3)
    # Pp/Ppk use overall sigma (ordinary SD), a DIFFERENT estimator than the
    # within-subgroup R-bar/d2 used for Cp/Cpk -> the two indices differ.
    assert cap.sigma_overall != cap.sigma_within
    assert cap.pp is not None and cap.pp != cap.cp


def test_one_sided_cpl(one_sided_qc):
    """Oracle 2: one-sided lower spec -> only Cpl reported, Cpl=0.667."""
    cap = one_sided_qc.capability()
    assert cap.cpl == pytest.approx(0.667, abs=2e-3)
    assert cap.cp is None
    assert cap.cpu is None
    assert cap.cpk == pytest.approx(cap.cpl)


def test_normality_pass_glass(glass_values):
    """Oracle 3 (pass): glass data should NOT reject normality at 0.05."""
    check = assumptions.check_normality(glass_values)
    assert check.passed is True
    assert check.p_value is not None and check.p_value >= 0.05
    assert check.reliability == "ok"


def test_normality_fail_skewed(skewed_qc):
    """Oracle 3 (fail): skewed data -> binary FAIL (AD), with Cpk impact as context."""
    cap = skewed_qc.capability()  # normal method
    norm = next(a for a in cap.assumptions if a.name == "normality")
    assert norm.passed is False
    assert norm.magnitude_label == "est. Cpk impact"  # practical impact is CONTEXT
    assert norm.recommendation is not None
    assert "clements" in norm.recommendation.lower() or "johnson" in norm.recommendation.lower()


def test_capability_requires_a_limit():
    import pandas as pd
    import mfgqc
    df = pd.DataFrame({"y": np.arange(30.0)})
    qc = mfgqc.load(df, measure="y")  # no spec
    with pytest.raises(ValueError, match="spec limit"):
        qc.capability()
