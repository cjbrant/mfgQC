"""Capability confidence intervals (Montgomery Sec. 8.3.5)."""

from __future__ import annotations

import pytest

import mfgqc
from mfgqc.capability import _capability_cis


def test_ci_oracle_montgomery_example_8_5():
    # n=20, Cpk=1.33 -> 95% CI (0.88, 1.78); Cp=1.33 -> (0.911, 1.749).
    cp_ci, cpk_ci = _capability_cis(20, 1.33, 1.33, 0.05)
    assert cp_ci[0] == pytest.approx(0.911, abs=2e-3)
    assert cp_ci[1] == pytest.approx(1.749, abs=2e-3)
    assert cpk_ci[0] == pytest.approx(0.88, abs=1e-2)
    assert cpk_ci[1] == pytest.approx(1.78, abs=1e-2)


def test_ci_populated_and_brackets_point(montgomery_qc):
    cap = montgomery_qc.capability()
    assert cap.cp_ci is not None and cap.cpk_ci is not None
    assert cap.cp_ci[0] < cap.cp < cap.cp_ci[1]
    assert cap.cpk_ci[0] < cap.cpk < cap.cpk_ci[1]
    assert "95% CI" in cap.report()


def test_smaller_alpha_widens_interval(montgomery_qc):
    c95 = montgomery_qc.capability(alpha=0.05)
    c99 = montgomery_qc.capability(alpha=0.01)
    assert (c99.cpk_ci[1] - c99.cpk_ci[0]) > (c95.cpk_ci[1] - c95.cpk_ci[0])
    assert c99.summary()["confidence"] == 99


def test_nonnormal_method_has_no_ci(skewed_qc):
    cap = skewed_qc.capability(method="clements")
    assert cap.cp_ci is None and cap.cpk_ci is None
    assert "n/a (non-normal method)" in cap.report()


def test_summary_is_flat_dashboard_dict(montgomery_qc):
    s = montgomery_qc.capability().summary()
    assert {"Cp", "Cpk", "Cpk_CI_low", "Cpk_CI_high", "confidence"} <= set(s)
    assert s["confidence"] == 95
    assert all(not isinstance(v, (list, dict)) for v in s.values())  # flat
