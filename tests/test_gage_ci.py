"""Gage R&R component confidence intervals (AIAG MSA 4th ed., Table III-B.9)."""

from __future__ import annotations

import pytest

import mfgqc


def test_ci_reproduces_aiag_table_iiib9(aiag_qc):
    # ANOVA method, 90% confidence limits (alpha=0.10 default).
    # AIAG Table III-B.9: EV 0.177/0.200/0.231, AV 0.129/0.227/1.001, GRR 0.237/0.302/1.033.
    g = aiag_qc.gage_rr()
    assert g.alpha == 0.10
    assert g.ev_ci[0] == pytest.approx(0.177, abs=2e-3)
    assert g.ev_ci[1] == pytest.approx(0.231, abs=2e-3)
    assert g.grr_ci[0] == pytest.approx(0.237, abs=4e-3)
    assert g.grr_ci[1] == pytest.approx(1.033, abs=3e-3)
    # AV bounds match the table to published rounding (the wide upper is MLS-characteristic)
    assert g.av_ci[0] == pytest.approx(0.129, abs=2e-3)
    assert g.av_ci[1] == pytest.approx(1.001, abs=2e-2)


def test_point_estimates_lie_inside_intervals(aiag_qc):
    g = aiag_qc.gage_rr()
    for sd, ci in [(g.ev, g.ev_ci), (g.av, g.av_ci), (g.grr, g.grr_ci), (g.pv, g.pv_ci)]:
        assert ci[0] <= sd <= ci[1]


def test_report_shows_confidence_columns(aiag_qc):
    rep = aiag_qc.gage_rr().report()
    assert "lower 90%" in rep and "upper 90%" in rep


def test_xbar_r_method_omits_cis(aiag_qc):
    g = aiag_qc.gage_rr(method="xbar_r")
    assert g.ev_ci is None and g.grr_ci is None and g.av_ci is None


def test_summary_is_flat_dashboard_dict(aiag_qc):
    s = aiag_qc.gage_rr().summary()
    assert {"GRR", "GRR_CI_low", "GRR_CI_high", "ndc", "verdict", "confidence"} <= set(s)
    assert s["confidence"] == 90
    assert all(not isinstance(v, (list, dict)) for v in s.values())
