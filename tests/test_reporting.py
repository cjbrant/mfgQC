"""The result __repr__ must render the full practitioner report, not a dataclass dump."""

from __future__ import annotations


def test_capability_repr_is_full_report(montgomery_qc):
    text = repr(montgomery_qc.capability())
    assert "Process Capability" in text
    assert "Cp" in text and "Cpk" in text
    assert "Assumption checks:" in text
    # not the dataclass auto-repr
    assert not text.startswith("CapabilityResult(")


def test_gage_repr_shows_verdict_and_recommendation(aiag_qc):
    text = repr(aiag_qc.gage_rr())
    assert "Gage R&R" in text
    assert "Verdict:" in text
    assert "Assumption checks:" in text
    assert "Recommendations:" in text  # ndc < 5 fires a recommendation
    assert not text.startswith("GageRRResult(")


def test_control_repr_reports_control_state(montgomery_qc):
    text = repr(montgomery_qc.control_chart())
    assert "Control Chart" in text
    assert "in control" in text  # this dataset is in control
    assert not text.startswith("ControlChartResult(")
