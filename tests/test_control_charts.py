"""Control-chart oracles (Montgomery hard-bake)."""

from __future__ import annotations

import numpy as np
import pytest


def test_hardbake_xbar_r_limits(montgomery_qc):
    """Oracle 1: the six CL/UCL/LCL values for the X-bar and R charts."""
    cc = montgomery_qc.control_chart()  # kind inferred
    assert cc.kind == "xbar_r"
    assert cc.inferred is True

    # X-bar (location)
    assert cc.location_cl == pytest.approx(1.506, abs=2e-3)
    assert float(np.unique(cc.location_ucl)[0]) == pytest.approx(1.693, abs=2e-3)
    assert float(np.unique(cc.location_lcl)[0]) == pytest.approx(1.318, abs=2e-3)

    # R (dispersion)
    assert cc.disp_label == "R"
    assert cc.disp_cl == pytest.approx(0.3252, abs=2e-3)
    assert float(np.unique(cc.disp_ucl)[0]) == pytest.approx(0.6875, abs=2e-3)
    assert float(np.unique(cc.disp_lcl)[0]) == pytest.approx(0.0, abs=1e-9)


def test_hardbake_in_control_no_violations(montgomery_qc):
    """Run-rules engine: this in-control dataset yields zero violations."""
    cc = montgomery_qc.control_chart(rules="nelson")
    assert cc.violations == []


def test_explicit_kind_not_overridden(montgomery_qc):
    cc = montgomery_qc.control_chart(kind="xbar_s")
    assert cc.kind == "xbar_s"
    assert cc.inferred is False
    assert cc.disp_label == "S"
