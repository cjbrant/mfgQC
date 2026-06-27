"""Correctness: Shewhart control charts vs NIST e-Handbook worked examples.

Sources (NIST/SEMATECH e-Handbook, chapter 6.3):
* Individuals / moving-range chart -- 6.3.2.2
  (https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc322.htm)
* c-chart (counts of defects) -- 6.3.3.1
  (https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc331.htm)

These cover one variables control chart and one attribute control chart with
data + published limits stated by NIST. mfgQC was not built against either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mfgqc

# NIST 6.3.2.2 individuals chart data (in order).
_NIST_INDIVIDUALS = [49.6, 47.6, 49.9, 51.3, 47.8, 51.2, 52.6, 52.4, 53.6, 52.1]


def test_nist_individuals_chart_limits():
    """NIST 6.3.2.2: x-bar = 50.81, UCL = 55.8041, LCL = 45.8159."""
    r = mfgqc.load(pd.DataFrame({"x": _NIST_INDIVIDUALS}), measure="x").control_chart(kind="i")
    assert r.location_cl == pytest.approx(50.81, abs=1e-2)
    assert float(np.unique(r.location_ucl)[0]) == pytest.approx(55.8041, abs=1e-3)
    assert float(np.unique(r.location_lcl)[0]) == pytest.approx(45.8159, abs=1e-3)


def test_nist_moving_range_chart():
    """NIST 6.3.2.2: average moving range MR-bar = 1.8778; MR UCL = D4*MR-bar."""
    r = mfgqc.load(pd.DataFrame({"x": _NIST_INDIVIDUALS}), measure="x").control_chart(kind="i")
    assert r.disp_cl == pytest.approx(1.8778, abs=1e-3)
    # NIST states the MR-chart UCL = 3.267 * MR-bar for n=2 (D4 constant).
    assert float(np.unique(r.disp_ucl)[0]) == pytest.approx(3.267 * 1.8778, abs=2e-3)


def test_nist_c_chart_limits():
    """NIST 6.3.3.1: 25 wafers, 400 total defects -> c-bar=16, UCL=28, LCL=4."""
    counts = [16] * 25  # sum = 400; c-chart limits depend only on c-bar
    assert sum(counts) == 400
    r = mfgqc.load(pd.DataFrame({"d": counts}), measure="d").control_chart(kind="c")
    assert r.location_cl == pytest.approx(16.0, abs=1e-9)
    assert float(np.unique(r.location_ucl)[0]) == pytest.approx(28.0, abs=1e-9)
    assert float(np.unique(r.location_lcl)[0]) == pytest.approx(4.0, abs=1e-9)
