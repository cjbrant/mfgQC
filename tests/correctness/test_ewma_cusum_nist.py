"""Correctness: EWMA and CUSUM charts vs NIST e-Handbook worked examples.

Sources (NIST/SEMATECH e-Handbook):
* EWMA control chart -- 6.3.2.4
  (https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm):
  lambda=0.3, EWMA0=50, s=2.0539; published EWMA series, UCL=52.5884,
  LCL=47.4115, no out-of-control points.
* Tabular CUSUM -- 6.3.2.3
  (https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm):
  target=325, sigma(of means)=0.635, k=0.3175 (=0.5 sigma), h=4.1959;
  the first out-of-control signal is observation 14.

mfgQC was not built against these examples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mfgqc

# NIST 6.3.2.4 EWMA example.
_EWMA_DATA = [52.0, 47.0, 53.0, 49.3, 50.1, 47.0, 51.0, 50.1, 51.2, 50.5,
              49.6, 47.6, 49.9, 51.3, 47.8, 51.2, 52.6, 52.4, 53.6, 52.1]
# Published EWMA_t for t=1..20 (the table's leading 50.00 is EWMA0, dropped here).
_EWMA_PUBLISHED = [50.60, 49.52, 50.56, 50.18, 50.16, 49.21, 49.75, 49.85, 50.26, 50.33,
                   50.11, 49.36, 49.52, 50.05, 49.38, 49.92, 50.73, 51.23, 51.94, 51.99]

# NIST 6.3.2.3 CUSUM example.
_CUSUM_DATA = [324.925, 324.675, 324.725, 324.350, 325.350, 325.225, 324.125, 324.525,
               325.225, 324.600, 324.625, 325.150, 328.325, 327.250, 327.825, 328.500,
               326.675, 327.775, 326.875, 328.350]


def _ewma():
    return (mfgqc.load(pd.DataFrame({"x": _EWMA_DATA}), measure="x")
            .ewma_chart(lam=0.3, L=3, mu0=50.0, sigma=2.0539))


def test_nist_ewma_series():
    """NIST 6.3.2.4: the EWMA_t series matches the published table."""
    z = np.asarray(_ewma().z, dtype=float)
    assert z == pytest.approx(np.array(_EWMA_PUBLISHED), abs=5e-3)


def test_nist_ewma_limits_and_in_control():
    """NIST 6.3.2.4: asymptotic UCL=52.5884, LCL=47.4115, process in control."""
    ew = _ewma()
    assert float(ew.ucl[-1]) == pytest.approx(52.5884, abs=1e-3)
    assert float(ew.lcl[-1]) == pytest.approx(47.4115, abs=1e-3)
    assert len(ew.violations) == 0


def test_nist_cusum_first_signal_at_14():
    """NIST 6.3.2.3: tabular CUSUM first signals out-of-control at observation 14."""
    # NIST k=0.3175 and h=4.1959 are in data units; mfgQC takes them in sigma units.
    cu = (mfgqc.load(pd.DataFrame({"x": _CUSUM_DATA}), measure="x")
          .cusum_chart(k=0.5, h=4.1959 / 0.635, mu0=325.0, sigma=0.635))
    points = sorted(v.point for v in cu.violations)
    assert points, "expected at least one CUSUM signal"
    assert points[0] == 14
    # NIST: observations 14 through 20 remain out of control.
    assert set(points) == {14, 15, 16, 17, 18, 19, 20}
