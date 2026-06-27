"""Correctness: control charts vs the R ``qcc`` package run as an independent engine.

``qcc`` (Scrucca, 2004) is a widely used R SPC package. Each test runs an R
snippet that loads a qcc dataset, computes the chart in qcc, and returns BOTH the
raw data and qcc's center/limits as JSON. mfgQC is then fed qcc's data and compared
to qcc's answer -- no mfgQC value is ever the expected value. Tests skip if Rscript
or qcc is unavailable.

Small (~1e-5) differences between qcc and mfgQC limits come from the unbiasing
constants: mfgQC uses the standard tabulated A2/D3/D4 factors, qcc uses exact d2/d3
integrals. Tolerances are set to accept that while still catching real errors.

Datasets:
* ``pistonrings`` -- 25 calibration subgroups of 5: X-bar and R charts.
* ``orangejuice`` -- frozen-juice can defectives. The "revised" p-chart drops the
  two assignable-cause samples (15, 23); this is Lawson's worked p-chart example
  (revised p-bar=0.2150, UCL=0.3893, LCL=0.0407).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mfgqc
from .conftest import run_r

_PISTON_R = """
data(pistonrings, package="qcc")
d <- with(pistonrings, qcc.groups(diameter, sample))
cal <- d[1:25,]
qx <- qcc(cal, type="xbar", plot=FALSE); qr <- qcc(cal, type="R", plot=FALSE)
out <- list(rows=lapply(1:nrow(cal), function(i) as.numeric(cal[i,])),
  xbar_center=qx$center, xbar_lcl=qx$limits[1,1], xbar_ucl=qx$limits[1,2],
  r_center=qr$center, r_lcl=qr$limits[1,1], r_ucl=qr$limits[1,2])
cat(toJSON(out, auto_unbox=TRUE, digits=12))
"""

_OJ_R = """
data(orangejuice, package="qcc")
oj <- orangejuice[orangejuice$trial,]
keep <- !(oj$sample %in% c(15, 23))
D <- oj$D[keep]; sz <- oj$size[keep]
qp <- qcc(D, sizes=sz, type="p", plot=FALSE)
out <- list(D=as.numeric(D), size=as.numeric(sz),
  pbar=qp$center, lcl=qp$limits[1,1], ucl=qp$limits[1,2])
cat(toJSON(out, auto_unbox=TRUE, digits=12))
"""


def test_qcc_pistonrings_xbar_r():
    """qcc pistonrings X-bar and R charts: centers and 3-sigma limits."""
    o = run_r(_PISTON_R, packages=("qcc",))
    recs = [{"sg": i + 1, "y": v} for i, row in enumerate(o["rows"]) for v in row]
    r = mfgqc.load(pd.DataFrame(recs), measure="y", subgroup="sg").control_chart(kind="xbar_r")
    assert r.location_cl == pytest.approx(o["xbar_center"], abs=1e-5)
    assert float(np.unique(r.location_ucl)[0]) == pytest.approx(o["xbar_ucl"], abs=1e-4)
    assert float(np.unique(r.location_lcl)[0]) == pytest.approx(o["xbar_lcl"], abs=1e-4)
    assert r.disp_cl == pytest.approx(o["r_center"], abs=1e-5)
    assert float(np.unique(r.disp_ucl)[0]) == pytest.approx(o["r_ucl"], abs=1e-4)


def test_qcc_orangejuice_revised_p_chart():
    """qcc orangejuice revised p-chart (Lawson): p-bar=0.2150, UCL=0.3893, LCL=0.0407."""
    o = run_r(_OJ_R, packages=("qcc",))
    df = pd.DataFrame({"d": [int(x) for x in o["D"]], "n": [int(x) for x in o["size"]]})
    r = mfgqc.load(df, measure="d").control_chart(kind="p", n="n")
    assert r.location_cl == pytest.approx(o["pbar"], abs=1e-6)
    assert float(np.unique(r.location_ucl)[0]) == pytest.approx(o["ucl"], abs=1e-4)
    assert float(np.unique(r.location_lcl)[0]) == pytest.approx(o["lcl"], abs=1e-4)
    # Cross-check against Lawson's published revised limits.
    assert o["pbar"] == pytest.approx(0.2150, abs=1e-3)
    assert o["ucl"] == pytest.approx(0.3893, abs=1e-3)
    assert o["lcl"] == pytest.approx(0.0407, abs=1e-3)
