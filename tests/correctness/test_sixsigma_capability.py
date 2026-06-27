"""Correctness: capability indices vs the R ``SixSigma`` package (independent engine).

``SixSigma`` accompanies Cano, Moguerza & Redchuk, *Six Sigma with R* (Springer,
2012). The ``ss.data.ca`` dataset is bottle-fill volumes with specification
740 / 760 and target 750. The R snippet computes the long-term Cp and Cpk and
returns them together with the raw data; mfgQC is fed the same data and compared to
R's indices. Tests skip if Rscript or SixSigma is unavailable.
"""

from __future__ import annotations

import pandas as pd
import pytest

import mfgqc
from .conftest import run_r

_CA_R = """
data(ss.data.ca, package="SixSigma")
x <- ss.data.ca$Volume
LSL <- 740; USL <- 760
mu <- mean(x); s <- sd(x)
Cp <- (USL - LSL) / (6 * s)
Cpk <- min(USL - mu, mu - LSL) / (3 * s)
cat(toJSON(list(volume=as.numeric(x), Cp=Cp, Cpk=Cpk, mean=mu, sd=s),
           auto_unbox=TRUE, digits=12))
"""


def test_sixsigma_ca_capability():
    """SixSigma ss.data.ca: Cp and Cpk match the R computation on the same data."""
    o = run_r(_CA_R, packages=("SixSigma",))
    r = (mfgqc.load(pd.DataFrame({"v": o["volume"]}), measure="v")
         .spec(lower=740.0, upper=760.0, target=750.0)
         .capability())
    assert r.mean == pytest.approx(o["mean"], rel=1e-9)
    assert r.cp == pytest.approx(o["Cp"], rel=1e-6)
    assert r.cpk == pytest.approx(o["Cpk"], rel=1e-6)
