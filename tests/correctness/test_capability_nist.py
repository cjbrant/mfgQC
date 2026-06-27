"""Correctness: process-capability indices vs the NIST e-Handbook worked example.

Source: NIST/SEMATECH e-Handbook 6.1.6 "What is Process Capability?"
(https://www.itl.nist.gov/div898/handbook/pmc/section1/pmc16.htm). The handbook
states: USL=20, LSL=8, process mean=16, sigma=2, giving Cp=1.0, Cpu=0.6667,
Cpl=1.3333, Cpk=0.6667. We feed mfgQC a sample whose sample mean and sample sd are
exactly 16 and 2 and check it reproduces the published indices.
"""

from __future__ import annotations

import pandas as pd
import pytest

import mfgqc
from .conftest import exact_sample


def _nist_capability():
    x = exact_sample(mean=16.0, sd=2.0, n=50)
    return (mfgqc.load(pd.DataFrame({"v": x}), measure="v")
            .spec(lower=8.0, upper=20.0)
            .capability())


def test_nist_cp():
    """NIST 6.1.6: Cp = (USL-LSL)/(6 sigma) = 12/12 = 1.0."""
    assert _nist_capability().cp == pytest.approx(1.0, abs=1e-4)


def test_nist_cpu_cpl():
    """NIST 6.1.6: Cpu = 0.6667 (upper) and Cpl = 1.3333 (lower)."""
    r = _nist_capability()
    assert r.cpu == pytest.approx(0.6667, abs=1e-4)
    assert r.cpl == pytest.approx(1.3333, abs=1e-4)


def test_nist_cpk_is_min_of_cpu_cpl():
    """NIST 6.1.6: Cpk = min(Cpu, Cpl) = 0.6667."""
    assert _nist_capability().cpk == pytest.approx(0.6667, abs=1e-4)
