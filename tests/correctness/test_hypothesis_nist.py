"""Correctness: hypothesis tests vs NIST e-Handbook worked examples.

Sources (NIST/SEMATECH e-Handbook):
* Two-sample t-test -- 1.3.5.3 / 7.3.1 (AUTO83B.DAT miles-per-gallon)
  (https://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm):
  US n=249, mean=20.14458, sd=6.41470; Japan n=79, mean=30.48101, sd=6.10771;
  pooled t = -12.62059 on 326 degrees of freedom.
* One-way ANOVA -- 7.4.3
  (https://www.itl.nist.gov/div898/handbook/prc/section4/prc433.htm):
  three groups of five, SS(treatments)=27.897, SS(error)=17.452, F=9.59.

mfgQC was not built against these examples.
"""

from __future__ import annotations

import pandas as pd
import pytest

import mfgqc
from .conftest import exact_sample


def test_nist_two_sample_t():
    """NIST AUTO83B two-sample (pooled) t-test: t = -12.62059, df = 326."""
    us = exact_sample(mean=20.14458, sd=6.41470, n=249)
    jp = exact_sample(mean=30.48101, sd=6.10771, n=79)
    r = mfgqc.test_means(us, jp, method="pooled")
    assert r.statistic == pytest.approx(-12.62059, abs=1e-4)
    assert r.df == pytest.approx(326.0, abs=1e-9)


def test_nist_one_way_anova():
    """NIST 7.4.3 one-way ANOVA: SS_treat=27.897, SS_err=17.452, F=9.59."""
    g1 = [6.9, 5.4, 5.8, 4.6, 4.0]
    g2 = [8.3, 6.8, 7.8, 9.2, 6.5]
    g3 = [8.0, 10.5, 8.1, 6.9, 9.3]
    rows = ([{"y": v, "g": "1"} for v in g1]
            + [{"y": v, "g": "2"} for v in g2]
            + [{"y": v, "g": "3"} for v in g3])
    a = mfgqc.load(pd.DataFrame(rows), measure="y").anova(factors=["g"])
    assert a.table["g"]["ss"] == pytest.approx(27.897, abs=1e-2)
    assert a.table["residual"]["ss"] == pytest.approx(17.452, abs=1e-2)
    assert a.table["g"]["df"] == 2
    assert a.table["residual"]["df"] == 12
    assert a.table["g"]["f"] == pytest.approx(9.59, abs=1e-2)
