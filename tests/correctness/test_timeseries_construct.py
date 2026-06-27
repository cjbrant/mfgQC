"""Correctness: Mann-Kendall trend test by construction + Theil-Sen vs scipy.

The Mann-Kendall statistic has a closed-form textbook definition, so the oracle is
computed directly from that definition in-test (not from mfgQC):

    S    = sum over i<j of sign(x_j - x_i)
    tau  = S / (n(n-1)/2)
    Var  = n(n-1)(2n+5)/18           (no ties)
    Z    = (S-1)/sqrt(Var)  if S>0;  (S+1)/sqrt(Var) if S<0

The reported OLS trend slope is cross-checked against scipy.stats.linregress, an
independent implementation. Data is seeded with no ties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import mfgqc


def _mk_by_definition(x):
    n = len(x)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(x[j] - x[i])
    tau = s / (n * (n - 1) / 2)
    var = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var)
    elif s < 0:
        z = (s + 1) / np.sqrt(var)
    else:
        z = 0.0
    return s, tau, z


def test_mann_kendall_tau_and_z_by_construction():
    """MK tau and Z match the closed-form definition computed in-test."""
    rng = np.random.default_rng(0)
    # Continuous data + linear drift => no ties, so the no-ties variance applies.
    x = np.arange(40) * 0.3 + rng.normal(scale=2.0, size=40)
    _s, tau, z = _mk_by_definition(x)
    r = mfgqc.load(pd.DataFrame({"y": x}), measure="y").timeseries()
    assert r.mk_tau == pytest.approx(tau, rel=1e-9)
    assert r.mk_z == pytest.approx(z, rel=1e-9)


def test_trend_slope_vs_scipy_linregress():
    """The reported OLS trend slope and its p-value match scipy.stats.linregress."""
    rng = np.random.default_rng(2)
    x = np.arange(50) * 0.5 + rng.normal(scale=1.5, size=50)
    lr = stats.linregress(np.arange(50), x)
    r = mfgqc.load(pd.DataFrame({"y": x}), measure="y").timeseries()
    assert r.slope == pytest.approx(lr.slope, rel=1e-9)
    assert r.slope_p == pytest.approx(lr.pvalue, rel=1e-6)
