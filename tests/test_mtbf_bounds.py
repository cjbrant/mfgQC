"""Slice 4: MTBF with chi-square bounds. Closed forms are self-verifying."""
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, pytest, mfgqc
from scipy import stats


def test_point_estimate_total_time_over_failures():
    r = mfgqc.reliability.mtbf(10000, failures=5)
    assert abs(r.mtbf - 2000) < 1e-9


def test_time_terminated_bounds_match_chi_square():
    T, r, conf = 10000, 5, 0.90
    res = mfgqc.reliability.mtbf(T, failures=r, kind="time_terminated", conf=conf)
    a = 1 - conf
    assert abs(res.lower - 2 * T / stats.chi2.ppf(1 - a / 2, 2 * r + 2)) < 1e-6
    assert abs(res.upper - 2 * T / stats.chi2.ppf(a / 2, 2 * r)) < 1e-6


def test_failure_terminated_uses_2r_lower_df():
    T, r = 10000, 5
    res = mfgqc.reliability.mtbf(T, failures=r, kind="failure_terminated", conf=0.90)
    assert abs(res.lower - 2 * T / stats.chi2.ppf(0.95, 2 * r)) < 1e-6
    # time-terminated adds uncertainty (df 2r+2), so its lower bound is more conservative (lower)
    tt = mfgqc.reliability.mtbf(T, failures=r, kind="time_terminated", conf=0.90)
    assert tt.lower < res.lower


def test_from_qcdata_roles():
    df = pd.DataFrame({"h": [100, 200, 300, 400, 500], "failed": [1, 1, 0, 1, 0]})
    res = mfgqc.load(df, measure="h").roles(time="h", event="failed").mtbf()
    assert res.failures == 3 and abs(res.mtbf - 1500 / 3) < 1e-9


def test_constant_rate_surfaced():
    res = mfgqc.reliability.mtbf(10000, failures=5)
    flag = next(a for a in res.assumptions if a.name == "constant_failure_rate")
    assert "constant failure rate" in flag.recommendation
