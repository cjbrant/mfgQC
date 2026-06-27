"""Reliability oracle gates (TEST_ORACLE_RELIABILITY.md).

Two tiers:
- Closed-form values (Section 1) are exact and pinned as numbers here.
- Life-data values (Section 2) are pinned to the PRIMARY held-back R key,
  reliability_oracle_key.json, generated blind by gen_reliability_oracle.R
  (survival::survreg with censoring). The build reproduces it to well under the
  3-significant-figure target.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import mfgqc

_KEY_PATH = os.path.join(os.path.dirname(__file__), "reliability_oracle_key.json")
BB = np.array([17.88, 28.92, 33.00, 41.52, 42.12, 45.60, 48.48, 51.84, 51.96, 54.12,
               55.56, 67.80, 68.64, 68.64, 68.88, 84.12, 93.12, 98.64, 105.12, 105.84,
               127.92, 128.04, 173.40])


# --------------------------------------------------------------------------- #
# Section 1: closed-form oracle values (exact)
# --------------------------------------------------------------------------- #
def test_bearing_l10_oracle():
    b = mfgqc.reliability.bearing_life(C=26, P=4.5, rpm=1500, kind="ball")
    assert b.L10_revs_million == pytest.approx(192.878, rel=1e-5)
    assert b.L10_hours == pytest.approx(2143.09, rel=1e-5)
    r = mfgqc.reliability.bearing_life(C=26, P=4.5, rpm=1500, kind="roller")
    assert r.L10_revs_million == pytest.approx(346.101, rel=1e-5)
    assert r.L10_hours == pytest.approx(3845.57, rel=1e-5)


def test_system_oracle():
    assert mfgqc.reliability.series([0.90, 0.95, 0.99]).reliability == pytest.approx(0.846450)
    assert mfgqc.reliability.parallel([0.90, 0.90]).reliability == pytest.approx(0.990000)
    assert mfgqc.reliability.k_of_n(2, 3, 0.90).reliability == pytest.approx(0.972000)


def test_mtbf_bounds_oracle():
    m = mfgqc.reliability.mtbf(5000, failures=8, kind="time_terminated", conf=0.90)
    assert m.mtbf == pytest.approx(625.000)
    assert m.lower == pytest.approx(346.389, rel=1e-5)
    assert m.upper == pytest.approx(1256.022, rel=1e-5)
    mf = mfgqc.reliability.mtbf(5000, failures=8, kind="failure_terminated", conf=0.90)
    assert mf.lower == pytest.approx(380.283, rel=1e-5)


def test_demonstration_oracle():
    assert mfgqc.reliability.demonstration_test(reliability=0.95, confidence=0.90).n == 45
    inv = mfgqc.reliability.demonstration_test(confidence=0.90, n=22)
    assert inv.reliability == pytest.approx(0.90063, rel=1e-4)


# --------------------------------------------------------------------------- #
# Section 2: life-data, pinned to the primary R survreg key
# --------------------------------------------------------------------------- #
def _key():
    if not os.path.exists(_KEY_PATH):
        pytest.skip("reliability_oracle_key.json (R survreg key) not present")
    return json.load(open(_KEY_PATH))


def _fit(times, status, dist):
    df = pd.DataFrame({"h": times, "failed": status})
    return mfgqc.load(df, measure="h").roles(time="h", event="failed").life_fit(dist=dist)


@pytest.mark.parametrize("case,times,status", [
    ("complete_ballbearing", BB, np.ones(23)),
    ("censored_ballbearing", np.minimum(BB, 100.0), (BB <= 100).astype(int)),
])
def test_life_fit_matches_survreg_key(case, times, status):
    k = _key()[case]
    rtol = 3e-3                                    # 3 significant figures
    w = _fit(times, status, "weibull")
    assert w.params["shape"] == pytest.approx(k["weibull"]["beta"], rel=rtol)
    assert w.params["scale"] == pytest.approx(k["weibull"]["eta"], rel=rtol)
    assert w.mttf == pytest.approx(k["weibull"]["MTTF"], rel=rtol)
    assert w.b10 == pytest.approx(k["weibull"]["B10"], rel=rtol)
    assert w.b50 == pytest.approx(k["weibull"]["median"], rel=rtol)
    assert w.n_fail == k["failures"]              # suspensions honored, not dropped

    ln = _fit(times, status, "lognormal")
    assert ln.params["mu"] == pytest.approx(k["lognormal"]["mulog"], rel=rtol)
    assert ln.params["sigma"] == pytest.approx(k["lognormal"]["sigmalog"], rel=rtol)
    assert ln.mttf == pytest.approx(k["lognormal"]["MTTF"], rel=rtol)

    ex = _fit(times, status, "exponential")
    assert 1.0 / ex.params["scale"] == pytest.approx(k["exponential"]["rate"], rel=rtol)
    assert ex.mttf == pytest.approx(k["exponential"]["MTTF"], rel=rtol)


def test_kaplan_meier_matches_survfit_key():
    k = _key()["complete_ballbearing"]["km"]
    km = mfgqc.load(pd.DataFrame({"h": BB, "failed": np.ones(23)}),
                   measure="h").roles(time="h", event="failed").life_table()
    assert km.median_life == pytest.approx(k["median"], rel=3e-3)
    for t, R in zip(k["probe_times"], k["R_at_probes"]):
        assert km.R(t) == pytest.approx(R, rel=3e-3)


@pytest.mark.parametrize("case,times,status", [
    ("complete_ballbearing", BB, np.ones(23)),
    ("censored_ballbearing", np.minimum(BB, 100.0), (BB <= 100).astype(int)),
])
def test_weibull_lr_profile_ci_matches_key(case, times, status):
    # mfgQC's MLE confidence interval is a likelihood-ratio PROFILE interval, and
    # the key's primary shape_lr_ci/scale_lr_ci are the matching profile intervals
    # (R, profiled censored Weibull log-likelihood). The key's *_logwald_ci are the
    # secondary log-Wald intervals (a different method) and are NOT used here.
    w = _key()[case]["weibull"]
    df = pd.DataFrame({"h": times, "failed": status})
    fit = mfgqc.load(df, measure="h").roles(time="h", event="failed").life_fit(
        dist="weibull", method="mle")
    sh_lo, sh_hi = fit.param_ci["shape"]
    sc_lo, sc_hi = fit.param_ci["scale"]
    assert sh_lo == pytest.approx(w["shape_lr_ci"][0], rel=3e-3)
    assert sh_hi == pytest.approx(w["shape_lr_ci"][1], rel=3e-3)
    assert sc_lo == pytest.approx(w["scale_lr_ci"][0], rel=3e-3)
    assert sc_hi == pytest.approx(w["scale_lr_ci"][1], rel=3e-3)


def test_censoring_changes_the_fit():
    # negative control: honoring vs ignoring suspensions gives visibly different fits.
    complete = _fit(BB, np.ones(23), "weibull")
    censored = _fit(np.minimum(BB, 100.0), (BB <= 100).astype(int), "weibull")
    assert abs(complete.params["shape"] - censored.params["shape"]) > 0.05
