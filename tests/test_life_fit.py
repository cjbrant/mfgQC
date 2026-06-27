"""Slices 1-2: life_fit (MLE + rank regression, censoring).

ORACLE NOTE: the spec pins the life-data core to a held-back R answer key
(survreg/flexsurv/WeibullR) generated blind. That key is not in the project, so
this slice pins to (a) the published Lieblein-Zelen ball-bearing Weibull MLE
(shape 2.102, scale 81.88, cited in Meeker and Lawless), (b) scipy's independent
CensoredData fit as the secondary library cross-check, and (c) negative controls.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

import mfgqc

matplotlib.use("Agg")

# classic ball-bearing fatigue life (millions of revolutions), complete
BEARING = [17.88, 28.92, 33.00, 41.52, 42.12, 45.60, 48.48, 51.84, 51.96, 54.12,
           55.56, 67.80, 68.64, 68.64, 68.88, 84.12, 93.12, 98.64, 105.12, 105.84,
           127.92, 128.04, 173.40]


def _qc(times, events=None):
    df = pd.DataFrame({"h": times})
    if events is not None:
        df["failed"] = events
        return mfgqc.load(df, measure="h").roles(time="h", event="failed")
    return mfgqc.load(df, measure="h").roles(time="h")


def test_weibull_mle_matches_published_bearing_anchor():
    fit = _qc(BEARING).life_fit(dist="weibull", method="mle")
    assert abs(fit.params["shape"] - 2.102) < 0.01
    assert abs(fit.params["scale"] - 81.88) < 0.1
    assert fit.n_fail == 23 and fit.n_susp == 0


def test_mttf_from_distribution_not_sample_mean():
    fit = _qc(BEARING).life_fit(dist="weibull")
    # Weibull MTTF = scale * Gamma(1 + 1/shape)
    from scipy.special import gamma
    assert abs(fit.mttf - fit.params["scale"] * gamma(1 + 1 / fit.params["shape"])) < 1e-6
    assert abs(fit.b50 - fit._frozen.ppf(0.5)) < 1e-9


def test_censored_mle_matches_scipy_censoreddata():
    from scipy.stats import CensoredData, weibull_min
    rng = np.random.default_rng(0)
    t = weibull_min.rvs(1.8, scale=120, size=40, random_state=rng)
    event = (rng.random(40) > 0.3).astype(int)          # ~30% suspensions
    t[event == 0] *= 0.7                                 # suspended earlier than failure
    fit = _qc(t, event).life_fit(dist="weibull", method="mle")
    sh, _loc, sc = weibull_min.fit(
        CensoredData(uncensored=t[event == 1], right=t[event == 0]), floc=0)
    assert abs(fit.params["shape"] - sh) < 1e-2
    assert abs(fit.params["scale"] - sc) < 1e-1


def test_lognormal_mle_matches_scipy():
    from scipy.stats import lognorm
    rng = np.random.default_rng(1)
    t = lognorm.rvs(0.5, scale=np.exp(4), size=60, random_state=rng)
    fit = _qc(t).life_fit(dist="lognormal", method="mle")
    s, _loc, scale = lognorm.fit(t, floc=0)
    assert abs(fit.params["sigma"] - s) < 1e-2
    assert abs(fit.params["mu"] - np.log(scale)) < 1e-2


def test_rank_regression_runs_and_is_close_to_mle():
    rr = _qc(BEARING).life_fit(dist="weibull", method="rankreg")
    assert rr.method == "rankreg"
    assert abs(rr.params["shape"] - 2.102) < 0.5        # rank-reg near MLE on a clean set
    assert rr.ppcc > 0.95


def test_competing_aic_and_distribution_flag():
    fit = _qc(BEARING).life_fit(dist="weibull")
    assert set(fit.competing_aic) == {"exponential", "weibull", "lognormal", "normal"}
    flag = next(a for a in fit.assumptions if a.name == "distribution_fit")
    assert flag.passed is True                          # bearing data: good Weibull fit


def test_weibull_shape_flags_nonconstant_rate():
    # wear-out (shape ~2) -> constant-rate / MTBF would mislead; flag fires.
    fit = _qc(BEARING).life_fit(dist="weibull")
    cf = next(a for a in fit.assumptions if a.name == "constant_failure_rate")
    assert cf.passed is False and "not constant" in cf.recommendation.lower()


def test_exponential_negative_control_weibull_shape_near_one():
    rng = np.random.default_rng(2)
    t = rng.exponential(50, 200)
    fit = _qc(t).life_fit(dist="weibull")
    assert abs(fit.params["shape"] - 1.0) < 0.2         # exponential -> Weibull shape ~ 1


def test_suspensions_are_not_dropped():
    rng = np.random.default_rng(3)
    t = mfgqc  # placeholder to keep linter calm
    base = np.r_[np.random.default_rng(3).weibull(2.0, 30) * 100]
    all_fail = _qc(base).life_fit(dist="weibull").params["scale"]
    # mark the longest 10 as suspended at their time -> a different (larger-scale) estimate
    ev = np.ones(30); ev[np.argsort(base)[-10:]] = 0
    with_susp = _qc(base, ev).life_fit(dist="weibull").params["scale"]
    assert abs(with_susp - all_fail) > 1e-3             # honoring suspensions changes the fit


def test_too_few_failures_refused():
    with pytest.raises(ValueError, match="too few"):
        _qc([100.0, 200.0], [1, 0]).life_fit(dist="weibull")


def test_views_render():
    fit = _qc(BEARING).life_fit(dist="weibull")
    for k in ("probability_plot", "survival", "hazard", "cdf"):
        assert fit.view(kind=k) is not None
