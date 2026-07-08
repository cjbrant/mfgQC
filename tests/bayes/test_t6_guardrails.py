"""T6 - guardrails for mfgqc.bayes (plan D5).

Every guardrail is an AssumptionCheck.recommendation or a raise, matching the
existing idiom (assumptions.py); there is no warnings.warn and no Warning class.
"""
from __future__ import annotations

import numpy as np
import pytest

from mfgqc.bayes._results import fit_normal
from mfgqc.bayes.priors import NormalPrior


def _data(seed: int = 0, n: int = 40, mu: float = 25.0, sd: float = 0.1) -> np.ndarray:
    return np.random.default_rng(seed).normal(mu, sd, size=n)


def _check(result, name):
    return next(a for a in result.assumptions if a.name == name)


def test_t6_1_prior_weight_disclosure_always_present_and_warns_above_half():
    """G1: prior-weight w = k0/(k0+n) is always disclosed (in statistic/magnitude);
    the recommendation fires only when w > 0.5."""
    heavy = fit_normal(_data(n=10), NormalPrior(25.0, 100, 100, 0.25), seed=1, draws=100)
    pw = _check(heavy, "prior_weight")
    assert pw.magnitude is not None  # disclosure always present
    assert pw.statistic > 0.5
    assert pw.passed is False and pw.recommendation is not None

    light = fit_normal(_data(n=50), NormalPrior(25.0, 1, 1, 0.25), seed=1, draws=100)
    pw2 = _check(light, "prior_weight")
    assert pw2.magnitude is not None  # still disclosed
    assert pw2.passed is True and pw2.recommendation is None


def test_t6_2_prior_data_conflict_warns_when_far_agrees_when_close():
    """G2: mu0 many prior-predictive SDs from ybar warns; agreement does not.
    Prior predictive: ybar ~ t_{nu0}(mu0, s20*(1/k0 + 1/n))."""
    y = _data(seed=0, n=40, mu=25.0, sd=0.1)

    conflict = fit_normal(y, NormalPrior(30.0, 50, 50, 0.01), seed=1, draws=100)
    c = _check(conflict, "prior_data_conflict")
    assert c.statistic >= 5.0
    assert c.passed is False and c.recommendation is not None

    agree = fit_normal(y, NormalPrior(25.0, 50, 50, 0.01), seed=1, draws=100)
    a = _check(agree, "prior_data_conflict")
    assert a.statistic < 5.0
    assert a.passed is True and a.recommendation is None


def test_t6_3_n1_raises_and_small_sample_warns():
    """G3: n=1 raises (cannot form the sample variance); a small sample warns."""
    with pytest.raises(ValueError):
        fit_normal(np.array([25.0]), NormalPrior(25.0, 10, 10, 0.25), seed=1, draws=100)

    small = fit_normal(_data(n=5), NormalPrior(25.0, 10, 10, 0.25), seed=1, draws=100)
    ss = _check(small, "small_sample")
    assert ss.passed is False and ss.recommendation is not None
    assert ss.reliability == "low_power"

    big = fit_normal(_data(n=50), NormalPrior(25.0, 10, 10, 0.25), seed=1, draws=100)
    ss2 = _check(big, "small_sample")
    assert ss2.passed is True and ss2.recommendation is None
