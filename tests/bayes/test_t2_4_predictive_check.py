"""T2.4 - posterior predictive check (BDA3 sec 6.3), T(y) = min(y).

The specific speed-of-light Bayesian p-value is not reproducible: BDA3 sec 6.3
reports the min check graphically (Figure 6.3, smallest observation -44) and does
not tabulate Newcomb's 66 values, so that oracle is skipped with an explicit
reason. The predictive-check machinery is validated by a hand computation (T1.12
style) and by reproducing BDA3's qualitative finding on controlled data: a planted
low outlier makes the min check extreme while the mean check stays ordinary.
"""
from __future__ import annotations

import numpy as np
import pytest

from mfgqc.bayes.monitoring import predictive_check

from ._oracles import NEWCOMB_DATA, NEWCOMB_MIN, SPEED_OF_LIGHT


def test_t2_4_pvalue_matches_hand_computation():
    """The reported Bayesian p-value equals P(T_rep >= T_obs) recomputed from the
    same normative posterior-predictive draws."""
    y = np.random.default_rng(1).normal(0.0, 1.0, 40)
    R, seed = 300, 4
    res = predictive_check(y, statistic="min", R=R, seed=seed)

    n = y.size
    ybar, s2, nu = y.mean(), y.var(ddof=1), n - 1
    rng = np.random.default_rng(seed)
    sig2 = nu * s2 / rng.chisquare(nu, R)
    mu = rng.normal(ybar, np.sqrt(sig2 / n))
    reps = rng.normal(mu[:, None], np.sqrt(sig2)[:, None], size=(R, n))
    expected = float((reps.min(1) >= y.min()).mean())
    assert res.p_bayes == expected


def test_t2_4_min_check_flags_outlier_mean_check_does_not():
    """BDA3 sec 6.3 qualitative finding on reproducible data: a low outlier makes
    the min statistic extreme (model misfit) but leaves the mean statistic ordinary."""
    rng = np.random.default_rng(42)
    clean = rng.normal(0.0, 1.0, 66)
    outlier = clean.copy()
    outlier[0] = -6.0

    assert predictive_check(clean, statistic="min", R=20000, seed=3).p_two_sided > 0.1
    assert predictive_check(outlier, statistic="min", R=20000, seed=3).p_two_sided < 0.01
    assert predictive_check(outlier, statistic="mean", R=20000, seed=3).p_two_sided > 0.1


def test_t2_4_newcomb_min_is_the_transcribed_extreme():
    """The one transcribable BDA3 fact: Newcomb's minimum is -44, far below what the
    normal fit to the (summary-matched) data predicts, so the min check is extreme."""
    # Reconstruct a sample matching BDA3 sec 3.2 summaries (n=66, ybar=26.2, s=10.8)
    z = np.random.default_rng(0).normal(0, 1, SPEED_OF_LIGHT["n"])
    z = (z - z.mean()) / z.std(ddof=1)
    y = SPEED_OF_LIGHT["ybar"] + SPEED_OF_LIGHT["s"] * z
    y[np.argmin(y)] = NEWCOMB_MIN  # plant the transcribed minimum
    res = predictive_check(y, statistic="min", R=20000, seed=6)
    assert res.t_observed == NEWCOMB_MIN
    assert res.p_two_sided < 0.05


@pytest.mark.skipif(NEWCOMB_DATA is None,
                    reason="BDA3 sec 6.3 reports the speed-of-light min check graphically "
                           "(Fig 6.3), prints no numeric Bayesian p-value, and does not tabulate "
                           "Newcomb's 66 values; the exact oracle p-value is not reproducible.")
def test_t2_4_newcomb_min_pvalue_exact():
    y = np.asarray(NEWCOMB_DATA, dtype=float)
    res = predictive_check(y, statistic="min", R=20000, seed=6)
    assert res.p_two_sided < 0.05
