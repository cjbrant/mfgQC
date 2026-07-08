"""compare() - swap symmetry (T1.9) and the identical-process limit (T3.4)."""
from __future__ import annotations

import math

import numpy as np

from mfgqc.bayes.capability import capability_from_values
from mfgqc.bayes.comparison import compare


def test_t1_9_compare_swap_complements_exactly():
    """T1.9: swapping the arguments complements every probability exactly, because
    the child seed assigned to each result (by provenance digest) is independent of
    argument order, so both calls reuse identical draws."""
    a = capability_from_values(np.random.default_rng(1).normal(25.0, 0.5, 80),
                               lower=23.0, upper=27.0, seed=1, draws=50_000)
    b = capability_from_values(np.random.default_rng(2).normal(25.2, 0.4, 80),
                               lower=23.0, upper=27.0, seed=2, draws=50_000)
    ab = compare(a, b, seed=99, draws=100_000)
    ba = compare(b, a, seed=99, draws=100_000)

    assert abs(ab.prob_mean_gt + ba.prob_mean_gt - 1.0) <= 1e-12
    assert abs(ab.prob_sd_lt + ba.prob_sd_lt - 1.0) <= 1e-12
    assert abs(ab.prob_ppk_gt + ba.prob_ppk_gt - 1.0) <= 1e-12


def test_t3_4_identical_posteriors_give_half():
    """T3.4: two analyses of the same data have identical posteriors, so every
    P(B better than A) is 0.5 within MC tolerance (all orderings)."""
    y = np.random.default_rng(5).normal(25.0, 0.5, 120)
    a = capability_from_values(y, lower=23.0, upper=27.0, seed=1, draws=100)
    b = capability_from_values(y, lower=23.0, upper=27.0, seed=2, draws=100)
    c = compare(a, b, seed=7, draws=500_000)

    tol = 5.0 * math.sqrt(0.25 / 500_000)
    assert abs(c.prob_mean_gt - 0.5) <= tol
    assert abs(c.prob_sd_lt - 0.5) <= tol
    assert abs(c.prob_ppk_gt - 0.5) <= tol
