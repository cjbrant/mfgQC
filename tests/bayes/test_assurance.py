"""assurance() - sample-size decision curve (T1.11 monotonicity)."""
from __future__ import annotations

import numpy as np

from mfgqc.bayes.capability import capability_from_values
from mfgqc.bayes.decisions import assurance


def test_t1_11_assurance_non_decreasing_in_n():
    """T1.11: on a fixed capable process the assurance (probability the future
    analysis concludes capability) is non-decreasing in the sample size, within
    MC tolerance. Deterministic given seed."""
    y = np.random.default_rng(11).normal(25.0, 0.010, 40)
    r = capability_from_values(y, lower=24.95, upper=25.05, seed=1, draws=100)
    res = assurance(r, target=("ppk", 1.33), decide=(0.9, 0.1),
                    n_grid=(20, 40, 80, 160), sims=1000, inner_draws=2000, seed=7)

    a = res.assurance
    assert len(a) == 4
    for i in range(1, len(a)):
        assert a[i] >= a[i - 1] - 0.03  # non-decreasing within MC tolerance
    assert a[-1] >= a[0]  # net increase over the grid


def test_assurance_is_deterministic_given_seed():
    y = np.random.default_rng(11).normal(25.0, 0.010, 40)
    r = capability_from_values(y, lower=24.95, upper=25.05, seed=1, draws=100)
    kw = dict(target=("ppk", 1.33), n_grid=(20, 80), sims=500, inner_draws=1000, seed=3)
    assert assurance(r, **kw).assurance == assurance(r, **kw).assurance
