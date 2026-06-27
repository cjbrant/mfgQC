"""Correctness: Weibull life fit vs scipy.stats.weibull_min (independent engine).

The reliability module's build oracle was R ``survreg``/``flexsurv``. scipy's
``weibull_min`` is a different, independent maximum-likelihood implementation, so
agreement here is a genuine cross-engine check rather than re-running the build
oracle. Data is freshly seeded; the oracle is computed in-test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import mfgqc


def test_weibull_mle_vs_scipy():
    """Complete-data Weibull MLE shape/scale vs scipy.weibull_min.fit(floc=0)."""
    rng = np.random.default_rng(42)
    t = stats.weibull_min.rvs(2.0, scale=100.0, size=60, random_state=rng)
    r = mfgqc.load(pd.DataFrame({"t": t}), measure="t").life_fit(dist="weibull", method="mle")
    shape, _loc, scale = stats.weibull_min.fit(t, floc=0)
    assert r.params["shape"] == pytest.approx(shape, rel=1e-4)
    assert r.params["scale"] == pytest.approx(scale, rel=1e-4)


def test_weibull_b10_vs_quantile():
    """B10 life vs the Weibull 10th-percentile quantile from the fitted params."""
    rng = np.random.default_rng(43)
    t = stats.weibull_min.rvs(1.5, scale=50.0, size=80, random_state=rng)
    r = mfgqc.load(pd.DataFrame({"t": t}), measure="t").life_fit(dist="weibull", method="mle")
    # B10 is the time by which 10% have failed: scale * (-ln(0.9))^(1/shape).
    shape, scale = r.params["shape"], r.params["scale"]
    b10_expected = scale * (-np.log(0.9)) ** (1.0 / shape)
    assert r.b10 == pytest.approx(b10_expected, rel=1e-6)
