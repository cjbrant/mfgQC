"""Correctness: factorial-design effects vs statsmodels OLS (independent engine).

The DOE module's build oracles were Lawson's worked designs. statsmodels OLS is an
independent regression engine; for a two-level design coded as +/-1, the factor
*effect* equals twice the OLS regression coefficient (a standard identity). We fit
a full 2^3 factorial both ways on the same seeded data and check mfgQC's effects.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

import mfgqc


def _factorial_df():
    rng = np.random.default_rng(1)
    rows = []
    for a, b, c in itertools.product([-1, 1], repeat=3):
        for _ in range(2):  # 2 replicates
            y = 10 + 3 * a - 2 * b + 1.5 * a * b + rng.normal(scale=0.5)
            rows.append({"A": a, "B": b, "C": c, "y": y})
    return pd.DataFrame(rows)


def test_doe_effects_equal_twice_ols_coef():
    """2^3 factorial: each mfgQC effect == 2 x the statsmodels OLS coefficient."""
    df = _factorial_df()
    d = mfgqc.load(df, measure="y").doe(factors=["A", "B", "C"])
    m = smf.ols("y ~ A*B*C", data=df).fit()
    # statsmodels uses 'A:B' interaction naming, matching mfgQC's term keys.
    for term, effect in d.effect.items():
        assert effect == pytest.approx(2.0 * m.params[term], abs=1e-9), term


def test_doe_intercept_matches_ols():
    """The DOE grand mean (intercept) matches the statsmodels OLS intercept."""
    df = _factorial_df()
    d = mfgqc.load(df, measure="y").doe(factors=["A", "B", "C"])
    m = smf.ols("y ~ A*B*C", data=df).fit()
    assert d.intercept == pytest.approx(m.params["Intercept"], abs=1e-9)
    assert d.r_squared == pytest.approx(m.rsquared, abs=1e-9)
