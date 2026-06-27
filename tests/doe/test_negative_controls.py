"""Negative controls. A guardrail that fires on everything is as broken as one
that never fires:

- a design with no real effects yields no active effects and a clean half-normal;
- a replicated design that meets its assumptions passes the residual checks;
- a fractional design always surfaces its aliasing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mfgqc
from mfgqc.doe import generate as gen


def _coded_df(k, y):
    m = gen.coded_full_matrix(k)
    cols = {chr(ord("A") + j): m[:, j] for j in range(k)}
    cols["y"] = y
    return pd.DataFrame(cols)


def test_no_real_effects_yields_no_active():
    # pure noise on an unreplicated 2^4: Lenth must not manufacture active effects.
    rng = np.random.default_rng(2024)
    y = rng.normal(50, 1.0, 16)
    df = _coded_df(4, y)
    res = mfgqc.load(df, measure="y").doe(factors=["A", "B", "C", "D"])
    assert res.fit_kind == "saturated"
    assert res.active == ()                       # nothing clears SME
    assert len(res.possibly_active) <= 1          # clean half-normal line


def test_clean_replicated_passes_residual_checks():
    # a real planar response + small gaussian noise on a replicated 2^3
    rng = np.random.default_rng(11)
    m = gen.coded_full_matrix(3)
    m2 = np.vstack([m, m])
    y = 100 + 5 * m2[:, 0] + 3 * m2[:, 1] + rng.normal(0, 0.4, m2.shape[0])
    df = pd.DataFrame({"A": m2[:, 0], "B": m2[:, 1], "C": m2[:, 2], "y": y})
    res = mfgqc.load(df, measure="y").doe(factors=["A", "B", "C"])
    assert res.fit_kind == "replicated"
    norm = next(a for a in res.assumptions if a.name == "normality")
    assert norm.passed                            # residuals are clean


def test_fractional_always_surfaces_aliasing():
    d = mfgqc.design.fractional_factorial(["A", "B", "C", "D", "E"], fraction="1/2")
    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, 16)
    df = pd.DataFrame({f: d.matrix[:, i] for i, f in enumerate("ABCDE")})
    df["y"] = y
    res = mfgqc.load(df, measure="y").doe(design=d, order=2)
    assert res.aliases                            # alias list present
    alias_flag = next(a for a in res.adequacy if a.name == "aliasing")
    assert alias_flag.passed is False             # warned, never silent


def test_full_factorial_reports_no_aliasing():
    # the contrast: a full factorial is unconfounded -> aliasing flag passes.
    rng = np.random.default_rng(1)
    df = _coded_df(3, rng.normal(0, 1, 8))
    res = mfgqc.load(df, measure="y").doe(factors=["A", "B", "C"], order=2)
    alias_flag = next(a for a in res.adequacy if a.name == "aliasing")
    assert alias_flag.passed is True


def test_no_crossings_when_no_interactions():
    # additive response: no two-factor interaction should be flagged as a crossing.
    m = gen.coded_full_matrix(3)
    y = 50 + 8 * m[:, 0] + 6 * m[:, 1] + 4 * m[:, 2]
    df = pd.DataFrame({"A": m[:, 0], "B": m[:, 1], "C": m[:, 2], "y": y})
    res = mfgqc.load(df, measure="y").doe(factors=["A", "B", "C"], order=2)
    assert not any(res.crossings().values())
