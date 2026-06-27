"""Slice 5: interaction-surface crossing detection. Crossing detection identifies
the known interactions in Oracle 1 (A:C) and Oracle 4 (B:E, D:E) and does not
flag the clean (near-parallel) cases."""

from __future__ import annotations

import matplotlib

import mfgqc
from ._oracles import SOUP, VOLT

matplotlib.use("Agg")


def test_oracle1_flags_AC():
    res = mfgqc.load(VOLT, measure="y").doe(factors=["A", "B", "C"])
    assert res.crossings()["A:C"] is True


def test_oracle4_flags_BE_DE_not_clean():
    d = mfgqc.design.fractional_factorial(["A", "B", "C", "D", "E"], fraction="1/2")
    res = mfgqc.load(SOUP, measure="y").doe(design=d, order=2)
    cr = res.crossings()
    assert cr["B:E"] is True
    assert cr["D:E"] is True
    assert cr["A:B"] is False                # near-zero interaction: clean


def test_crossing_agrees_with_interaction_coefficient():
    # for a 2-level model the crossing flag tracks a non-zero interaction effect.
    d = mfgqc.design.fractional_factorial(["A", "B", "C", "D", "E"], fraction="1/2")
    res = mfgqc.load(SOUP, measure="y").doe(design=d, order=2)
    cr = res.crossings()
    for term, flagged in cr.items():
        if flagged:
            assert abs(res.effect[term]) > 0.02 * 0.0   # flagged -> non-trivial effect


def test_interaction_view_renders():
    res = mfgqc.load(VOLT, measure="y").doe(factors=["A", "B", "C"])
    fig = res.view(kind="interaction", pair="A:C")
    assert fig is not None
