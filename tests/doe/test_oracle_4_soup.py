"""Slice 3 gate: regular fractional 2^(5-1) (Lawson soup), E=ABCD. Generation,
generators, defining relation, resolution V, the full alias list, and the
fractional analysis: 16 coefficients to 5 dp.

Active set: the published verdict {E, B:E, D:E} is the half-normal visual read.
Lenth's automated margin places E and B:E in the possibly-active band and D:E
just under the individual margin (pseudo-t 2.40 vs 2.57): D:E is the borderline
third point that lifts off the half-normal line. We assert the principled Lenth
classification AND that the three lift-off points (top-3 |effect|) are exactly
{E, B:E, D:E}, which is what the half-normal plot shows.
"""

from __future__ import annotations

import numpy as np

import mfgqc
from ._oracles import SOUP, SOUP_ALIAS, SOUP_COEF


def _design():
    return mfgqc.design.fractional_factorial(["A", "B", "C", "D", "E"], fraction="1/2")


def _res():
    return mfgqc.load(SOUP, measure="y").doe(design=_design(), order=2)


def test_generation_E_equals_ABCD():
    d = _design()
    # E column of the coded matrix equals the product A*B*C*D
    A, B, C, D, E = (d.matrix[:, i] for i in range(5))
    assert np.array_equal(E, A * B * C * D)


def test_generators_defining_relation_resolution():
    d = _design()
    assert d.generators == ("E=ABCD",)
    assert "ABCDE" in d.defining_relation
    assert d.resolution == 5


def test_alias_list_matches_oracle():
    res = _res()
    assert list(res.aliases) == SOUP_ALIAS


def test_sixteen_coefficients_match_to_5dp():
    res = _res()
    assert abs(res.intercept - SOUP_COEF["intercept"]) < 5e-6
    for term, exp in SOUP_COEF.items():
        if term == "intercept":
            continue
        assert abs(res.coef[term] - exp) < 5e-6, (term, res.coef[term], exp)


def test_halfnormal_liftoff_points_are_E_BE_DE():
    res = _res()
    top3 = sorted(res.terms, key=lambda t: -abs(res.effect[t]))[:3]
    assert set(top3) == {"E", "B:E", "D:E"}


def test_lenth_classification_is_principled():
    res = _res()
    # Lenth SME-active is empty; E and B:E are possibly-active; D:E is borderline
    # (just below the individual margin), surfaced via the half-normal view.
    assert set(res.possibly_active) == {"E", "B:E"}
    assert abs(res.lenth.pseudo_t["D:E"]) < abs(res.lenth.me / res.lenth.pse)


def test_alias_attribution_lower_order():
    res = _res()
    # D:E is aliased with A:B:C under hierarchical ordering; the 2fi is attributed.
    assert res.alias_of["D:E"] == "ABC"
