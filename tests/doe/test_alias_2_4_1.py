"""No-package alias sub-oracle: the 2^(4-1) design with generator D=ABC,
defining relation I=ABCD, gives the alias pairs A+BCD, ..., AD+BC (Lawson p.197).
The alias-list construction must reproduce this by multiplying the defining word
into each effect."""

from __future__ import annotations

import numpy as np

from mfgqc.doe import alias as al
from mfgqc.doe import generate as gen

EXPECTED_2_4_1 = ["A = BCD", "B = ACD", "C = ABD", "D = ABC",
                  "AB = CD", "AC = BD", "AD = BC"]


def test_2_4_1_alias_pairs():
    factors = ["A", "B", "C", "D"]
    group = al.defining_group([al.parse_word("ABCD")])
    assert al.resolution(group) == 4
    assert al.alias_list(factors, group) == EXPECTED_2_4_1


def test_2_5_1_resolution_V():
    factors = ["A", "B", "C", "D", "E"]
    group = al.defining_group([al.parse_word("ABCDE")])
    assert al.resolution(group) == 5
    # main effects aliased with 4-factor interactions
    assert "A = BCDE" in al.alias_list(factors, group)


def test_word_product_is_symmetric_difference():
    assert al.multiply(al.parse_word("AB"), al.parse_word("BC")) == al.parse_word("AC")
    assert al.multiply(al.parse_word("ABCD"), al.parse_word("ABCD")) == al.parse_word("I")


def test_generator_column_is_product():
    m = gen.coded_full_matrix(4)
    e = gen.product_column(m, [0, 1, 2, 3])
    assert np.array_equal(e, m[:, 0] * m[:, 1] * m[:, 2] * m[:, 3])
