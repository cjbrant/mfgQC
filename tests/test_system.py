"""Slice 5: system reliability. Closed-form identities are self-verifying."""
import matplotlib; matplotlib.use("Agg")
import numpy as np, pytest, mfgqc
from scipy import stats


def test_series_is_product():
    r = mfgqc.reliability.series([0.99, 0.98, 0.97])
    assert abs(r.reliability - 0.99 * 0.98 * 0.97) < 1e-12


def test_parallel_is_one_minus_product_of_unreliabilities():
    r = mfgqc.reliability.parallel([0.8, 0.9])
    assert abs(r.reliability - (1 - 0.2 * 0.1)) < 1e-12


def test_n_identical_series_equals_R_to_the_n():
    r = mfgqc.reliability.series([0.95] * 4)
    assert abs(r.reliability - 0.95 ** 4) < 1e-12


def test_k_of_n_binomial_tail():
    r = mfgqc.reliability.k_of_n(k=2, n=3, reliability=0.9)
    assert abs(r.reliability - float(stats.binom.sf(1, 3, 0.9))) < 1e-12


def test_block_diagram_nested():
    # series of [0.99, parallel(0.8,0.8)]
    blocks = {"series": [0.99, {"parallel": [0.8, 0.8]}]}
    r = mfgqc.reliability.system(blocks)
    assert abs(r.reliability - 0.99 * (1 - 0.2 * 0.2)) < 1e-12
    assert r.n_components == 3


def test_independence_is_surfaced():
    r = mfgqc.reliability.series([0.99, 0.98])
    flag = next(a for a in r.assumptions if a.name == "independence")
    assert "independent" in flag.recommendation
    assert r.view() is not None
