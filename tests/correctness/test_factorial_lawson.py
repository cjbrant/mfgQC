"""Correctness: two-factor factorial ANOVA vs Lawson's printed worked example.

Source: J. Lawson, *Design and Analysis of Experiments with R* (CRC, 2017),
Section 3.5, Table 3.3 and pp. 65-66 -- the ethanol/CO-emission factorial of
Hunter (1983), distributed as ``COdata`` in the ``daewr`` package. The book prints
the ``aov`` ANOVA table and the grand mean. This is an independent Lawson example,
NOT one of the volt/chem/soup datasets used as mfgQC's DOE build oracles.

Lawson's printed ANOVA table (p. 65):

               Df  Sum Sq  Mean Sq  F value
    Eth         2   324.0    162.0    31.36
    Ratio       2   652.0    326.0    63.10
    Eth:Ratio   4   678.0    169.5    32.81
    Residuals   9    46.5      5.2

Grand mean (model.tables, p. 66): 72.83333.
"""

from __future__ import annotations

import pandas as pd
import pytest

import mfgqc

# Table 3.3: A = ethanol additions, B = air/fuel ratio, y = CO emissions (2 reps).
_CO = [
    (0.1, 14, 66), (0.1, 14, 62), (0.1, 15, 72), (0.1, 15, 67), (0.1, 16, 68),
    (0.1, 16, 66), (0.2, 14, 78), (0.2, 14, 81), (0.2, 15, 80), (0.2, 15, 81),
    (0.2, 16, 66), (0.2, 16, 69), (0.3, 14, 90), (0.3, 14, 94), (0.3, 15, 75),
    (0.3, 15, 78), (0.3, 16, 60), (0.3, 16, 58),
]


def _anova():
    df = pd.DataFrame(_CO, columns=["Eth", "Ratio", "CO"])
    df["Eth"] = df["Eth"].astype(str)
    df["Ratio"] = df["Ratio"].astype(str)
    return mfgqc.load(df, measure="CO").anova(factors=["Eth", "Ratio"], interaction=True)


def test_lawson_co_main_effects():
    """Lawson Table 3.3: Eth and Ratio sums of squares, df and F-ratios."""
    a = _anova()
    assert a.table["Eth"]["ss"] == pytest.approx(324.0, abs=1e-6)
    assert a.table["Eth"]["df"] == 2
    assert a.table["Eth"]["f"] == pytest.approx(31.36, abs=1e-2)
    assert a.table["Ratio"]["ss"] == pytest.approx(652.0, abs=1e-6)
    assert a.table["Ratio"]["df"] == 2
    assert a.table["Ratio"]["f"] == pytest.approx(63.10, abs=1e-2)


def test_lawson_co_interaction_and_error():
    """Lawson Table 3.3: Eth:Ratio interaction and residual sums of squares."""
    a = _anova()
    assert a.table["Eth:Ratio"]["ss"] == pytest.approx(678.0, abs=1e-6)
    assert a.table["Eth:Ratio"]["df"] == 4
    assert a.table["Eth:Ratio"]["f"] == pytest.approx(32.81, abs=1e-2)
    assert a.table["residual"]["ss"] == pytest.approx(46.5, abs=1e-6)
    assert a.table["residual"]["df"] == 9
    assert a.table["residual"]["ms"] == pytest.approx(5.1667, abs=1e-3)


def test_lawson_co_grand_mean():
    """Lawson p. 66 (model.tables): grand mean = 72.83333."""
    df = pd.DataFrame(_CO, columns=["Eth", "Ratio", "CO"])
    assert df["CO"].mean() == pytest.approx(72.83333, abs=1e-4)
