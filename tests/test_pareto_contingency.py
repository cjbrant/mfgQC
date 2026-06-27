"""Tests for mfgqc.pareto: Pareto analysis + chi-square / contingency.

Oracles verified against scipy:
- contingency [[30,20],[15,35]] (correction=False): chi2=9.091, p=0.0026,
  dof=1, Cramer's V=0.3015.
- Pareto {A:50, B:30, C:15, D:5}: cum%=50,80,95,100; vital_few=(A,B).
"""

from __future__ import annotations

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

import mfgqc.pareto_analysis as pc
from mfgqc.pareto_analysis import ContingencyResult, ParetoResult, contingency, pareto


# --------------------------------------------------------------------------- #
# Pareto oracle
# --------------------------------------------------------------------------- #
def test_pareto_oracle_series():
    s = pd.Series({"A": 50, "B": 30, "C": 15, "D": 5})
    res = pareto(s)
    assert isinstance(res, ParetoResult)
    assert res.categories == ("A", "B", "C", "D")
    assert res.counts == (50, 30, 15, 5)
    assert res.cum_pct == pytest.approx([50.0, 80.0, 95.0, 100.0])
    assert res.vital_few == ("A", "B")
    assert res.threshold == 0.80
    assert res.assumptions == []  # pure descriptive


def test_pareto_dataframe_with_count_column():
    df = pd.DataFrame({"defect": ["A", "B", "C", "D"], "n": [50, 30, 15, 5]})
    res = pareto(df, category="defect", count="n")
    assert res.categories == ("A", "B", "C", "D")
    assert res.cum_pct == pytest.approx([50.0, 80.0, 95.0, 100.0])
    assert res.vital_few == ("A", "B")


def test_pareto_dataframe_frequency_counts():
    df = pd.DataFrame({"defect": ["A"] * 50 + ["B"] * 30 + ["C"] * 15 + ["D"] * 5})
    res = pareto(df, category="defect")
    assert res.categories == ("A", "B", "C", "D")
    assert res.counts == (50, 30, 15, 5)
    assert res.vital_few == ("A", "B")


def test_pareto_summary_is_flat():
    res = pareto(pd.Series({"A": 50, "B": 30, "C": 15, "D": 5}))
    summary = res.summary()
    assert isinstance(summary, dict)
    assert all(not isinstance(v, (dict, list, tuple)) for v in summary.values())
    assert summary["total"] == 100
    assert summary["n_categories"] == 4
    assert summary["vital_few_count"] == 2
    assert summary["top_category"] == "A"
    assert summary["top_category_pct"] == pytest.approx(50.0)


def test_pareto_view_returns_figure():
    res = pareto(pd.Series({"A": 50, "B": 30, "C": 15, "D": 5}))
    fig = res.view()
    assert isinstance(fig, matplotlib.figure.Figure)


# --------------------------------------------------------------------------- #
# Contingency oracle
# --------------------------------------------------------------------------- #
def test_contingency_oracle():
    res = contingency([[30, 20], [15, 35]])
    assert isinstance(res, ContingencyResult)
    assert res.chi2 == pytest.approx(9.091, abs=1e-3)
    assert res.p_value == pytest.approx(0.0026, abs=1e-3)
    assert res.dof == 1
    assert res.cramers_v == pytest.approx(0.3015, abs=1e-3)
    assert res.n == 100
    # Expected table from the marginals (row sums 50/50, col sums 45/55, N=100).
    np.testing.assert_allclose(res.expected, [[22.5, 27.5], [22.5, 27.5]], atol=1e-9)


def test_independence_matches_contingency():
    # Raw two-column DataFrame that yields the same table [[30,20],[15,35]].
    rows = (
        [{"group": "X", "outcome": "yes"}] * 30
        + [{"group": "X", "outcome": "no"}] * 20
        + [{"group": "Y", "outcome": "yes"}] * 15
        + [{"group": "Y", "outcome": "no"}] * 35
    )
    df = pd.DataFrame(rows)
    res = pc.test_independence(df, row="group", col="outcome")
    assert res.chi2 == pytest.approx(9.091, abs=1e-3)
    assert res.p_value == pytest.approx(0.0026, abs=1e-3)
    assert res.dof == 1
    assert res.cramers_v == pytest.approx(0.3015, abs=1e-3)
    assert res.n == 100


def test_expected_count_assumption_fails_small():
    # A table with a tiny row marginal forces a small expected cell (min ~1.48).
    table = [[1, 2], [40, 40]]
    res = contingency(table)
    check = next(a for a in res.assumptions if a.name == "expected_count")
    assert check.passed is False
    assert check.recommendation is not None
    assert "Fisher" in check.recommendation
    assert check.magnitude < 5.0
    assert check.magnitude_label == "min expected count"


def test_expected_count_assumption_passes_large():
    res = contingency([[30, 20], [15, 35]])
    check = next(a for a in res.assumptions if a.name == "expected_count")
    assert check.passed is True
    assert check.recommendation is None
    assert check.magnitude >= 5.0


def test_contingency_summary_is_flat():
    res = contingency([[30, 20], [15, 35]])
    summary = res.summary()
    assert isinstance(summary, dict)
    assert all(not isinstance(v, (dict, list, tuple, np.ndarray)) for v in summary.values())
    assert summary["dof"] == 1
    assert summary["n"] == 100
    assert summary["chi2"] == pytest.approx(9.091, abs=1e-3)


def test_contingency_view_returns_figure():
    res = contingency([[30, 20], [15, 35]])
    fig = res.view()
    assert isinstance(fig, matplotlib.figure.Figure)


def test_contingency_report_renders():
    # The base QCResult.report() must work (exercises _title/_summary_lines).
    res = contingency([[30, 20], [15, 35]])
    text = res.report()
    assert "Chi-Square" in text
    assert "PASS" in text or "FAIL" in text
