"""Gage R&R oracles (AIAG MSA 4th ed.)."""

from __future__ import annotations

import pytest


def test_aiag_xbar_r_components(aiag_qc):
    """Oracle 4a: Average & Range method EV/AV/GRR/PV/TV/ndc + percentages."""
    grr = aiag_qc.gage_rr(method="xbar_r")
    assert grr.ev == pytest.approx(0.20188, abs=3e-3)
    assert grr.av == pytest.approx(0.22963, abs=3e-3)
    assert grr.grr == pytest.approx(0.30575, abs=3e-3)
    assert grr.pv == pytest.approx(1.10456, abs=5e-3)
    assert grr.tv == pytest.approx(1.14610, abs=5e-3)
    assert grr.ndc == 5

    assert grr.pct_study["EV"] == pytest.approx(17.62, abs=0.2)
    assert grr.pct_study["AV"] == pytest.approx(20.04, abs=0.2)
    assert grr.pct_study["GRR"] == pytest.approx(26.68, abs=0.2)
    assert grr.pct_study["PV"] == pytest.approx(96.38, abs=0.2)


def test_aiag_anova_table(aiag_qc):
    """Oracle 4b: the ANOVA SS/MS/F table to reported precision."""
    grr = aiag_qc.gage_rr(method="anova")
    t = grr.anova_table

    assert t["operator"]["ss"] == pytest.approx(3.1673, rel=2e-3)
    assert t["operator"]["ms"] == pytest.approx(1.58363, rel=2e-3)
    assert t["operator"]["f"] == pytest.approx(34.44, rel=5e-3)

    assert t["parts"]["ss"] == pytest.approx(88.3619, rel=2e-3)
    assert t["parts"]["ms"] == pytest.approx(9.81799, rel=2e-3)
    assert t["parts"]["f"] == pytest.approx(213.52, rel=5e-3)

    assert t["interaction"]["ss"] == pytest.approx(0.3590, rel=5e-3)
    assert t["interaction"]["ms"] == pytest.approx(0.01994, rel=5e-3)
    assert t["interaction"]["f"] == pytest.approx(0.434, rel=1e-2)

    assert t["equipment"]["ss"] == pytest.approx(2.7589, rel=2e-3)
    assert t["equipment"]["ms"] == pytest.approx(0.04598, rel=2e-3)

    assert t["total"]["ss"] == pytest.approx(94.6471, rel=1e-3)


def test_aiag_anova_components_and_ndc(aiag_qc):
    """Oracle 4b: variance components, std devs, ndc=4 (truncated), pooled model."""
    grr = aiag_qc.gage_rr(method="anova")

    assert grr.pooled is True  # interaction not significant -> pooled into error
    assert grr.var_repeat == pytest.approx(0.039973, rel=3e-3)
    assert grr.var_oper == pytest.approx(0.051455, rel=3e-3)
    assert grr.var_part == pytest.approx(1.086446, rel=3e-3)

    assert grr.ev == pytest.approx(0.199933, rel=3e-3)
    assert grr.av == pytest.approx(0.226838, rel=3e-3)
    assert grr.grr == pytest.approx(0.302373, rel=3e-3)
    assert grr.pv == pytest.approx(1.042327, rel=3e-3)
    assert grr.tv == pytest.approx(1.085, rel=3e-3)

    assert grr.ndc == 4  # AIAG truncates 4.861 -> 4

    # %study components do NOT sum to 100 (root-sum-square), so this must NOT be enforced.
    total = grr.pct_study["EV"] + grr.pct_study["AV"] + grr.pct_study["PV"]
    assert total != pytest.approx(100.0, abs=1.0)


def test_unbalanced_design_errors(aiag_qc):
    # drop one measurement -> unbalanced -> legible error, not wrong numbers
    frame = aiag_qc.frame.iloc[1:]  # remove a row
    import mfgqc
    qc = mfgqc.load(
        frame, measure="y",
        roles={"part": "part", "operator": "operator", "replicate": "trial"},
    )
    with pytest.raises(ValueError, match="balanced") as exc:
        qc.gage_rr(method="anova")
    # Bug 2: the message identifies HOW it's unbalanced (trials/part by operator)
    assert "trials/part" in str(exc.value)
