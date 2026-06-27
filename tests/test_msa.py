"""MSA bias, linearity, and stability studies (AIAG MSA 4th ed.)."""

from __future__ import annotations

import matplotlib.figure as mfig
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import mfgqc

# AIAG MSA 4th ed. linearity raw data (Table III-B.4): 5 references x 12 trials.
_AIAG_LINEARITY = {
    2.00: [2.70, 2.50, 2.40, 2.50, 2.70, 2.30, 2.50, 2.50, 2.40, 2.40, 2.60, 2.40],
    4.00: [5.10, 3.90, 4.20, 5.00, 3.80, 3.90, 3.90, 3.90, 3.90, 4.00, 4.10, 3.80],
    6.00: [5.80, 5.70, 5.90, 5.90, 6.00, 6.10, 6.00, 6.10, 6.40, 6.30, 6.00, 6.10],
    8.00: [7.60, 7.70, 7.80, 7.70, 7.80, 7.80, 7.80, 7.70, 7.80, 7.50, 7.60, 7.70],
    10.00: [9.10, 9.30, 9.50, 9.30, 9.40, 9.50, 9.50, 9.50, 9.60, 9.20, 9.30, 9.40],
}


def _aiag_linearity_qc():
    rows = [{"ref": r, "m": v} for r, vals in _AIAG_LINEARITY.items() for v in vals]
    return mfgqc.load(pd.DataFrame(rows), measure="m")


# --------------------------------------------------------------------------- #
# Bias
# --------------------------------------------------------------------------- #
def test_bias_reproduces_aiag_table_iiib3():
    # AIAG bias example: reference 6.01, mean 6.021, sd 0.2048, n=100 -> bias 0.011,
    # t = bias/SE = 0.5371, 95% CI ~ (-0.030, 0.052), accept. The CI/t depend only
    # on (bias, sd, n), so a sample matched to mean=6.021, sd=0.2048 reproduces them.
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, 100)
    z = (z - z.mean()) / z.std(ddof=1)
    vals = z * 0.2048 + 6.021
    qc = mfgqc.load(pd.DataFrame({"m": vals}), measure="m")
    b = qc.bias_study(reference=6.01)
    assert b.bias == pytest.approx(0.011, abs=1e-3)
    assert b.sigma_repeat == pytest.approx(0.2048, abs=1e-4)
    assert b.t_stat == pytest.approx(0.5371, abs=2e-3)
    assert b.ci[0] == pytest.approx(-0.0299, abs=1e-3)
    assert b.ci[1] == pytest.approx(0.0519, abs=1e-3)
    assert b.verdict == "acceptable"
    assert b.ci[0] <= 0 <= b.ci[1]


def test_bias_flags_a_real_bias():
    rng = np.random.default_rng(3)
    vals = rng.normal(10.5, 0.1, 30)  # ref 10.0 -> bias ~0.5 with tight repeatability
    qc = mfgqc.load(pd.DataFrame({"m": vals}), measure="m")
    b = qc.bias_study(reference=10.0)
    assert b.verdict == "not acceptable"
    assert not (b.ci[0] <= 0 <= b.ci[1])


# --------------------------------------------------------------------------- #
# Linearity
# --------------------------------------------------------------------------- #
def test_linearity_reproduces_aiag_table_iiib4():
    # AIAG's deliberately failing linearity example -> exact published statistics.
    lin = _aiag_linearity_qc().linearity_study(reference="ref")
    assert lin.slope == pytest.approx(-0.1317, abs=1e-4)
    assert lin.intercept == pytest.approx(0.7367, abs=1e-4)
    assert lin.t_slope == pytest.approx(-12.043, abs=2e-3)
    assert lin.t_intercept == pytest.approx(10.158, abs=2e-3)
    assert lin.r_squared == pytest.approx(0.7143, abs=1e-3)
    assert lin.df == 58  # g*m - 2 = 60 - 2
    assert lin.verdict == "not acceptable"
    # per-reference bias averages (Table III-B.5)
    assert lin.ref_bias == pytest.approx(
        (0.491667, 0.125, 0.025, -0.291667, -0.616667), abs=1e-5)


def test_linearity_tstats_match_ols():
    lin = _aiag_linearity_qc().linearity_study(reference="ref")
    refs = np.array([r for r, vals in _AIAG_LINEARITY.items() for _ in vals])
    meas = np.array([v for _, vals in _AIAG_LINEARITY.items() for v in vals])
    reg = stats.linregress(refs, meas - refs)
    assert lin.t_slope == pytest.approx(reg.slope / reg.stderr, rel=1e-6)


def test_linearity_acceptable_when_flat_and_zero():
    refs = np.repeat([2, 4, 6, 8, 10], 12).astype(float)
    rng = np.random.default_rng(5)
    meas = refs + rng.normal(0, 0.05, 60)  # ~zero bias across the whole range
    qc = mfgqc.load(pd.DataFrame({"m": meas, "ref": refs}), measure="m")
    lin = qc.linearity_study(reference="ref")
    assert lin.verdict == "acceptable"


def test_linearity_reference_mapping():
    parts = np.repeat([1, 2, 3], 10)
    rng = np.random.default_rng(7)
    meas = np.repeat([2.0, 4.0, 6.0], 10) + rng.normal(0, 0.05, 30)
    qc = mfgqc.load(pd.DataFrame({"m": meas, "p": parts}),
                             measure="m", roles={"part": "p"})
    lin = qc.linearity_study(reference={1: 2.0, 2: 4.0, 3: 6.0})
    assert len(lin.references) == 3


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #
def test_stability_clean_series_is_stable():
    # In-control reference series (normal variation, no special causes).
    rng = np.random.default_rng(0)
    rows = [{"sg": i + 1, "x": float(v)} for i in range(25) for v in rng.normal(10, 1, 5)]
    qc = mfgqc.load(pd.DataFrame(rows), measure="x",
                             roles={"subgroup": "sg"}, subgroup_size=5)
    st = qc.stability_study()
    assert st.stable is True
    assert st.n_signals == 0
    assert st.verdict == "stable"


def test_stability_individuals_default_imr():
    # Individual readings over time, NO subgroup role/size -> defaults to I-MR (not an error).
    rng = np.random.default_rng(20260618)
    stab = np.concatenate([rng.normal(50, 1, 15), rng.normal(50, 1, 15) + np.linspace(0, 2, 15)])
    qc = mfgqc.load(pd.DataFrame({"m": stab, "order": range(1, 31)}), measure="m")
    st = qc.stability_study()
    assert st.chart.kind == "i_mr"
    assert st.n_signals > 0  # the drifting half trips signals
    # a genuinely stable individual series
    qc2 = mfgqc.load(pd.DataFrame({"m": rng.normal(50, 1, 30)}), measure="m")
    st2 = qc2.stability_study()
    assert st2.chart.kind == "i_mr"


def test_stability_drift_is_flagged():
    rng = np.random.default_rng(0)
    rows = [{"sg": i + 1, "x": float(v) + i * 0.4}
            for i in range(25) for v in rng.normal(10, 0.5, 5)]
    qc = mfgqc.load(pd.DataFrame(rows), measure="x",
                             roles={"subgroup": "sg"}, subgroup_size=5)
    st = qc.stability_study()
    assert st.stable is False
    assert st.n_signals > 0
    assert st.n_signals == len(st.chart.violations)


# --------------------------------------------------------------------------- #
# Dashboard-ready: views + flat summaries
# --------------------------------------------------------------------------- #
def test_views_return_figures_and_summaries_flat():
    rng = np.random.default_rng(1)
    qc = mfgqc.load(pd.DataFrame({"m": rng.normal(6.02, 0.1, 50)}), measure="m")
    bias = qc.bias_study(reference=6.0)
    assert isinstance(bias.view(), mfig.Figure)
    assert all(not isinstance(v, (list, dict)) for v in bias.summary().values())

    refs = np.repeat([2, 4, 6, 8, 10], 10).astype(float)
    meas = refs + rng.normal(0, 0.1, 50)
    q2 = mfgqc.load(pd.DataFrame({"m": meas, "ref": refs}), measure="m")
    lin = q2.linearity_study(reference="ref")
    assert isinstance(lin.view(), mfig.Figure)
    assert "verdict" in lin.summary()
