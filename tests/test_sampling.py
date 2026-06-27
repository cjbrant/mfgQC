"""Attribute acceptance-sampling tests, pinned to scipy-computed oracles.

Oracles (computed with scipy, see mfgqc/sampling.py docstring):
- Binomial OC n=134,c=3: Pa(0.01)=0.9537, Pa(0.05)=0.0931.
- Risk points n=134,c=3: AQL=1.03%, LTPD=4.92%, indifference=2.73%.
- N=500,n=134,c=3,p=0.02: binomial Pa=0.719, hypergeometric Pa=0.734.
- z14_plan(1200, 0.65, II, normal) -> code K, n=125, Ac=2, Re=3.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

from mfgqc import sampling
from mfgqc.sampling import (
    AOQResult,
    LotDisposition,
    OCCurveResult,
    SamplingPlan,
    find_plan,
    sampling_plan,
    z14_plan,
)
from mfgqc.sampling import _pa  # internal Pa for direct oracle checks


# --------------------------------------------------------------------------- #
# Pa oracle values
# --------------------------------------------------------------------------- #
def test_binomial_oc_oracle():
    assert _pa(0.01, 134, 3, "binomial", None) == pytest.approx(0.9537, abs=1e-3)
    assert _pa(0.05, 134, 3, "binomial", None) == pytest.approx(0.0931, abs=1e-3)


def test_finite_lot_oracle():
    assert _pa(0.02, 134, 3, "binomial", 500) == pytest.approx(0.719, abs=1e-3)
    assert _pa(0.02, 134, 3, "hypergeometric", 500) == pytest.approx(0.734, abs=1e-3)


def test_poisson_pa_matches_definition():
    from scipy import stats
    expected = float(stats.poisson.cdf(3, 134 * 0.02))
    assert _pa(0.02, 134, 3, "poisson", None) == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------- #
# Derived risk points
# --------------------------------------------------------------------------- #
def test_risk_points_oracle():
    plan = sampling_plan(134, 3)
    assert plan.aql == pytest.approx(0.0103, abs=0.0005)      # 1.03%
    assert plan.ltpd == pytest.approx(0.0492, abs=0.0005)     # 4.92%
    assert plan.indifference_point == pytest.approx(0.0273, abs=0.0005)  # 2.73%
    # producer risk alpha = 1 - Pa(AQL) ~= 0.05; consumer risk beta = Pa(LTPD) ~= 0.10
    assert plan.alpha == pytest.approx(0.05, abs=1e-6)
    assert plan.beta == pytest.approx(0.10, abs=1e-6)


# --------------------------------------------------------------------------- #
# Model auto-selection guardrail
# --------------------------------------------------------------------------- #
def test_model_default_is_binomial():
    plan = sampling_plan(50, 1)
    assert plan.model == "binomial"


def test_model_binomial_when_fraction_small():
    # n/N = 134/5000 = 0.027 <= 0.1 -> binomial
    plan = sampling_plan(134, 3, lot_size=5000)
    assert plan.model == "binomial"


def test_model_hypergeometric_when_fraction_large():
    # n/N = 134/500 = 0.268 > 0.1 -> hypergeometric, surfaced with a reason
    plan = sampling_plan(134, 3, lot_size=500)
    assert plan.model == "hypergeometric"
    reasons = [a.recommendation for a in plan.assumptions if a.name == "model_choice"]
    assert reasons and "hypergeometric" in reasons[0]


def test_poisson_only_when_requested():
    plan = sampling_plan(134, 3, model="poisson")
    assert plan.model == "poisson"


def test_binomial_approximation_flag_when_forced():
    # Force binomial on a finite lot with a large fraction -> guardrail flags it.
    plan = sampling_plan(134, 3, lot_size=500, model="binomial")
    flagged = [a for a in plan.assumptions if a.name == "binomial_approximation"]
    assert flagged and flagged[0].passed is False


# --------------------------------------------------------------------------- #
# find_plan (inverse problem)
# --------------------------------------------------------------------------- #
def test_find_plan_meets_both_risk_points():
    plan = find_plan(0.01, 0.05, alpha=0.05, beta=0.10)
    assert isinstance(plan, SamplingPlan)
    # Achieved OC must protect both producer and consumer.
    assert _pa(0.01, plan.n, plan.c, plan.model, None) >= 1 - 0.05
    assert _pa(0.05, plan.n, plan.c, plan.model, None) <= 0.10
    assert plan.requested_aql == pytest.approx(0.01)
    assert plan.requested_ltpd == pytest.approx(0.05)


def test_find_plan_is_minimal_n():
    plan = find_plan(0.01, 0.05)
    # No smaller n with this c can satisfy both constraints.
    smaller_ok = (
        _pa(0.01, plan.n - 1, plan.c, "binomial", None) >= 0.95
        and _pa(0.05, plan.n - 1, plan.c, "binomial", None) <= 0.10
    )
    assert not smaller_ok


# --------------------------------------------------------------------------- #
# Z1.4
# --------------------------------------------------------------------------- #
def test_z14_oracle():
    # Real ANSI/ASQ Z1.4 (general level II): 1201-3200 -> code K (n=125).
    # (Lot exactly 1200 falls in code J per the published lot-size ranges.)
    plan = z14_plan(lot_size=1300, aql=0.65, level="II", severity="normal")
    assert plan.code_letter == "K"
    assert plan.n == 125
    assert plan.c == 2   # Ac
    assert plan.re == 3  # Re


def test_z14_lot_size_boundary():
    # Published lot-size ranges: 501-1200 -> J (n=80); 1201-3200 -> K (n=125).
    assert z14_plan(1200, 0.65).code_letter == "J"
    assert z14_plan(1200, 0.65).n == 80
    assert z14_plan(1201, 0.65).code_letter == "K"
    assert z14_plan(1201, 0.65).n == 125


def test_z14_other_aqls_at_code_k():
    # Published code-K (n=125) acceptance numbers along the master-table staircase.
    assert z14_plan(1300, 1.0).c == 3
    assert z14_plan(1300, 1.5).c == 5
    assert z14_plan(1300, 2.5).c == 7
    assert z14_plan(1300, 4.0).c == 10
    assert z14_plan(1300, 6.5).c == 14
    assert z14_plan(1300, 10.0).c == 21


def test_z14_severity_not_implemented():
    with pytest.raises(NotImplementedError):
        z14_plan(1200, 0.65, severity="tightened")


def test_z14_level_not_implemented():
    with pytest.raises(NotImplementedError):
        z14_plan(1200, 0.65, level="I")


# --------------------------------------------------------------------------- #
# LotDisposition
# --------------------------------------------------------------------------- #
def test_inspect_accept():
    plan = sampling_plan(134, 3)
    d = plan.inspect(2)
    assert d.decision == "accept"
    assert d.defectives_found == 2
    assert d.c == 3


def test_inspect_boundary_accept():
    plan = sampling_plan(134, 3)
    assert plan.inspect(3).decision == "accept"  # found == c accepts


def test_inspect_reject():
    plan = sampling_plan(134, 3)
    d = plan.inspect(4)
    assert d.decision == "reject"


# --------------------------------------------------------------------------- #
# AOQ
# --------------------------------------------------------------------------- #
def test_aoql_is_max_at_interior():
    plan = sampling_plan(134, 3, lot_size=100000)
    aoq = plan.aoq_curve()
    assert aoq.aoql == pytest.approx(float(np.max(aoq.aoq)))
    # Interior maximum: not at either endpoint of the grid.
    assert aoq.p_grid[0] < aoq.aoql_at_p < aoq.p_grid[-1]
    assert aoq.aoql > 0


def test_aoq_no_lot_size_uses_p_times_pa():
    plan = sampling_plan(134, 3)
    aoq = plan.aoq_curve()
    # AOQ(p) = p * Pa(p) when no finite lot.
    p = aoq.p_grid
    expected = p * np.array([_pa(float(x), 134, 3, "binomial", None) for x in p])
    assert np.allclose(aoq.aoq, expected)


# --------------------------------------------------------------------------- #
# Result types: curves, summaries, views
# --------------------------------------------------------------------------- #
def test_oc_curve_arrays_and_type():
    oc = sampling_plan(134, 3).oc_curve()
    assert isinstance(oc, OCCurveResult)
    assert isinstance(oc.p_grid, np.ndarray)
    assert isinstance(oc.pa, np.ndarray)
    assert oc.p_grid.shape == oc.pa.shape
    # Pa is monotone decreasing in p.
    assert np.all(np.diff(oc.pa) <= 1e-9)


@pytest.mark.parametrize("make", [
    lambda: sampling_plan(134, 3),
    lambda: sampling_plan(134, 3).oc_curve(),
    lambda: sampling_plan(134, 3).aoq_curve(),
    lambda: sampling_plan(134, 3).inspect(2),
    lambda: z14_plan(1200, 0.65),
])
def test_summary_is_flat_dict(make):
    result = make()
    summary = result.summary()
    assert isinstance(summary, dict)
    # Flat: no nested dicts / lists as values.
    for k, v in summary.items():
        assert isinstance(k, str)
        assert not isinstance(v, (dict, list, tuple, np.ndarray))


def test_lot_disposition_summary_shape():
    d = sampling_plan(134, 3).inspect(2)
    s = d.summary()
    assert s["decision"] == "accept"
    assert s["found"] == 2
    assert s["limit"] == 3
    assert s["n"] == 134


@pytest.mark.parametrize("make", [
    lambda: sampling_plan(134, 3),
    lambda: sampling_plan(134, 3).oc_curve(),
    lambda: sampling_plan(134, 3, lot_size=100000).aoq_curve(),
    lambda: sampling_plan(134, 3).inspect(5),
    lambda: z14_plan(1200, 0.65),
])
def test_view_returns_figure(make):
    fig = make().view()
    assert isinstance(fig, Figure)


def test_report_text_mentions_stated_assumptions():
    text = sampling_plan(134, 3).report()
    assert "random sampling" in text
    assert "homogeneity" in text


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_invalid_inputs():
    with pytest.raises(ValueError):
        sampling_plan(0, 1)
    with pytest.raises(ValueError):
        sampling_plan(10, -1)
    with pytest.raises(ValueError):
        sampling_plan(100, 1, lot_size=50)  # n > N
    with pytest.raises(ValueError):
        find_plan(0.05, 0.01)  # aql >= ltpd
