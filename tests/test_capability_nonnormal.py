"""Non-normal capability oracles (Test Oracle Part 2 - self-verifying).

The percentile method (Clements / ISO 22514) has a closed-form definition, so
data drawn from a distribution with KNOWN percentiles has an exact ground-truth
capability. That is a stronger oracle than a textbook transcription: exact,
controlled, reproducible.

- Oracle C: percentile method on Normal(10,1), spec [6,14] -> Cp = Cpk ~ 1.333.
- Oracle A: lognormal(logmu=1, logsigma=0.5), spec [0,20] -> Cp~1.728, Cpk~1.287.
- Oracle B: exponential(1), USL=5 only -> Ppu ~ 0.728.
- Oracle D: non-normal methods agree in a band; normal method is wrong + warns.

Box-Cox / Johnson get convergence tests only (finite-sample, implementation-
sensitive), never hardcoded pins.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import mfgqc
from mfgqc.capability import _percentile_indices
from mfgqc.data import _Limits

LOGNORMAL_CPK_TRUTH = 1.2872
LOGNORMAL_CP_TRUTH = 1.7277


def _qc(x, lo=None, hi=None):
    return mfgqc.load(
        pd.DataFrame({"y": np.asarray(x, dtype=float)}),
        measure="y",
    ).spec(lower=lo, upper=hi)


# --------------------------------------------------------------------------- #
# Exact closed-form formula (no fitting) - the strongest pin
# --------------------------------------------------------------------------- #
def test_percentile_formula_exact_lognormal():
    """_percentile_indices reproduces Oracle A's closed-form values exactly."""
    cp, cpu, cpl, cpk = _percentile_indices(0.60654, 2.71828, 12.18235,
                                            _Limits(lower=0.0, upper=20.0))
    assert cp == pytest.approx(1.7277, abs=1e-3)
    assert cpu == pytest.approx(1.8260, abs=1e-3)
    assert cpl == pytest.approx(1.2872, abs=1e-3)
    assert cpk == pytest.approx(1.2872, abs=1e-3)


def test_percentile_formula_reduces_to_classic_for_normal():
    """+/-3 sigma percentiles of a normal reduce to the classic 6-sigma index."""
    from scipy.stats import norm
    lo, m, hi = norm.ppf([0.00135, 0.5, 0.99865], loc=10, scale=1)
    cp, _, _, cpk = _percentile_indices(lo, m, hi, _Limits(lower=6.0, upper=14.0))
    assert cp == pytest.approx(8.0 / 6.0, abs=1e-3)   # (14-6)/(6 sigma) = 1.3333
    assert cpk == pytest.approx(8.0 / 6.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# Convergence oracles on large fixed-seed samples
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def _rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def lognormal_results():
    """All capability methods on one large lognormal sample (computed once)."""
    rng = np.random.default_rng(0)
    x = rng.lognormal(1.0, 0.5, 100_000)
    qc = _qc(x, 0.0, 20.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {m: qc.capability(method=m)
                for m in ("normal", "clements", "percentile", "johnson", "boxcox")}


def test_percentile_reduces_to_classic_on_normal():
    """Oracle C: percentile method on Normal(10,1), spec [6,14] -> Cp = Cpk ~ 1.333."""
    rng = np.random.default_rng(1)
    x = rng.normal(10.0, 1.0, 100_000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cap = _qc(x, 6.0, 14.0).capability(method="clements")
    assert cap.cp == pytest.approx(1.3333, abs=0.01)
    assert cap.cpk == pytest.approx(1.3333, abs=0.01)


def test_percentile_lognormal_converges(lognormal_results):
    """Oracle A: percentile method converges to the exact lognormal truth."""
    cap = lognormal_results["clements"]
    assert cap.cp == pytest.approx(LOGNORMAL_CP_TRUTH, abs=0.03)
    assert cap.cpk == pytest.approx(LOGNORMAL_CPK_TRUTH, abs=0.03)
    assert cap.cpl == pytest.approx(1.2872, abs=0.03)
    # 'percentile' is an alias for the same method
    assert lognormal_results["percentile"].cpk == pytest.approx(cap.cpk)


def test_percentile_exponential_one_sided():
    """Oracle B: percentile method on exponential(1), USL=5 only -> Ppu ~ 0.728."""
    rng = np.random.default_rng(2)
    x = rng.exponential(1.0, 100_000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cap = _qc(x, None, 5.0).capability(method="clements")
    assert cap.ppu == pytest.approx(0.728, abs=0.03)
    assert cap.cp is None   # one-sided
    assert cap.cpl is None


def test_nonnormal_methods_agree_band(lognormal_results):
    """Oracle D: non-normal methods land in a tight band around the true Cpk."""
    for name in ("clements", "johnson", "boxcox"):
        assert lognormal_results[name].cpk == pytest.approx(LOGNORMAL_CPK_TRUTH, abs=0.1)


def test_normal_method_on_skewed_fails_and_warns(lognormal_results):
    """Oracle D guardrail: normal method is materially wrong AND flags normality."""
    normal = lognormal_results["normal"]
    assert normal.cpk < 0.8  # wrong (~0.63), far from the true 1.287
    norm_check = next(a for a in normal.assumptions if a.name == "normality")
    assert norm_check.passed is False
    rec = (norm_check.recommendation or "").lower()
    assert "clements" in rec or "johnson" in rec


# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #
def test_method_recorded_in_history():
    rng = np.random.default_rng(5)
    qc = _qc(rng.lognormal(1.0, 0.5, 1000), 0.0, 20.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cap = qc.capability(method="clements")
    step = next(s for s in cap.history if s.operation == "capability")
    assert step.params["method"] == "clements"
    assert "lognormal" in cap.sigma_used  # auto-selected distribution recorded


def test_invalid_method_errors():
    qc = _qc(np.arange(1.0, 50.0), 0.0, 60.0)
    with pytest.raises(ValueError, match="method must be one of"):
        qc.capability(method="bogus")
