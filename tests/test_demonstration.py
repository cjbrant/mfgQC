"""Slice 8: bearing life (ISO 281) + demonstration test sizing. ISO 281 is exact
against the standard formula; the success-run plan is the standard result."""
import matplotlib; matplotlib.use("Agg")
import math, pytest, mfgqc


# --- bearing life (slice 6) ---
def test_bearing_l10_iso281_formula():
    b = mfgqc.reliability.bearing_life(C=26000, P=4000, rpm=1800, kind="ball")
    assert abs(b.L10_revs_million - (26000 / 4000) ** 3) < 1e-9
    assert abs(b.L10_hours - (1e6 / (60 * 1800)) * (26000 / 4000) ** 3) < 1e-6


def test_bearing_roller_exponent():
    b = mfgqc.reliability.bearing_life(C=50000, P=5000, rpm=500, kind="roller")
    assert abs(b.exponent - 10 / 3) < 1e-12
    assert b.rated[0.90] == pytest.approx(b.L10_hours)        # L10 is the 90% basis


def test_bearing_refuses_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        mfgqc.reliability.bearing_life(C=26000, P=0, rpm=1800)


# --- demonstration test (slice 8) ---
def test_zero_failure_success_run_n():
    d = mfgqc.reliability.demonstration_test(reliability=0.95, confidence=0.90)
    assert d.n == math.ceil(math.log(0.10) / math.log(0.95))   # = 45
    assert d.failures == 0


def test_solve_for_reliability_given_n():
    d = mfgqc.reliability.demonstration_test(confidence=0.90, n=22)
    assert abs(d.reliability - (1 - 0.90) ** (1 / 22)) < 1e-9


def test_general_plan_with_failures():
    d = mfgqc.reliability.demonstration_test(reliability=0.90, confidence=0.90, failures=2)
    # more allowed failures -> need a larger sample than zero-failure
    z = mfgqc.reliability.demonstration_test(reliability=0.90, confidence=0.90)
    assert d.n > z.n


def test_exactly_one_unknown_required():
    with pytest.raises(ValueError, match="exactly one"):
        mfgqc.reliability.demonstration_test(reliability=0.95, confidence=0.90, n=45)
