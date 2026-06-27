"""Slice 7: availability. Inherent identity is self-verifying."""
import matplotlib; matplotlib.use("Agg")
import pytest, mfgqc


def test_inherent_availability_identity():
    a = mfgqc.availability(mtbf=500, mttr=10, kind="inherent")
    assert abs(a.availability - 500 / 510) < 1e-12


def test_operational_adds_logistics_delay():
    a = mfgqc.availability(mtbf=500, mttr=10, kind="operational", logistics_delay=40)
    assert abs(a.availability - 500 / 550) < 1e-12
    assert a.availability < mfgqc.availability(mtbf=500, mttr=10).availability


def test_achieved_includes_preventive_maintenance():
    a = mfgqc.availability(mtbf=500, mttr=10, kind="achieved", pm_time=2, pm_freq=0.001)
    assert 0 < a.availability < 1
    assert "pm_freq" in a.summary()


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        mfgqc.availability(mtbf=0, mttr=10)
    with pytest.raises(ValueError, match="kind"):
        mfgqc.availability(mtbf=500, mttr=10, kind="weird")
