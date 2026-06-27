"""Slice 3: Kaplan-Meier R(t) with Greenwood bounds. Pinned to the manual
product-limit computation and a known worked example; survival::survfit is the
held-back secondary key."""
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, pytest, mfgqc


def _qc(t, e):
    return mfgqc.load(pd.DataFrame({"h": t, "failed": e}), measure="h").roles(time="h", event="failed")


def test_km_matches_manual_product_limit():
    # 6 units: failures at 1,3,5; suspensions at 2,4,6
    t = [1, 2, 3, 4, 5, 6]; e = [1, 0, 1, 0, 1, 0]
    res = _qc(t, e).life_table()
    # manual: S(1)=5/6; S(3)=5/6*3/4=0.625; S(5)=0.625*1/2=0.3125
    s = dict(zip(res.times, res.survival))
    assert abs(s[1] - 5/6) < 1e-9
    assert abs(s[3] - 5/6 * 3/4) < 1e-9
    assert abs(s[5] - 5/6 * 3/4 * 1/2) < 1e-9


def test_km_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    t = rng.exponential(50, 60); e = (rng.random(60) > 0.25).astype(int)
    res = _qc(t, e).life_table()
    assert np.all(np.diff(res.survival) <= 1e-12)             # non-increasing
    assert np.all((res.lower >= -1e-9) & (res.upper <= 1 + 1e-9))
    assert np.all(res.lower <= res.survival + 1e-9)


def test_km_median_life():
    rng = np.random.default_rng(1)
    t = rng.weibull(2.0, 100) * 100; e = np.ones(100)
    res = _qc(t, e).life_table()
    assert np.isfinite(res.median_life)
    assert abs(res.R(res.median_life) - 0.5) < 0.1


def test_greenwood_bounds_widen_with_fewer_at_risk():
    t = [10, 20, 30, 40, 50]; e = [1, 1, 1, 1, 1]
    res = _qc(t, e).life_table()
    w = res.upper - res.lower
    # bounds widen through the interior (the final S=0 step has zero width)
    assert w[-2] >= w[1]


def test_view_renders():
    res = _qc([1, 2, 3, 4, 5], [1, 0, 1, 1, 0]).life_table()
    assert res.view() is not None
