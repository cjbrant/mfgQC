"""Chart palette: mfgQC figures use the phosphor (dark) theme; set_theme switches it."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

import mfgqc
from mfgqc import palette


def _cap():
    df = pd.DataFrame({"x": np.random.default_rng(0).normal(100, 2, 60)})
    return mfgqc.load(df, measure="x").spec(lower=92, upper=108).capability()


def test_default_theme_is_phosphor_and_dark():
    assert palette.active().name == "phosphor"
    fig = _cap().view()
    r, g, b = fig.get_facecolor()[:3]
    assert max(r, g, b) < 0.12          # near-black CRT canvas


def test_result_chart_does_not_pollute_global_rcparams():
    import matplotlib as mpl
    before = mpl.rcParams["figure.facecolor"]
    _cap().view()                        # styled via rc_context, not globally
    assert mpl.rcParams["figure.facecolor"] == before


def test_overview_chart_is_themed():
    df = pd.DataFrame({"x": np.random.default_rng(1).normal(10, 1, 60)})
    fig = mfgqc.overview(df).view()
    assert max(fig.get_facecolor()[:3]) < 0.12


def test_set_theme_switches_and_light_is_white():
    try:
        mfgqc.set_theme("amber")
        assert palette.active().name == "amber"
        mfgqc.set_theme("light")
        fig = _cap().view()
        assert min(fig.get_facecolor()[:3]) > 0.9      # light theme -> white canvas
    finally:
        mfgqc.set_theme("phosphor")


def test_set_theme_rejects_unknown():
    with pytest.raises(ValueError, match="theme must be one of"):
        mfgqc.set_theme("neon")
