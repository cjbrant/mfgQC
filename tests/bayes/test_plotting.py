"""Smoke tests for view() on every bayes result type.

Each bayes QCResult must render both standalone (returns a Figure) and into a
caller-supplied Axes (returns that Axes), and must honor save= when standalone.
The charts are derived only from fields the frozen result already stores.
"""
from __future__ import annotations

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mfgqc.bayes import (
    Censoring,
    NormalPrior,
    assurance,
    capability_censored,
    capability_from_values,
    compare,
    fit_normal,
    guardband,
    monitor,
    phase1,
    pooled_capability,
    predictive_check,
    proportion_capability,
    rate_capability,
    shortrun,
)

_RNG = np.random.default_rng(0)
_Y = _RNG.normal(25.0, 0.5, 60)
_Y2 = _RNG.normal(25.3, 0.4, 60)


def _normal():
    return fit_normal(_Y, NormalPrior(25.0, 20, 20, 0.25), seed=1, draws=1000)


def _capability():
    return capability_from_values(_Y, lower=23.0, upper=27.0, target=25.0,
                                  seed=1, draws=4000)


def _proportion():
    return proportion_capability(n_fail=3, n_trials=200, max_proportion=0.05)


def _rate():
    counts = np.array([2, 1, 0, 3, 1, 2])
    return rate_capability(counts, max_rate=2.5)


def _censored():
    return capability_censored(_Y, lower=23.0, upper=27.0,
                               censoring=Censoring(upper=26.0), seed=1, draws=4000)


def _comparison():
    a = capability_from_values(_Y, lower=23.0, upper=27.0, seed=1, draws=4000)
    b = capability_from_values(_Y2, lower=23.0, upper=27.0, seed=2, draws=4000)
    return compare(a, b, seed=3, draws=4000, labels=("line A", "line B"))


def _assurance():
    r = capability_from_values(_Y, lower=23.0, upper=27.0, seed=1, draws=4000)
    return assurance(r, seed=5, n_grid=(20, 50, 100), sims=100, inner_draws=200)


def _guardband():
    r = capability_from_values(_Y, lower=23.0, upper=27.0, seed=1, draws=4000)
    return guardband(r, sigma_gauge=0.15, c_scrap=1.0, c_escape=10.0, seed=6, ndraws=1000)


def _pooled():
    groups = [_RNG.normal(25.0, 0.5, 12) for _ in range(4)]
    return pooled_capability(groups, lower=23.0, upper=27.0, target=1.33,
                             seed=7, draws=4000)


def _shortrun():
    subs = [_RNG.normal(25.0, 0.5, 5) for _ in range(6)]
    return shortrun(subs, target=25.0, lower=23.0, upper=27.0, allow_vague=True)


def _monitor():
    ref = phase1(_Y)
    subs = [_RNG.normal(25.0, 0.5, 5) for _ in range(5)]
    return monitor(ref, subs, seed=8, R=1000)


def _predictive_check():
    return predictive_check(_Y, statistic="min", seed=9, R=1000)


_BUILDERS = [
    ("normal", _normal),
    ("capability", _capability),
    ("proportion", _proportion),
    ("rate", _rate),
    ("censored", _censored),
    ("comparison", _comparison),
    ("assurance", _assurance),
    ("guardband", _guardband),
    ("pooled", _pooled),
    ("shortrun", _shortrun),
    ("monitor", _monitor),
    ("predictive_check", _predictive_check),
]


@pytest.mark.parametrize("name,builder", _BUILDERS, ids=[n for n, _ in _BUILDERS])
def test_view_standalone_returns_figure(name, builder):
    fig = builder().view()
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.parametrize("name,builder", _BUILDERS, ids=[n for n, _ in _BUILDERS])
def test_view_into_axes_returns_axes(name, builder):
    fig, ax = plt.subplots()
    out = builder().view(ax=ax)
    assert isinstance(out, Axes)
    plt.close(fig)


@pytest.mark.parametrize("name,builder", _BUILDERS, ids=[n for n, _ in _BUILDERS])
def test_view_save_writes_file(name, builder, tmp_path):
    path = tmp_path / f"{name}.png"
    fig = builder().view(save=str(path))
    assert isinstance(fig, Figure)
    assert path.exists() and path.stat().st_size > 0
    plt.close(fig)
