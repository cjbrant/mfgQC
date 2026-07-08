"""Guardband: cost-optimal acceptance limits (spec Algorithm K; BDA3 ch. 9).
Gate: T1.17 (symmetry / gauge->0 limits), T6.7 (non-unimodal fallback, gauge dominance).
"""
from __future__ import annotations

import numpy as np
import pytest

from mfgqc.bayes.capability import capability_from_values
from mfgqc.bayes.decisions import GuardbandResult, guardband, _minimize_two_sided


def _capable_result(seed=1, center=0.0, sd=0.2, lower=-1.0, upper=1.0, demean=False):
    y = np.random.default_rng(seed).normal(center, sd, 200)
    if demean:
        y = y - y.mean() + center  # process mean exactly at ``center``
    return capability_from_values(y, lower=lower, upper=upper, seed=1, draws=20000)


# ---- T1.17a symmetric costs + symmetric spec -> symmetric limits --------- #
def test_t1_17a_symmetric_costs_give_symmetric_limits():
    """Symmetric spec + symmetric costs + a process centered on the spec center
    give acceptance limits symmetric about that center."""
    res = _capable_result(center=0.0, sd=0.2, lower=-1.0, upper=1.0, demean=True)
    gb = guardband(res, sigma_gauge=0.1, c_scrap=1.0, c_escape=1.0, seed=5, grid=2001)
    assert isinstance(gb, GuardbandResult)
    dx = (gb.grid["x_hi"] - gb.grid["x_lo"]) / (gb.grid["points"] - 1)
    # limits symmetric about the spec center (0)
    assert abs(gb.a_lo + gb.a_hi) <= 4.0 * dx


# ---- T1.17b sigma_gauge -> 0 -> limits approach the spec ------------------ #
def test_t1_17b_small_gauge_limits_approach_spec():
    res = _capable_result(center=0.0, sd=0.2, lower=-1.0, upper=1.0)
    process_sd = float(np.std(res._draws["mu"]) + np.mean(res._draws["sigma"]))
    gb = guardband(res, sigma_gauge=1e-3 * process_sd, c_scrap=1.0, c_escape=1.0,
                   seed=5, grid=4001)
    dx = (gb.grid["x_hi"] - gb.grid["x_lo"]) / (gb.grid["points"] - 1)
    assert abs(gb.a_lo - (-1.0)) <= 5.0 * dx
    assert abs(gb.a_hi - 1.0) <= 5.0 * dx
    assert gb.scrap_pct <= 0.5 and gb.escape_ppm <= 500.0


def test_guardband_optimum_beats_naive():
    """The optimized limits never cost more than the naive (spec) limits, since the
    naive limits are inside the search domain."""
    res = _capable_result(center=0.05, sd=0.28, lower=-1.0, upper=1.0)
    gb = guardband(res, sigma_gauge=0.15, c_scrap=1.0, c_escape=20.0, seed=2, grid=2001)
    assert gb.expected_cost <= gb.naive_expected_cost + 1e-9


# ---- T6.7b gauge dominates process -> raise ------------------------------ #
def test_t6_7b_gauge_dominates_raises():
    res = _capable_result(seed=3, center=25.0, sd=0.01, lower=24.9, upper=25.1)
    pp_sd = float(np.sqrt(np.mean(res._draws["sigma"] ** 2) + np.var(res._draws["mu"])))
    with pytest.raises(ValueError, match="gauge dominates"):
        guardband(res, sigma_gauge=1.5 * pp_sd, c_scrap=1.0, c_escape=1.0, seed=1)
    # just below the threshold is allowed
    gb = guardband(res, sigma_gauge=0.5 * pp_sd, c_scrap=1.0, c_escape=1.0, seed=1, grid=2001)
    assert gb.pp_sd == pytest.approx(pp_sd, rel=1e-9)


# ---- T6.7a non-unimodal cost surface -> warn + grid fallback ------------- #
def test_t6_7a_non_unimodal_scan_falls_back_to_grid_minimum():
    """T6.7a: the coordinate optimizer detects a non-unimodal expected-cost slice
    and falls back to the 2-D grid global minimum.

    Note: for a single spec interval with a Gaussian gauge the true expected-cost
    surface is PROVABLY unimodal in each coordinate (scrap monotone decreasing,
    escape monotone increasing), verified by a broad numerical sweep - no physical
    capability posterior triggers this path. So the guard is exercised directly on
    a synthetic double-well cost surface, which is exactly what the fallback
    defends against."""
    # a_lo has two wells (at -1.5 and -0.5), tilted so -1.5 is the global minimum
    def ec(a_lo, a_hi):
        return (a_lo + 1.5) ** 2 * (a_lo + 0.5) ** 2 + 0.05 * a_lo + (a_hi - 1.0) ** 2

    a_lo, a_hi, non_unimodal, fallback_used = _minimize_two_sided(
        ec, lo_domain=(-2.0, 0.0), hi_domain=(0.0, 2.0), tol=1e-4)
    assert non_unimodal is True and fallback_used is True
    assert abs(a_lo - (-1.5)) <= 0.05  # global well, not the local one at -0.5
    assert abs(a_hi - 1.0) <= 0.05


def test_quadrature_grid_covers_predictive_density_off_center():
    """Review finding 2 (major): the quadrature grid must cover the predictive
    density even when the process is centered far outside the spec, so the
    integrals are not silently evaluated on ~zero density. Grid bounds must union
    the spec-anchored and predictive-mean-anchored ranges."""
    res = _capable_result(seed=1, center=5.0, sd=0.2, lower=-1.0, upper=1.0)
    gb = guardband(res, sigma_gauge=0.05, c_scrap=1.0, c_escape=20.0, seed=2, grid=4001)
    pred_mean = float(np.mean(res._draws["mu"]))
    assert gb.grid["x_lo"] <= pred_mean - 4.0 * gb.pp_sd
    assert gb.grid["x_hi"] >= pred_mean + 4.0 * gb.pp_sd  # density at x~5 is inside the grid


def test_t6_7a_unimodal_surface_no_fallback():
    """A single-well surface is optimized by golden-section with no fallback."""
    def ec(a_lo, a_hi):
        return (a_lo + 1.0) ** 2 + (a_hi - 1.0) ** 2

    a_lo, a_hi, non_unimodal, fallback_used = _minimize_two_sided(
        ec, lo_domain=(-2.0, 0.0), hi_domain=(0.0, 2.0), tol=1e-4)
    assert non_unimodal is False and fallback_used is False
    assert abs(a_lo - (-1.0)) <= 1e-3 and abs(a_hi - 1.0) <= 1e-3
