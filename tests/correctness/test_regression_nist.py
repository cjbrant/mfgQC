"""Correctness: linear regression vs the NIST StRD certified "Norris" dataset.

Source: NIST Statistical Reference Datasets, linear least squares, "Norris"
(https://www.itl.nist.gov/div898/strd/lls/data/Norris.shtml). The certified
estimates are stated to 15 significant digits and are the reference values the
StRD project distributes for validating regression software. mfgQC was not built
against this dataset.
"""

from __future__ import annotations

import pytest

import mfgqc
from .conftest import NORRIS_CERT


def test_norris_coefficients(norris_df):
    """NIST StRD Norris: certified slope/intercept (15-digit reference)."""
    reg = mfgqc.load(norris_df, measure="y").regress(on="x")
    assert reg.coef["x"] == pytest.approx(NORRIS_CERT["slope"], rel=1e-9)
    assert reg.coef["intercept"] == pytest.approx(NORRIS_CERT["intercept"], rel=1e-7)


def test_norris_standard_errors(norris_df):
    """NIST StRD Norris: certified standard deviations of the estimates."""
    reg = mfgqc.load(norris_df, measure="y").regress(on="x")
    assert reg.se["x"] == pytest.approx(NORRIS_CERT["slope_sd"], rel=1e-7)
    assert reg.se["intercept"] == pytest.approx(NORRIS_CERT["intercept_sd"], rel=1e-7)


def test_norris_fit_statistics(norris_df):
    """NIST StRD Norris: certified residual sd, R-squared, residual df."""
    reg = mfgqc.load(norris_df, measure="y").regress(on="x")
    assert reg.resid_std_err == pytest.approx(NORRIS_CERT["resid_sd"], rel=1e-8)
    assert reg.r_squared == pytest.approx(NORRIS_CERT["r_squared"], rel=1e-10)
    assert reg.df_resid == NORRIS_CERT["df_resid"]
