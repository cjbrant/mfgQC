"""T2.5 - the pinned worked example (this repo's own oracle).

fixtures_worked_example.py regenerates every number independently (numpy/scipy
only, no mfgqc import) and self-asserts, so matching it validates the bayes
capability engine. The normative sampling call order (chisquare then normal)
makes the MC parts of (b) bit-reproducible against the fixture.
"""
from __future__ import annotations

import math

from mfgqc.bayes.capability import capability_from_values
from mfgqc.bayes.priors import NormalPrior

from .fixtures_worked_example import DRAWS, LSL, PINNED, TARGET, USL, worked_example_data


def _fit(prior=None):
    return capability_from_values(
        worked_example_data(), lower=LSL, upper=USL, target=TARGET,
        prior=prior, seed=7, draws=DRAWS)


def test_t2_5a_noninformative_closed_form():
    """(a) closed-form summaries: n, ybar, s, mu 95% interval, sd 95% interval,
    Ppk point estimate."""
    r = _fit()
    assert r.n == PINNED["n"]
    assert round(r.mean, 4) == PINNED["ybar"]
    assert round(r.s, 5) == PINNED["s"]

    mu_lo, mu_hi = r.interval("mu", 0.95)
    assert [round(mu_lo, 4), round(mu_hi, 4)] == PINNED["mu95"]

    sd_lo, sd_hi = r.interval("sd", 0.95)
    assert [round(sd_lo, 5), round(sd_hi, 5)] == PINNED["sd95"]

    assert round(r.ppk, 3) == PINNED["ppk_point"]


def test_t2_5b_mc_pushthrough_bit_reproduces_fixture():
    """(b) seeded MC push-through (seed 7, 100000 draws): P(Ppk>=1.33) with MCSE,
    Ppk median + 95%, ppm median + 95%. Bit-reproducible against the fixture."""
    r = _fit()
    p, mcse = r.prob("ppk", 1.33)
    assert round(p, 3) == PINNED["p_ppk_133"]
    assert round(mcse, 3) == 0.001  # spec T2.5(b): 0.074 +/- 0.001

    assert [round(v, 3) for v in r.quantiles("ppk", [0.5, 0.025, 0.975])] == PINNED["ppk_med_95"]
    assert [round(v) for v in r.quantiles("ppm", [0.5, 0.025, 0.975])] == PINNED["ppm_med_95"]


def test_t2_5c_informative_fit_closed_form_and_prob():
    """(c) informative prior N-Inv-chi2(25.005, k0=20, s0=0.012, nu0=20): prior
    weight, posterior hyperparameters exact; P(Ppk>=1.33) within 3*MCSE (the
    fixture's own value is entangled with (b)'s RNG stream, so not bit-identical)."""
    prior = NormalPrior(mu0=25.005, k0=20, nu0=20, s20=0.012 ** 2)
    r = _fit(prior=prior)

    pw = next(a for a in r.assumptions if a.name == "prior_weight")
    assert round(pw.statistic, 4) == PINNED["prior_weight"]
    assert round(r.mun, 4) == PINNED["mun"]
    assert r.kn == PINNED["kn"]
    assert r.nun == PINNED["nun"]
    assert round(math.sqrt(r.sn2), 5) == PINNED["sn"]

    p, mcse = r.prob("ppk", 1.33)
    assert abs(p - PINNED["p_ppk_133_inf"]) <= 3.0 * mcse
