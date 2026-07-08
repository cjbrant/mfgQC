"""Bayesian attribute capability surfaces: proportion (Beta-Binomial) and rate
(Gamma-Poisson). Closed-form; oracles cross-checked against BDA3 sec 2.4 / 2.6.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mfgqc.bayes.attributes import (
    BayesProportionResult,
    BayesRateResult,
    proportion_capability,
    rate_capability,
)
from mfgqc.bayes.priors import BetaPrior, GammaPrior

from ._oracles import PLACENTA, PLACENTA_CI95, RATE_EXAMPLE


# ---- proportion surface --------------------------------------------------- #
def test_proportion_reproduces_placenta_with_uniform_prior():
    """Beta(1,1) prior, y=437/980 -> posterior mean 0.446, 95% CI [0.415,0.477]."""
    r = proportion_capability(n_fail=PLACENTA["y"], n_trials=PLACENTA["n"],
                              prior=BetaPrior(1.0, 1.0))
    assert isinstance(r, BayesProportionResult)
    assert abs(r.mean - PLACENTA["posterior_mean"]) <= 5e-4
    lo, hi = r.interval()
    assert (round(lo, 3), round(hi, 3)) == PLACENTA_CI95


def test_proportion_default_prior_is_jeffreys():
    """Default noninformative prior is Jeffreys Beta(0.5,0.5)."""
    r = proportion_capability(n_fail=437, n_trials=980)
    assert r.a_post == 0.5 + 437 and r.b_post == 0.5 + 980 - 437
    assert abs(r.mean - 0.446) <= 5e-3


def test_proportion_prob_within_spec_and_directions():
    r = proportion_capability(n_fail=5, n_trials=200, max_proportion=0.05)
    # P(p <= 0.05) should be high (p_hat = ~0.0275) and complement P(p >= 0.05) low
    assert r.prob(0.05, "<=") + r.prob(0.05, ">=") == pytest.approx(1.0, abs=1e-9)
    assert r.prob_within_spec() == pytest.approx(r.prob(0.05, "<="), abs=1e-12)
    assert r.prob_within_spec() > 0.9


def test_proportion_accepts_bool_zero_one_and_mapping():
    a = proportion_capability(data=[False, False, True, False])          # bool
    b = proportion_capability(data=[0, 0, 1, 0])                          # {0,1}
    c = proportion_capability(data=["ok", "ok", "fail", "ok"],
                              mapping={"ok": 0, "fail": 1})               # labels
    assert a.n_fail == b.n_fail == c.n_fail == 1
    assert a.n_trials == b.n_trials == c.n_trials == 4


def test_proportion_rejects_non_binary_without_mapping():
    with pytest.raises(ValueError):
        proportion_capability(data=[0, 1, 2, 0])


def test_proportion_predictive_pmf_is_a_distribution():
    r = proportion_capability(n_fail=5, n_trials=50)
    m = 10
    total = sum(r.predictive_pmf(k, m) for k in range(m + 1))
    assert total == pytest.approx(1.0, abs=1e-9)


# ---- rate surface --------------------------------------------------------- #
def test_rate_reproduces_asthma_with_gamma_prior():
    """Gamma(3,5) prior, y=3 over exposure 2.0 -> Gamma(6,7): mean 0.857,
    P(rate>1)=0.30 (BDA3 sec 2.6)."""
    r = rate_capability([3.0], exposures=[2.0], prior=GammaPrior(3.0, 5.0))
    assert isinstance(r, BayesRateResult)
    assert (r.a_post, r.b_post) == RATE_EXAMPLE["posterior"]
    assert abs(r.mean - RATE_EXAMPLE["posterior_mean"]) <= 5e-3
    assert abs(r.prob(1.0, ">=") - RATE_EXAMPLE["p_theta_gt_1"]) <= 5e-3


def test_rate_default_prior_is_jeffreys_and_unit_exposure():
    """Default Jeffreys Gamma(0.5,0); exposures default to one each."""
    r = rate_capability([2, 4, 3, 5])
    assert r.a_post == 0.5 + 14 and r.b_post == 0.0 + 4
    assert r.mean == pytest.approx(14.5 / 4.0, abs=1e-12)


def test_rate_prob_within_spec():
    r = rate_capability([1, 0, 2, 1, 0, 1], max_rate=2.0)
    assert r.prob_within_spec() == pytest.approx(r.prob(2.0, "<="), abs=1e-12)
    assert 0.0 < r.prob_within_spec() < 1.0


def test_rate_predictive_pmf_is_a_distribution():
    r = rate_capability([2, 4, 3, 5])
    total = sum(r.predictive_pmf(k, exposure=1.0) for k in range(200))
    assert total == pytest.approx(1.0, abs=1e-6)


# ---- T6.4 (G4 overdispersion) -------------------------------------------- #
def _check(result, name):
    return next(a for a in result.assumptions if a.name == name)


def test_t6_4_overdispersion_warns_for_nbinom_not_poisson():
    """G4: neg-binomial-generated counts fit as Poisson warn (overdispersed);
    genuine Poisson counts do not. Seeded, deterministic."""
    rng = np.random.default_rng(64)
    over = rate_capability(rng.negative_binomial(2, 0.25, size=60))
    d = _check(over, "dispersion")
    assert d.passed is False and d.recommendation is not None
    assert d.magnitude > 1.0

    ok = rate_capability(rng.poisson(6.0, size=60))
    d2 = _check(ok, "dispersion")
    assert d2.passed is True and d2.recommendation is None


# ---- provenance / immutability ------------------------------------------- #
def test_attribute_results_are_immutable_and_verifiable():
    r = proportion_capability(n_fail=5, n_trials=50)
    assert r.verify_provenance(r.provenance_digest()) is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.mean = 0.5  # type: ignore[misc]


def test_proportion_digest_changes_with_data():
    a = proportion_capability(n_fail=5, n_trials=50)
    b = proportion_capability(n_fail=6, n_trials=50)
    assert a.provenance_digest() != b.provenance_digest()
