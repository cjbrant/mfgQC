"""Priors: elicitation round-trip (T1.10) and to_params structure.

T1.10 is normative per the spec. Oracle-free (self-checking against the prior's
own marginal). Hoff sec 5.5.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from mfgqc.bayes.priors import BetaPrior, GammaPrior, NormalPrior


def test_t1_10_normal_prior_from_interval_round_trip():
    """T1.10 (normative): a NormalPrior built from (interval, confidence, worth)
    reproduces that interval from its own marginal for mu, t_{nu0}(mu0, s20/k0).
    rtol 1e-10.
    """
    rng = np.random.default_rng(10)
    for _ in range(200):
        lo = float(rng.normal(0, 5))
        width = float(rng.uniform(0.1, 10))
        hi = lo + width
        conf = float(rng.choice([0.5, 0.8, 0.9, 0.95, 0.99]))
        worth = float(rng.uniform(1, 50))

        prior = NormalPrior.from_interval((lo, hi), confidence=conf, worth=worth)

        marg = stats.t(df=prior.nu0, loc=prior.mu0, scale=math.sqrt(prior.s20 / prior.k0))
        q_lo = float(marg.ppf((1.0 - conf) / 2.0))
        q_hi = float(marg.ppf((1.0 + conf) / 2.0))
        assert abs(q_lo - lo) <= 1e-10 * (1.0 + abs(lo))
        assert abs(q_hi - hi) <= 1e-10 * (1.0 + abs(hi))


def test_from_interval_with_explicit_sd_sets_variance_scale():
    """When sd= is supplied, s20 is sd**2 and nu0 defaults to the worth."""
    prior = NormalPrior.from_interval((24.0, 26.0), confidence=0.95, worth=20, sd=0.5)
    assert prior.mu0 == 25.0
    assert prior.k0 == 20
    assert prior.nu0 == 20
    assert abs(prior.s20 - 0.25) <= 1e-12


def test_to_params_emits_family_and_finite_hyperparams():
    """to_params() feeds the provenance digest, so it must be a plain dict of the
    family name plus finite hyperparameters (no NaN/inf, no objects)."""
    for prior, family, keys in [
        (NormalPrior(25.0, 20, 20, 0.0144), "normal", {"mu0", "k0", "nu0", "s20"}),
        (BetaPrior(2.0, 3.0), "beta", {"a", "b"}),
        (GammaPrior(2.0, 1.0), "gamma", {"a", "b"}),
    ]:
        p = prior.to_params()
        assert p["family"] == family
        assert set(p["hyperparams"]) == keys
        for v in p["hyperparams"].values():
            assert isinstance(v, float) and math.isfinite(v)
