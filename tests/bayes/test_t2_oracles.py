"""T2 - published-oracle fixtures for mfgqc.bayes (BDA3 / Hoff / Murphy).

Active tests pin the engines against values transcribed from a cited source (see
_oracles.py for the transcription notes). Tests whose source values are still
[LOCK] are skipped with an explicit marker rather than filled from memory.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mfgqc.bayes.conjugate import beta_posterior, gamma_posterior, gamma_update

from ._oracles import NEWCOMB_DATA, PLACENTA, PLACENTA_CI95, RATE_EXAMPLE, SPEED_OF_LIGHT


def test_t2_1_placenta_previa_posterior_summaries():
    """T2.1: BDA3 sec 2.4. Uniform prior Beta(1,1), y=437, n=980 -> Beta(438,544);
    posterior mean 0.446, sd 0.016, central 95% interval [0.415, 0.477]."""
    a, b = PLACENTA["prior"]
    post = beta_posterior(a, b, PLACENTA["y"], PLACENTA["n"])
    assert abs(float(post.mean()) - PLACENTA["posterior_mean"]) <= 5e-4
    assert abs(float(post.std()) - PLACENTA["posterior_sd"]) <= 5e-4
    lo, hi = (float(v) for v in post.ppf([0.025, 0.975]))
    assert (round(lo, 3), round(hi, 3)) == PLACENTA_CI95


def test_t2_3_asthma_rate_gamma_poisson():
    """T2.3: BDA3 sec 2.6. Prior Gamma(3,5), y=3, exposure x=2.0 -> posterior
    Gamma(6,7), mean 0.86, P(theta>1)=0.30; prior mean 0.6, 97.5% below 1.44."""
    a, b = RATE_EXAMPLE["prior"]
    assert gamma_update(a, b, RATE_EXAMPLE["y"], RATE_EXAMPLE["x"]) == RATE_EXAMPLE["posterior"]

    prior = stats.gamma(a, scale=1.0 / b)
    assert abs(float(prior.mean()) - RATE_EXAMPLE["prior_mean"]) <= 5e-3
    assert abs(float(prior.ppf(0.975)) - RATE_EXAMPLE["prior_p975"]) <= 1e-2

    post = gamma_posterior(a, b, RATE_EXAMPLE["y"], RATE_EXAMPLE["x"])
    assert abs(float(post.mean()) - RATE_EXAMPLE["posterior_mean"]) <= 5e-3
    assert abs(float(post.sf(1.0)) - RATE_EXAMPLE["p_theta_gt_1"]) <= 5e-3


def test_t2_2_speed_of_light_t_multiplier():
    """T2.2 (partial): BDA3 sec 3.2 states the t_65 0.975 multiplier as 1.997.
    The full 95% interval [23.6, 28.8] reproduction needs the noninformative fit
    (P1) and Newcomb's raw values, which BDA3 does not tabulate (see below)."""
    assert abs(float(stats.t.ppf(0.975, SPEED_OF_LIGHT["n"] - 1)) - SPEED_OF_LIGHT["t_mult"]) <= 5e-4


@pytest.mark.skipif(NEWCOMB_DATA is None,
                    reason="BDA3 sec 3.2 shows Newcomb's 66 values only as a histogram, not a table "
                           "(its Table 3.1 is the bioassay data); the raw vector needed to reproduce "
                           "[23.6,28.8] exactly is not in the source. The noninformative fit exists.")
def test_t2_2_speed_of_light_mu_interval():
    y = np.asarray(NEWCOMB_DATA, dtype=float)
    n = y.size
    lo, hi = stats.t.ppf([0.025, 0.975], n - 1, loc=y.mean(), scale=y.std(ddof=1) / np.sqrt(n))
    assert (round(float(lo), 1), round(float(hi), 1)) == SPEED_OF_LIGHT["mu_ci95"]
