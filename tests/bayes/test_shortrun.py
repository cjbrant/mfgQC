"""Short-run sequential chart (spec Algorithm L; C&dC ch. 3/6, simplified).
Gate: T1.18 (chart statistic), T3.9 (in-control + drift absorption), T5.6
(stage provenance chain), T6.8 (vague-start guard)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mfgqc._result import history_digest
from mfgqc.bayes.priors import NormalPrior
from mfgqc.bayes.shortrun import ShortRunResult, shortrun

TARGET, LOWER, UPPER = 25.0, 24.85, 25.15  # default d = 0.3/12 = 0.025
PRIOR = NormalPrior(mu0=25.0, k0=8.0, nu0=8.0, s20=0.01 ** 2)


def _in_control(n_stages=20, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.normal(25.0, 0.01, 5) for _ in range(n_stages)]


# ---- T1.18 chart statistic ----------------------------------------------- #
def test_t1_18_chart_statistic_matches_t_cdf():
    """T1.18: the per-stage P(|mu - target| > d) equals a direct t-CDF evaluation
    on the stage posterior, exactly."""
    res = shortrun(_in_control(6), target=TARGET, lower=LOWER, upper=UPPER, prior=PRIOR)
    d = res.d
    for i, (mun, kn, nun, sn2) in enumerate(res.stage_posterior):
        scale = np.sqrt(sn2 / kn)
        expected = float(stats.t.sf(TARGET + d, nun, mun, scale)
                         + stats.t.cdf(TARGET - d, nun, mun, scale))
        assert res.chart_statistic(i) == expected


def test_default_d_is_spec_width_over_12():
    res = shortrun(_in_control(3), target=TARGET, lower=LOWER, upper=UPPER, prior=PRIOR)
    assert res.d == pytest.approx((UPPER - LOWER) / 12.0)


# ---- T3.9 in-control + drift absorption ---------------------------------- #
def test_t3_9_in_control_rarely_flags():
    """T3.9: a stable in-control process centered on target almost never flags."""
    res = shortrun(_in_control(30), target=TARGET, lower=LOWER, upper=UPPER, prior=PRIOR)
    assert sum(res.flags) == 0
    assert res.first_flag() is None


def test_t3_9_slow_drift_is_absorbed_expected_behavior():
    """T3.9: a slow drift is partly absorbed into the accumulated reference, so it
    signals later than an abrupt shift of the same magnitude and the running mean
    lags the true mean. This is the documented, expected v1 behavior; the report
    carries the drift-absorption caveat."""
    rng = np.random.default_rng(2)
    abrupt = ([rng.normal(25.0, 0.01, 5) for _ in range(3)]
              + [rng.normal(25.10, 0.01, 5) for _ in range(12)])
    r_ab = shortrun(abrupt, target=TARGET, lower=LOWER, upper=UPPER, prior=PRIOR)

    levels = np.linspace(25.0, 25.10, 15)
    drift = [rng.normal(m, 0.01, 5) for m in levels]
    r_sd = shortrun(drift, target=TARGET, lower=LOWER, upper=UPPER, prior=PRIOR)

    # absorption: the drift signals strictly later than the abrupt shift...
    assert r_sd.first_flag() > r_ab.first_flag()
    # ...and the reference mean lags the true drifted mean at the end.
    assert r_sd.stage_posterior[-1][0] < r_sd.stage_mean[-1] - 0.01
    # the caveat is disclosed as expected behavior
    assert "drift-absorption caveat" in r_sd.report().lower()


# ---- T5.6 stage provenance chain ----------------------------------------- #
def test_t5_6_each_stage_records_predecessor_digest():
    res = shortrun(_in_control(5), target=TARGET, lower=LOWER, upper=UPPER, prior=PRIOR)
    stage_steps = [s for s in res.history if s.operation == "bayes.shortrun_stage"]
    assert len(stage_steps) == 5
    for i, step in enumerate(stage_steps):
        assert step.params["prev_digest"] == history_digest(tuple(stage_steps[:i]))


def test_t5_6_chain_verifies_and_is_data_sensitive_end_to_end():
    res = shortrun(_in_control(5, seed=1), target=TARGET, lower=LOWER, upper=UPPER, prior=PRIOR)
    saved = res.provenance_digest()
    assert res.verify_provenance(saved) is True
    assert res.verify_provenance("deadbeef") is False
    # reruns are bit-identical
    again = shortrun(_in_control(5, seed=1), target=TARGET, lower=LOWER, upper=UPPER, prior=PRIOR)
    assert again.provenance_digest() == saved

    # editing an EARLY subgroup changes the end-to-end digest
    groups = _in_control(5, seed=1)
    groups[0] = groups[0] + 0.5
    assert shortrun(groups, target=TARGET, lower=LOWER, upper=UPPER,
                    prior=PRIOR).provenance_digest() != saved


# ---- T6.8 vague-start guard ---------------------------------------------- #
def test_t6_8_vague_start_without_flag_raises_with_flag_warns():
    with pytest.raises(ValueError, match="proper prior"):
        shortrun(_in_control(4), target=TARGET, lower=LOWER, upper=UPPER)  # prior=None

    res = shortrun(_in_control(4), target=TARGET, lower=LOWER, upper=UPPER, allow_vague=True)
    assert isinstance(res, ShortRunResult)
    chk = next(a for a in res.assumptions if a.name == "vague_start")
    assert chk.passed is False and chk.recommendation is not None


def test_vague_start_zero_variance_first_subgroup_raises_cleanly():
    """A noninformative start from an all-identical first subgroup has no scale;
    must raise a clear error, not a divide-by-zero / NaN statistic."""
    groups = [[25.0, 25.0, 25.0], [25.01, 25.02, 25.0], [24.99, 25.0, 25.01]]
    with pytest.raises(ValueError, match="variance"):
        shortrun(groups, target=TARGET, lower=LOWER, upper=UPPER, allow_vague=True)


def test_degenerate_zero_scale_statistic_is_point_mass_not_nan():
    """A degenerate zero-scale posterior (prior s20=0, data on the prior mean)
    yields the point-mass limit (0 or 1), never NaN."""
    prior = NormalPrior(mu0=25.0, k0=8.0, nu0=8.0, s20=0.0)
    res = shortrun([[25.0, 25.0]], target=TARGET, lower=LOWER, upper=UPPER, prior=prior)
    assert res.chart_stat[0] in (0.0, 1.0)


def test_shortrun_handles_single_measurement_short_runs():
    """A proper prior lets each 'run' be a single part (n=1)."""
    runs = [[25.01], [24.99], [25.0], [25.02], [24.98]]
    res = shortrun(runs, target=TARGET, lower=LOWER, upper=UPPER, prior=PRIOR)
    assert res.n_stages == 5
    assert all(n == 1 for n in res.stage_n)
