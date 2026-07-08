"""Phase-1 reference + Bayesian monitoring (spec Algorithm G; BDA3 sec 6.3).

Gate: T1.12 (p-value definition), T2.5(d) (worked-example monitor), T3.5
(calibration + shift detection), T5.4-T5.5 (provenance & custom-test hashing).
"""
from __future__ import annotations

import numpy as np
import pytest

from mfgqc.bayes.monitoring import FrozenReference, monitor, phase1


def _phase1_reference():
    """The worked-example phase-1 reference: noninformative fit of 60 values."""
    y = np.random.default_rng(20260703).normal(25.006, 0.011, size=(12, 5)).ravel()
    return phase1(y), y


def _worked_subgroups():
    stream = np.random.default_rng(17)
    return [stream.normal(25.006, 0.011, 5),
            stream.normal(25.006, 0.011, 5),
            stream.normal(25.026, 0.011, 5)]


# ---- T2.5(d) worked example ---------------------------------------------- #
def test_t2_5_d_monitor_worked_example():
    """T2.5(d): monitor (alpha=0.005, R=10000, tests mean+sd, seed 13) on the
    worked example gives p-values [[0.543,0.779],[0.628,0.059],[0.003,0.106]];
    only L-015 flags, on the mean."""
    ref, _ = _phase1_reference()
    res = monitor(ref, _worked_subgroups(), tests=("mean", "sd"),
                  alpha=0.005, R=10_000, seed=13)
    pv = [[round(p, 3) for p in row] for row in res.p_values_matrix()]
    assert pv == [[0.543, 0.779], [0.628, 0.059], [0.003, 0.106]]
    assert res.flags == [False, False, True]


# ---- T1.12 p-value definition -------------------------------------------- #
def test_t1_12_pvalue_matches_hand_computation():
    """T1.12: on a tiny seeded case the monitor p equals the explicit formula
    p = min(2*min(phi, 1-phi), 1) with phi = mean(T_rep >= T_obs), recomputed
    independently from the same normative draws."""
    ref, _ = _phase1_reference()
    sub = np.array([25.01, 25.00, 24.99, 25.02, 25.00])
    R, seed = 200, 5
    res = monitor(ref, [sub], tests=("mean",), alpha=0.005, R=R, seed=seed)

    rng = np.random.default_rng(seed)
    sig2r = ref.nun * ref.sn2 / rng.chisquare(ref.nun, R)
    mur = rng.normal(ref.mun, np.sqrt(sig2r / ref.kn))
    reps = rng.normal(mur[:, None], np.sqrt(sig2r)[:, None], size=(R, sub.size))
    phi = float((reps.mean(1) >= sub.mean()).mean())
    expected = min(2 * min(phi, 1 - phi), 1.0)
    assert res.p_values_matrix()[0][0] == expected


# ---- T5.4 provenance & construction contract ----------------------------- #
def test_t5_4_monitor_carries_reference_digest_and_rejects_raw_data():
    ref, y = _phase1_reference()
    res = monitor(ref, _worked_subgroups(), tests=("mean", "sd"), R=2000, seed=13)
    assert res.reference_digest == ref.digest

    with pytest.raises(TypeError):
        monitor(y, _worked_subgroups(), tests=("mean",), R=100, seed=1)  # raw data


def test_t5_4_frozen_reference_round_trips_with_digest_check():
    ref, _ = _phase1_reference()
    d = ref.to_dict()
    back = FrozenReference.from_dict(d)
    assert back.digest == ref.digest
    assert (back.mun, back.kn, back.nun, back.sn2) == (ref.mun, ref.kn, ref.nun, ref.sn2)

    tampered = dict(d)
    tampered["mun"] = ref.mun + 1.0  # edit content but keep the stored digest
    with pytest.raises(ValueError):
        FrozenReference.from_dict(tampered)


def test_reference_is_immutable():
    import dataclasses
    ref, _ = _phase1_reference()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.mun = 0.0  # type: ignore[misc]


# ---- T5.5 custom test-quantity source hashing ---------------------------- #
def test_t5_5_custom_test_source_hash_recorded_and_sensitive():
    ref, _ = _phase1_reference()

    def spread(a):
        return a.max(axis=-1) - a.min(axis=-1)

    r1 = monitor(ref, _worked_subgroups(), tests=(spread,), R=2000, seed=13)
    recorded = r1.test_specs()[0]
    assert recorded["name"] == "spread"
    assert "source_sha256" in recorded and len(recorded["source_sha256"]) == 64

    def spread(a):  # same name, different body -> different hash
        return a.max(axis=-1) - a.min(axis=-1) + 1.0

    r2 = monitor(ref, _worked_subgroups(), tests=(spread,), R=2000, seed=13)
    assert r2.test_specs()[0]["source_sha256"] != recorded["source_sha256"]
    assert r1.provenance_digest() != r2.provenance_digest()


# ---- T3.5 calibration + shift detection ---------------------------------- #
def test_t3_5_in_control_flag_rate_within_bound():
    """T3.5: an in-control stream flags at a rate no worse than the family-wise
    1-(1-alpha)^k bound (plus MC slack) over many subgroups."""
    ref, _ = _phase1_reference()
    rng = np.random.default_rng(2024)
    subs = [rng.normal(25.006, 0.011, 5) for _ in range(3000)]
    res = monitor(ref, subs, tests=("mean", "sd"), alpha=0.005, R=4000, seed=99)
    rate = sum(res.flags) / len(res.flags)
    bound = 1.0 - (1.0 - 0.005) ** 2
    assert rate <= bound + 0.01


def test_t3_5_three_sigma_shift_detected():
    """T3.5: a 3-sigma mean shift is caught by the mean test in nearly every rep."""
    ref, _ = _phase1_reference()
    sigma = 0.011
    detected = 0
    reps = 200
    for i in range(reps):
        rng = np.random.default_rng(1000 + i)
        shifted = rng.normal(25.006 + 3 * sigma, sigma, 5)
        res = monitor(ref, [shifted], tests=("mean", "sd"), alpha=0.005, R=2000, seed=7)
        if res.flags[0]:
            detected += 1
    assert detected / reps >= 0.95
