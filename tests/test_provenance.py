"""Provenance & immutability contract (the differentiating claim, made load-bearing).

These tests make mfgQC's "immutable, auditable lineage" pillar enforceable rather
than asserted:

1. QCData is immutable through every public surface (frozen, defensive copies,
   append-only tuple history).
2. Provenance propagates across every transform and is reconstructable end to end.
3. The chain is tamper-evident: a SHA-256 digest pins it, and editing any recorded
   step is detectable.
"""

from __future__ import annotations

import dataclasses

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import mfgqc


# --------------------------------------------------------------------------- #
# 1. Immutability through every public surface
# --------------------------------------------------------------------------- #
def _qc():
    df = pd.DataFrame({"x": np.random.default_rng(0).normal(0, 1, 60)})
    return mfgqc.load(df, measure="x").spec(lower=-3, upper=3)


def test_qcdata_is_frozen():
    qc = _qc()
    with pytest.raises(dataclasses.FrozenInstanceError):
        qc.meta = None
    with pytest.raises(dataclasses.FrozenInstanceError):
        qc.history = ()


def test_frame_accessor_returns_isolated_copy():
    qc = _qc()
    f = qc.frame
    f.iloc[0, 0] = 9999.0
    assert qc.frame.iloc[0, 0] != 9999.0  # mutation did not reach the original


def test_values_accessor_cannot_corrupt_original():
    qc = _qc()
    v = qc.values()
    try:
        v[:] = -1.0          # read-only array raises; a plain copy would not
    except ValueError:
        pass
    assert not np.allclose(qc.values(), -1.0)  # original data intact either way


def test_history_is_append_only_tuple():
    qc = _qc()
    assert isinstance(qc.history, tuple)  # cannot append/insert/reorder in place
    with pytest.raises(AttributeError):
        qc.history.append("x")  # type: ignore[attr-defined]


def test_ingest_defensively_copies_input():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    qc = mfgqc.load(df, measure="x")
    df.iloc[0, 0] = 9999.0  # mutate the caller's frame after load
    assert qc.frame.iloc[0, 0] == 1.0  # QCData held its own copy


def test_result_objects_are_frozen():
    cap = _qc().capability()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cap.cpk = 5.0


# --------------------------------------------------------------------------- #
# 2. Provenance propagates and is reconstructable
# --------------------------------------------------------------------------- #
def test_lineage_reconstructs_full_chain_through_transform():
    """A non-normal capability with an opted-in Box-Cox records the whole chain:
    ingest -> spec -> transform -> analysis -> assumption check."""
    rng = np.random.default_rng(2)
    skew = mfgqc.load(pd.DataFrame({"y": rng.lognormal(0, 0.6, 200)}),
                     measure="y").spec(lower=0.1, upper=8)
    cap = skew.transform("boxcox").capability()
    ops = [s["operation"] for s in cap.lineage()]
    assert ops == ["load", "spec", "transform", "capability", "assumption:normality"]


def test_lineage_steps_carry_running_digest():
    cap = _qc().capability()
    lin = cap.lineage()
    assert all("digest" in s and len(s["digest"]) == 64 for s in lin)
    # the chain is cumulative: each step's digest differs from the previous one
    digests = [s["digest"] for s in lin]
    assert len(set(digests)) == len(digests)


def test_qcdata_and_result_share_the_prefix_chain():
    qc = _qc()
    cap = qc.capability()
    # the QCData lineage is a prefix of the result lineage (same ingest/spec steps)
    qc_ops = [s["operation"] for s in qc.lineage()]
    cap_ops = [s["operation"] for s in cap.lineage()]
    assert cap_ops[:len(qc_ops)] == qc_ops


# --------------------------------------------------------------------------- #
# 3. Tamper-evidence (hash-chained, verifiable)
# --------------------------------------------------------------------------- #
def test_digest_is_stable_across_runs():
    """Same pipeline on the same data yields the same digest (timestamp excluded)."""
    df = pd.DataFrame({"x": np.random.default_rng(5).normal(0, 1, 50)})
    a = mfgqc.load(df, measure="x").spec(lower=-3, upper=3).capability().provenance_digest()
    b = mfgqc.load(df, measure="x").spec(lower=-3, upper=3).capability().provenance_digest()
    assert a == b and len(a) == 64


def test_verify_provenance_detects_tampering():
    cap = _qc().capability()
    good = cap.provenance_digest()
    assert cap.verify_provenance(good) is True
    # edit a recorded step's params in place (the known mutable-dict tamper vector)
    cap.history[0].params["measure"] = "TAMPERED"
    assert cap.verify_provenance(good) is False
    assert cap.provenance_digest() != good


def test_digest_changes_when_any_step_changes():
    base = _qc().capability().provenance_digest()
    # a different spec -> a different recorded 'spec' step -> a different digest
    other = (mfgqc.load(pd.DataFrame({"x": np.random.default_rng(0).normal(0, 1, 60)}),
                       measure="x").spec(lower=-2, upper=2).capability().provenance_digest())
    assert base != other


def test_to_dict_carries_provenance_digest():
    cap = _qc().capability()
    payload = cap.to_dict()
    assert payload["provenance_digest"] == cap.provenance_digest()
    assert payload["history"] == cap.lineage()
