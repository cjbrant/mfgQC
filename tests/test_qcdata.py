"""QCData immutability, structured provenance, and lazy view validation."""

from __future__ import annotations

import pandas as pd
import pytest

import mfgqc
from mfgqc.data import Step


def test_transform_returns_new_object(montgomery_qc):
    original_len = len(montgomery_qc)
    original_hist_len = len(montgomery_qc.history)

    cleaned = montgomery_qc.clean(drop_na=True)

    assert cleaned is not montgomery_qc
    # original untouched
    assert len(montgomery_qc) == original_len
    assert len(montgomery_qc.history) == original_hist_len
    # transform appended exactly one step
    assert len(cleaned.history) == original_hist_len + 1


def test_history_grows_and_is_structured(montgomery_qc):
    first = montgomery_qc.history[0]
    assert isinstance(first, Step)
    assert first.operation == "load"
    assert isinstance(first.params, dict)
    assert first.params["measure"] == "width"
    assert first.n_affected == len(montgomery_qc)

    cleaned = montgomery_qc.clean(sigma_clip=3.0)
    last = cleaned.history[-1]
    assert last.operation == "clean"
    assert last.params["sigma_clip"] == 3.0
    assert isinstance(last.n_affected, int)
    # steps are structured records, never prose strings
    assert all(isinstance(s, Step) for s in cleaned.history)


def test_view_missing_role_errors():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]})
    qc = mfgqc.load(df, measure="y")  # no roles declared

    with pytest.raises(ValueError, match="subgroup"):
        qc.subgroups()

    with pytest.raises(ValueError) as exc:
        qc.crossed()
    msg = str(exc.value)
    assert "gage R&R requires roles" in msg
    assert "part" in msg


def test_crossed_names_specific_missing_role():
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "part": [1, 1, 2, 2],
        "operator": ["A", "A", "A", "A"],
        # 'replicate' deliberately absent
    })
    qc = mfgqc.load(df, measure="y", roles={"part": "part", "operator": "operator"})
    with pytest.raises(ValueError, match="replicate"):
        qc.crossed()


def test_construction_validates_spec_and_measure():
    df = pd.DataFrame({"y": [1.0, 2.0], "label": ["a", "b"]})
    with pytest.raises(ValueError, match="not found"):
        mfgqc.load(df, measure="missing")
    with pytest.raises(ValueError, match="numeric"):
        mfgqc.load(df, measure="label")
    with pytest.raises(ValueError, match="lower"):
        mfgqc.load(df, measure="y").spec(lower=5.0, upper=1.0)
