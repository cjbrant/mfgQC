"""Fluent API idiom: load() -> .spec()/.roles() metadata setters -> analyses.

Spec limits and roles are QCData metadata (lowercase setters); there is no public
Spec class and no nested roles dict as the primary path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mfgqc


# --------------------------------------------------------------------------- #
# Regression: the oracle reproduces via the idiom (load + .spec)
# --------------------------------------------------------------------------- #
def test_capability_oracle_via_load_spec(montgomery_qc):
    df = montgomery_qc.frame  # tidy frame: columns subgroup, width
    data = mfgqc.load(df, measure="width", subgroup="subgroup", subgroup_size=5) \
               .spec(lower=1.0, upper=2.0, target=1.5)
    cap = data.capability()
    assert cap.sigma_within == pytest.approx(0.1398, abs=5e-4)
    assert cap.cp == pytest.approx(1.192, abs=5e-3)
    assert cap.cpk == pytest.approx(1.179, abs=5e-3)


# --------------------------------------------------------------------------- #
# No public Spec class
# --------------------------------------------------------------------------- #
def test_no_public_spec_class():
    assert not hasattr(mfgqc, "Spec")


# --------------------------------------------------------------------------- #
# .spec() is a metadata setter
# --------------------------------------------------------------------------- #
def test_spec_sets_fields(montgomery_qc):
    data = mfgqc.load(montgomery_qc.frame, measure="width").spec(lower=1.0, upper=2.0, target=1.5)
    assert data.meta.lower == 1.0 and data.meta.upper == 2.0 and data.meta.target == 1.5
    assert data.meta.has_spec is True


def test_spec_one_sided(montgomery_qc):
    data = mfgqc.load(montgomery_qc.frame, measure="width").spec(lower=1.0)
    assert data.meta.lower == 1.0 and data.meta.upper is None


def test_spec_lower_ge_upper_raises(montgomery_qc):
    data = mfgqc.load(montgomery_qc.frame, measure="width")
    with pytest.raises(ValueError, match="lower must be < upper"):
        data.spec(lower=2.0, upper=1.0)


def test_spec_is_immutable(montgomery_qc):
    data = mfgqc.load(montgomery_qc.frame, measure="width")
    specced = data.spec(lower=1.0, upper=2.0)
    assert specced is not data
    assert data.meta.has_spec is False      # original unchanged
    assert specced.meta.lower == 1.0
    assert any(s.operation == "spec" for s in specced.history)  # provenance


def test_spec_sweep_does_not_mutate_base(montgomery_qc):
    base = mfgqc.load(montgomery_qc.frame, measure="width", subgroup="subgroup", subgroup_size=5)
    results = [base.spec(lower=1.0, upper=u).capability() for u in (1.8, 2.0, 2.2)]
    cps = [r.cp for r in results]
    assert cps[0] < cps[1] < cps[2]         # wider tolerance -> larger Cp
    assert base.meta.has_spec is False      # base never mutated


def test_spec_from_copies_limits(montgomery_qc):
    src = mfgqc.load(montgomery_qc.frame, measure="width").spec(lower=1.0, upper=2.0, target=1.5)
    df2 = pd.DataFrame({"width": [1.4, 1.5, 1.6, 1.55]})
    dst = mfgqc.load(df2, measure="width").spec_from(src)
    assert dst.meta.lower == 1.0 and dst.meta.upper == 2.0 and dst.meta.target == 1.5


def test_dict_reuse_named_standard(montgomery_qc):
    WALL = dict(lower=1.0, upper=2.0, target=1.5)
    data = mfgqc.load(montgomery_qc.frame, measure="width").spec(**WALL)
    assert data.meta.upper == 2.0


# --------------------------------------------------------------------------- #
# .roles() setter (replaces the nested dict)
# --------------------------------------------------------------------------- #
def test_roles_setter_drives_gage_rr(aiag_qc):
    # rebuild from the raw frame using the .roles() setter (no nested dict)
    df = aiag_qc.frame
    data = mfgqc.load(df, measure="y").roles(part="part", operator="operator", replicate="trial")
    g = data.gage_rr(method="anova")
    assert g.n_parts == 10 and g.n_operators == 3 and g.n_trials == 3
    assert any(s.operation == "roles" for s in data.history)


def test_roles_merges_with_load_subgroup(montgomery_qc):
    data = mfgqc.load(montgomery_qc.frame, measure="width", subgroup="subgroup").roles(time="subgroup")
    assert data.meta.roles["subgroup"] == "subgroup"
    assert data.meta.roles["time"] == "subgroup"


def test_roles_reserved_quality_and_time_accepted():
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0], "q": ["a", "a", "b", "a"], "t": [1, 2, 3, 4]})
    data = mfgqc.load(df, measure="v").roles(quality="q", time="t")  # reserved, no error
    assert data.meta.roles["quality"] == "q" and data.meta.roles["time"] == "t"


def test_roles_unknown_column_raises():
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="not in the frame"):
        mfgqc.load(df, measure="v").roles(part="nope")


def test_spec_is_immutable_for_roles(montgomery_qc):
    data = mfgqc.load(montgomery_qc.frame, measure="width")
    withrole = data.roles(time="subgroup")
    assert withrole is not data
    assert "time" not in data.meta.roles


# --------------------------------------------------------------------------- #
# load() sugar
# --------------------------------------------------------------------------- #
def test_load_subgroup_sugar(montgomery_qc):
    data = mfgqc.load(montgomery_qc.frame, measure="width", subgroup="subgroup")
    assert data.meta.roles.get("subgroup") == "subgroup"


def test_load_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        mfgqc.load("some_path.csv", measure="x")


# --------------------------------------------------------------------------- #
# Wide -> long forward-compat path
# --------------------------------------------------------------------------- #
def test_from_wide_into_control_chart():
    rng = np.random.default_rng(0)
    wide = pd.DataFrame({"batch": range(1, 9)})
    for i in range(1, 6):  # 5 replicate columns
        wide[f"x{i}"] = rng.normal(10, 1, 8)
    data = mfgqc.from_wide(wide, id_vars=["batch"], measure="reading", subgroup="batch")
    cc = data.control_chart()
    assert cc.kind == "xbar_r"          # 8 subgroups of n=5
    assert data.meta.measure == "reading"
    assert any(s.operation == "from_wide" for s in data.history)
