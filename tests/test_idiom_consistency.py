"""Idiom-consistency fixes surfaced by blind validation v4.

- Hypothesis tests read the `group` role when `by` is omitted (the .roles() idiom).
- control_chart accepts kind='i' as an alias for 'i_mr' (individuals chart).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mfgqc


def _two_group(seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[rng.normal(50, 3, 40), rng.normal(50.5, 3, 40)]
    return pd.DataFrame({"y": y, "grp": ["A"] * 40 + ["B"] * 40})


def test_test_means_defaults_to_group_role():
    data = mfgqc.load(_two_group(), measure="y").roles(group="grp")
    r_role = data.test_means()            # via the group role
    r_expl = data.test_means(by="grp")    # explicit, must match
    assert r_role.statistic == pytest.approx(r_expl.statistic)


def test_test_variance_and_anova_default_to_group_role():
    data = mfgqc.load(_two_group(), measure="y").roles(group="grp")
    assert data.test_variance().statistic == pytest.approx(data.test_variance(by="grp").statistic)
    assert data.test_anova().statistic == pytest.approx(data.test_anova(by="grp").statistic)


def test_no_grouping_raises_clear_error():
    data = mfgqc.load(_two_group(), measure="y")  # no group role bound
    with pytest.raises(ValueError, match="group"):
        data.test_means()


def test_means_routes_by_default():
    # routing is the default: unequal variance -> Welch automatically, with the reason.
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": np.r_[rng.normal(10, 1, 30), rng.normal(10.5, 4, 30)],
                       "grp": ["A"] * 30 + ["B"] * 30})
    r = mfgqc.load(df, measure="y").roles(group="grp").test_means()
    assert r.test_used == "Welch's t" and r.routed is True
    assert "welch" in r.selection_reason.lower()
    # force pooled overrides and warns
    forced = mfgqc.load(df, measure="y").roles(group="grp").test_means(method="pooled")
    assert forced.test_used == "Student's t (pooled)"
    assert "welch" in (forced.recommendation or "").lower()


def test_control_chart_kind_i_alias():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(50, 1, 40)})
    a = mfgqc.load(df, measure="x", subgroup_size=1).control_chart(kind="i")
    b = mfgqc.load(df, measure="x", subgroup_size=1).control_chart(kind="i_mr")
    assert a.kind == "i_mr" and b.kind == "i_mr"
    assert float(a.location_cl) == pytest.approx(float(b.location_cl))


def test_i_chart_defaults_to_individuals_no_subgroup_size():
    # An explicit individuals chart with no subgroup role/size -> size-1 (not an error).
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(50, 1, 40)})        # no subgroup_size given
    r = mfgqc.load(df, measure="x").control_chart(kind="i")
    assert r.kind == "i_mr"
    assert len(r.location_points) == 40                    # each row is its own point
