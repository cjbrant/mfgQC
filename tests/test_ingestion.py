"""Ingestion layer: overview() three faces, clean() pipeline + the QC-safety boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mfgqc


@pytest.fixture
def messy_df():
    return pd.DataFrame({
        "Width (mm)": ["1.0", "2.0", "999", "3.0", "1.5", ""],  # numeric-as-text + sentinel + empty
        "Batch": [1, 1, 2, 2, 3, 3],
        "Constant": [7, 7, 7, 7, 7, 7],                          # constant column
    })


# --------------------------------------------------------------------------- #
# overview() - three faces
# --------------------------------------------------------------------------- #
def test_overview_dataframe_flags(messy_df):
    ov = mfgqc.overview(messy_df)
    text = repr(ov)
    assert "rows" in text and "cols" in text
    assert "Role candidates" in text
    assert "constant column" in text.lower()
    assert "numeric-as-text" in text.lower()
    assert "sentinel" in text.lower()
    assert ov.kind == "dataframe"


def test_overview_qcdata(montgomery_qc):
    data = mfgqc.load(montgomery_qc.frame, measure="width", subgroup="subgroup",
                     subgroup_size=5).spec(lower=1.0, upper=2.0, target=1.5)
    text = repr(mfgqc.overview(data))
    assert "measure='width'" in text
    assert "skew" in text and "kurtosis" in text
    assert "spec:" in text.lower()
    assert "subgroups" in text.lower()


def test_overview_result_presents_no_recompute(montgomery_qc):
    cap = mfgqc.load(montgomery_qc.frame, measure="width", subgroup="subgroup",
                    subgroup_size=5).spec(lower=1.0, upper=2.0).capability()
    text = repr(mfgqc.overview(cap))
    assert "Assumption checks:" in text
    assert "no recomputation" in text.lower()


# --------------------------------------------------------------------------- #
# clean() - structural only, never alters surviving values
# --------------------------------------------------------------------------- #
def test_clean_default_bundle(messy_df):
    out = mfgqc.clean(messy_df)
    steps = out.attrs["mfgqc_clean_steps"]
    ops = [s.operation for s in steps]
    assert "clean.fix_names" in ops
    assert "clean.recode_empty" in ops
    assert "clean.drop_duplicates" in ops
    assert "width_(mm)" in out.columns or "width_(mm)" in [c for c in out.columns]


def test_clean_recode_missing_to_na_never_imputes():
    raw = pd.DataFrame({"width": [1.0, 2.0, 999.0, 3.0], "batch": [1, 1, 2, 2]})
    out = mfgqc.clean(raw, [mfgqc.recode_missing(["width"], sentinels=[999])])
    # 999 -> NA
    assert pd.isna(out["width"].iloc[2])
    # nothing imputed: the NA stays NA, count is exactly 1
    assert out["width"].isna().sum() == 1
    # surviving values unchanged
    assert out["width"].dropna().tolist() == [1.0, 2.0, 3.0]


def test_clean_preserves_surviving_values():
    """Part 5 guard: clean alters NO surviving non-NA measurement value."""
    rng = np.random.default_rng(0)
    vals = rng.normal(10, 1, 50)
    raw = pd.DataFrame({"Measure": [str(v) for v in vals], "junk": ["x"] * 50})
    out = mfgqc.clean(raw, [mfgqc.fix_names(), mfgqc.coerce_numeric(["measure"])])
    np.testing.assert_allclose(out["measure"].to_numpy(), vals, rtol=1e-12)


def test_clean_report_absorbed_into_load_history():
    raw = pd.DataFrame({"Width": [1.0, 2.0, 3.0, 4.0], "Batch": [1, 1, 2, 2]})
    cleaned = mfgqc.clean(raw, [mfgqc.fix_names()])
    data = mfgqc.load(cleaned, measure="width", subgroup="batch")
    ops = [s.operation for s in data.history]
    assert any(o.startswith("clean.") for o in ops)
    assert "load" in ops


def test_standard_tidy_expands_to_atomic_steps():
    raw = pd.DataFrame({"Width": ["1", "2", "3"], "When": ["2024-01-01", "2024-01-02", "2024-01-03"],
                        "Const": [1, 1, 1]})
    out = mfgqc.clean(raw, [mfgqc.standard_tidy(date_cols=["when"], numeric_cols=["width"])])
    ops = [s.operation for s in out.attrs["mfgqc_clean_steps"]]
    # bundle expanded into atomic steps (auditable, not opaque)
    assert "clean.fix_names" in ops
    assert "clean.coerce_numeric" in ops
    assert "clean.parse_dates" in ops
    assert "clean.drop_constant" in ops


def test_clean_drop_constant_and_duplicates():
    raw = pd.DataFrame({"a": [1, 1, 2, 2], "const": [9, 9, 9, 9]})
    out = mfgqc.clean(raw, [mfgqc.drop_constant(), mfgqc.drop_duplicates()])
    assert "const" not in out.columns
    assert len(out) == 2  # (1),(2) after dropping dup rows
