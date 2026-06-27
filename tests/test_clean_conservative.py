"""Conservative clean(): mechanical recovery only; refuse-and-flag the ambiguous.

Per CLEAN_SCOPE_CONSERVATIVE: same input -> identical output; never fabricates a
value for an ambiguous token; surfaces what it could not safely do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mfgqc


# --------------------------------------------------------------------------- #
# coerce_numeric: mechanical recovery + refuse/flag
# --------------------------------------------------------------------------- #
def test_unit_strip_recovered():
    raw = pd.DataFrame({"w": ["1.48 mm", "1.55mm", "1.50", 1.42]})
    out = mfgqc.clean(raw, [mfgqc.coerce_numeric(["w"])], verbose=False)
    np.testing.assert_allclose(out["w"].to_numpy(), [1.48, 1.55, 1.50, 1.42])


def test_sentinels_to_na():
    raw = pd.DataFrame({"w": [999, -1, "N/A", "", "MISSING", 1.5]})
    out = mfgqc.clean(raw, [mfgqc.coerce_numeric(["w"]),
                           mfgqc.recode_missing(["w"], sentinels=[999, -1])], verbose=False)
    assert out["w"].isna().sum() == 5
    assert out["w"].dropna().tolist() == [1.5]


def test_ambiguous_comma_refused_and_flagged():
    raw = pd.DataFrame({"w": ["1,520", 2.0]})
    out = mfgqc.clean(raw, [mfgqc.coerce_numeric(["w"])], verbose=False)
    assert pd.isna(out["w"].iloc[0])           # NOT 1520, NOT 1.520
    flags = out.attrs["mfgqc_clean_flags"]
    assert any(f["reason"] == "ambiguous numeric" and f["value"] == "1,520" for f in flags)


def test_censored_refused_and_flagged():
    raw = pd.DataFrame({"w": ["<1.20", ">5.0", 3.0]})
    out = mfgqc.clean(raw, [mfgqc.coerce_numeric(["w"])], verbose=False)
    assert out["w"].isna().sum() == 2
    flags = out.attrs["mfgqc_clean_flags"]
    assert sum(f["reason"] == "censored" for f in flags) == 2


def test_never_imputes_na_stays_na():
    raw = pd.DataFrame({"w": [1.0, np.nan, "1,520", 2.0]})
    out = mfgqc.clean(raw, [mfgqc.coerce_numeric(["w"])], verbose=False)
    # original NaN + the refused ambiguous -> 2 NA; the two reals survive unchanged
    assert out["w"].isna().sum() == 2
    assert out["w"].dropna().tolist() == [1.0, 2.0]


# --------------------------------------------------------------------------- #
# parse_dates: unambiguous only
# --------------------------------------------------------------------------- #
def test_iso_parsed_ambiguous_flagged():
    raw = pd.DataFrame({"d": ["2026-01-15", "01/02/2026", "03/25/2026", "garbage"]})
    out = mfgqc.clean(raw, [mfgqc.parse_dates(["d"])], verbose=False)
    assert out["d"].iloc[0] == pd.Timestamp(2026, 1, 15)   # ISO
    assert pd.isna(out["d"].iloc[1])                        # 01/02 ambiguous -> NaT
    assert out["d"].iloc[2] == pd.Timestamp(2026, 3, 25)   # 25 > 12 -> M/D
    assert pd.isna(out["d"].iloc[3])                        # unparseable
    assert any(f["reason"] == "ambiguous date" for f in out.attrs["mfgqc_clean_flags"])


def test_parse_dates_fmt_honored_no_flags():
    raw = pd.DataFrame({"d": ["01/02/2026", "03/04/2026"]})
    out = mfgqc.clean(raw, [mfgqc.parse_dates(["d"], fmt="%m/%d/%Y")], verbose=False)
    assert out["d"].iloc[0] == pd.Timestamp(2026, 1, 2)    # user took responsibility
    assert not out.attrs["mfgqc_clean_flags"]


# --------------------------------------------------------------------------- #
# whitespace trimmed always; case NOT merged unless normalize_case
# --------------------------------------------------------------------------- #
def test_whitespace_trimmed_case_not_merged():
    raw = pd.DataFrame({"op": [" a ", "B ", " c", "A", "B", "C"]})
    out = mfgqc.clean(raw, [], verbose=False)
    assert set(out["op"]) == {"a", "B", "c", "A", "C"}     # trimmed, case kept distinct
    assert any(f["reason"] == "case/space variants" for f in out.attrs["mfgqc_clean_flags"])


def test_normalize_case_opt_in_merges():
    raw = pd.DataFrame({"op": [" a ", "A", "B", "b"]})
    out = mfgqc.clean(raw, [mfgqc.normalize_case(["op"])], verbose=False)
    assert set(out["op"]) == {"a", "b"}                    # merged on explicit opt-in


# --------------------------------------------------------------------------- #
# summary + determinism
# --------------------------------------------------------------------------- #
def test_end_of_run_summary_printed(capsys):
    raw = pd.DataFrame({"w": ["1,520", "<1.20", 2.0]})
    mfgqc.clean(raw, [mfgqc.coerce_numeric(["w"])])
    captured = capsys.readouterr().out
    assert "clean() flagged" in captured
    assert "ambiguous numeric" in captured and "censored" in captured


def test_deterministic_identical_output():
    raw = pd.DataFrame({"w": ["1.48 mm", "1,520", "<1.20", 999, 2.0], "op": [" a ", "A", "b", "B", "C"]})
    a = mfgqc.clean(raw, [mfgqc.coerce_numeric(["w"])], verbose=False)
    b = mfgqc.clean(raw, [mfgqc.coerce_numeric(["w"])], verbose=False)
    pd.testing.assert_frame_equal(a, b)
    assert a.attrs["mfgqc_clean_flags"] == b.attrs["mfgqc_clean_flags"]
