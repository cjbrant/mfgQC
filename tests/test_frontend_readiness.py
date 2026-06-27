"""Frontend-readiness contract tests (packaging pass for v0.1.0).

The frontend will be a thin shell over the public API: load -> spec/roles ->
analysis, then render .summary()/.view(). These tests pin the package-level
capabilities that shell depends on, so they can't silently regress:

1. Every result's .summary() and .to_dict() are JSON-serializable.
2. .view() renders headlessly and can save to PNG and SVG, returning a Figure.
3. A machine-readable analysis registry (mfgqc.ANALYSES) exists and is serializable.
4. Missing prerequisites raise a specific, catchable MissingPrerequisiteError.
5. clean()'s flagged items are machine-readable via clean_report().
"""

from __future__ import annotations

import itertools
import json
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless: no display required

import mfgqc
from mfgqc.errors import MissingPrerequisiteError, PyQCError


# --------------------------------------------------------------------------- #
# Build one instance of every result type, via the PUBLIC API only.
# --------------------------------------------------------------------------- #
def _build_all_results() -> dict:
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    R: dict = {}
    mont = pd.DataFrame({"sg": np.repeat(range(1, 26), 5), "w": rng.normal(1.5, 0.15, 125)})
    qc = mfgqc.load(mont, measure="w", subgroup="sg").spec(lower=1, upper=2, target=1.5)
    R["Capability"] = qc.capability()
    R["ControlChart_xbar"] = qc.control_chart(kind="xbar_r")
    R["EWMA"] = mfgqc.load(mont, measure="w").ewma_chart()
    R["CUSUM"] = mfgqc.load(mont, measure="w").cusum_chart()
    R["ControlChart_c"] = mfgqc.load(pd.DataFrame({"d": [16] * 25}), measure="d").control_chart(kind="c")
    a = rng.normal(0, 1, 30)
    b = rng.normal(0.5, 1, 30)
    R["Hypothesis_means"] = mfgqc.test_means(a, b)
    R["Hypothesis_mean"] = mfgqc.test_mean(a, 0.0)
    R["Hypothesis_variance"] = mfgqc.test_variance(a, b)
    R["Hypothesis_proportion"] = mfgqc.test_proportion(12, 100, 0.1)
    R["Hypothesis_medians"] = mfgqc.test_medians(a, b)
    R["Anova"] = mfgqc.load(pd.DataFrame({"y": np.r_[a, b], "g": ["a"] * 30 + ["b"] * 30}),
                           measure="y").anova(factors=["g"])
    df = pd.DataFrame({"y": a * 2 + rng.normal(0, 0.3, 30), "x": a})
    R["Regression"] = mfgqc.load(df, measure="y").regress(on="x")
    R["Correlation"] = mfgqc.correlation(df)
    R["Pareto"] = mfgqc.pareto(pd.Series({"A": 50, "B": 30, "C": 20}))
    R["Contingency"] = mfgqc.contingency([[30, 20], [15, 35]])
    R["ProcessSigma"] = mfgqc.process_sigma(23, 1000, 5)
    R["Posthoc"] = mfgqc.test_anova(a, b, rng.normal(1, 1, 30)).posthoc(method="tukey")
    R["Precontrol"] = mfgqc.load(mont, measure="w").spec(lower=1, upper=2, target=1.5).precontrol()
    R["TimeSeries"] = mfgqc.load(pd.DataFrame({"y": np.arange(40) * 0.3 + rng.normal(0, 1, 40)}),
                                measure="y").timeseries()
    R["Power"] = mfgqc.power.t_test(effect=0.5, n=64)
    rows = [{"A": A, "B": B, "C": C, "y": 10 + 3 * A - 2 * B + rng.normal(0, 0.5)}
            for A, B, C in itertools.product([-1, 1], repeat=3) for _ in range(2)]
    R["DOE"] = mfgqc.load(pd.DataFrame(rows), measure="y").doe(factors=["A", "B", "C"])
    t = rng.weibull(2.0, 80) * 100
    R["LifeFit"] = mfgqc.load(pd.DataFrame({"t": t}), measure="t").life_fit()
    R["MTBF"] = mfgqc.reliability.mtbf(1000, 5)
    R["System"] = mfgqc.reliability.series([0.9, 0.95, 0.99])
    R["Availability"] = mfgqc.availability(mtbf=100, mttr=5)
    R["Bearing"] = mfgqc.reliability.bearing_life(C=50, P=5, rpm=1000)
    R["Demonstration"] = mfgqc.reliability.demonstration_test(reliability=0.9, confidence=0.9)
    R["Sampling"] = mfgqc.sampling_plan(n=50, c=2)
    R["Z19"] = mfgqc.z19_plan(lot_size=100, aql=1.0)
    return R


_RESULTS = _build_all_results()
_RESULT_IDS = sorted(_RESULTS)


# --------------------------------------------------------------------------- #
# 1. Serialization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", _RESULT_IDS)
def test_summary_is_json_serializable(name):
    """Every result's .summary() is a flat, json.dumps-able dict."""
    r = _RESULTS[name]
    s = r.summary()
    assert isinstance(s, dict)
    json.dumps(s)  # raises if not serializable
    # flat per the contract {str: number|str|bool|list}: scalar values, or a
    # one-level list of scalars -- never a nested dict.
    for k, v in s.items():
        assert isinstance(k, str)
        if isinstance(v, list):
            assert all(x is None or isinstance(x, (bool, int, float, str)) for x in v), (name, k)
        else:
            assert v is None or isinstance(v, (bool, int, float, str)), (name, k, type(v))


@pytest.mark.parametrize("name", _RESULT_IDS)
def test_to_dict_is_json_serializable(name):
    """Every result's .to_dict() is a json.dumps-able structured payload."""
    r = _RESULTS[name]
    d = r.to_dict()
    json.dumps(d)
    assert d["result_type"] == type(r).__name__
    assert set(d) >= {"result_type", "title", "summary", "fields", "assumptions", "history"}


@pytest.mark.parametrize("name", _RESULT_IDS)
def test_history_is_serializable_in_to_dict(name):
    """Provenance history round-trips into to_dict as a list of step dicts."""
    d = _RESULTS[name].to_dict()
    assert isinstance(d["history"], list)
    json.dumps(d["history"])


# --------------------------------------------------------------------------- #
# 2. Charts to file/buffer, headless
# --------------------------------------------------------------------------- #
# Results whose .view() draws a chart (a few are text-only / not charted).
_CHARTABLE = [n for n in _RESULT_IDS
              if n not in ("ProcessSigma", "MTBF", "System", "Availability",
                           "Bearing", "Demonstration", "Sampling", "Z19", "Power")]


@pytest.mark.parametrize("name", _CHARTABLE)
def test_view_returns_figure(name):
    """view() returns a matplotlib Figure under the Agg backend."""
    from matplotlib.figure import Figure
    fig = _RESULTS[name].view()
    assert isinstance(fig, Figure)


@pytest.mark.parametrize("name", _CHARTABLE)
def test_view_saves_png_and_svg(name, tmp_path):
    """view(save=...) writes a non-empty PNG and SVG to disk."""
    for ext in ("png", "svg"):
        path = tmp_path / f"{name}.{ext}"
        fig = _RESULTS[name].view(save=str(path))
        assert path.exists() and path.stat().st_size > 0
        from matplotlib.figure import Figure
        assert isinstance(fig, Figure)


# --------------------------------------------------------------------------- #
# 3. Enumerable capabilities (registry)
# --------------------------------------------------------------------------- #
def test_registry_exists_and_serializable():
    assert len(mfgqc.ANALYSES) > 20
    catalog = mfgqc.list_analyses()
    json.dumps(catalog)
    names = {a["name"] for a in catalog}
    assert {"capability", "control_chart", "regress", "anova", "doe", "life_fit"} <= names


def test_registry_entries_well_formed():
    valid_kinds = {"fluent", "function"}
    for a in mfgqc.ANALYSES:
        assert a.kind in valid_kinds
        assert a.requires and all(isinstance(x, str) for x in a.requires)
        assert a.result_type and a.description


def test_registry_requirements_match_capability_needs():
    """capability is registered as needing a measure and a spec."""
    cap = mfgqc.ANALYSES_BY_NAME["capability"]
    assert "measure" in cap.requires and "spec" in cap.requires


# --------------------------------------------------------------------------- #
# 4. Specific, catchable missing-prerequisite errors
# --------------------------------------------------------------------------- #
def test_capability_without_spec_raises_specific_error():
    qc = mfgqc.load(pd.DataFrame({"x": np.random.default_rng(0).normal(size=50)}), measure="x")
    with pytest.raises(MissingPrerequisiteError) as exc:
        qc.capability()
    assert exc.value.analysis == "capability"
    assert "spec" in exc.value.missing


def test_missing_prereq_is_also_valueerror():
    """Subclassing ValueError keeps existing `except ValueError` callers working."""
    assert issubclass(MissingPrerequisiteError, ValueError)
    assert issubclass(MissingPrerequisiteError, PyQCError)


def test_gage_without_roles_reports_missing_roles():
    qc = mfgqc.load(pd.DataFrame({"x": np.random.default_rng(1).normal(size=30)}), measure="x")
    with pytest.raises(MissingPrerequisiteError) as exc:
        qc.gage_rr()
    assert any(m.startswith("role:") for m in exc.value.missing)


def test_precontrol_without_spec_raises():
    qc = mfgqc.load(pd.DataFrame({"x": np.random.default_rng(2).normal(size=40)}), measure="x")
    with pytest.raises(MissingPrerequisiteError):
        qc.precontrol()


# --------------------------------------------------------------------------- #
# 5. clean() machine-readable flag summary
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 6. Stable public API surface: load() is the single blessed idiom
# --------------------------------------------------------------------------- #
def test_load_is_the_public_idiom():
    assert "load" in mfgqc.__all__


def test_legacy_ingestion_removed():
    """The legacy from_dataframe / from_csv entry points are gone; load() is the
    single public ingestion idiom."""
    assert "from_dataframe" not in mfgqc.__all__
    assert "from_csv" not in mfgqc.__all__
    assert not hasattr(mfgqc, "from_dataframe")
    assert not hasattr(mfgqc, "from_csv")


def test_all_exports_resolve():
    """Everything advertised in __all__ actually exists on the package."""
    for name in mfgqc.__all__:
        assert hasattr(mfgqc, name), name


def test_clean_report_is_machine_readable():
    df = pd.DataFrame({"Part ID": ["a", "a", "b", "c"],
                       "Value mm": ["1.2", "1.3 mm", "bad", "1.5"],
                       "Extra": ["x", "X", " y", "y"]})
    out = mfgqc.clean(df, verbose=False)
    rep = mfgqc.clean_report(out)
    json.dumps(rep)
    assert rep["n_flagged"] == len(rep["flags"])
    assert rep["n_flagged"] >= 1
    for f in rep["flags"]:
        assert {"task", "column", "reason"} <= set(f)
