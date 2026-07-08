"""P1 integration: the bayes capability analysis attaches like every other
analysis - a QCData fluent method, an Analysis registry row, and a top-level
re-export - so mfgQC Studio can discover and render it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mfgqc
from mfgqc.bayes.capability import BayesCapabilityResult


def _qc(spec: bool = True):
    df = pd.DataFrame({"width": np.random.default_rng(0).normal(25.0, 0.5, 60)})
    qc = mfgqc.load(df, measure="width")
    return qc.spec(lower=23.0, upper=27.0) if spec else qc


def test_qcdata_bayes_capability_is_fluent_and_chains_provenance():
    r = _qc().bayes_capability(seed=7, draws=2000)
    assert isinstance(r, BayesCapabilityResult)
    assert r.n == 60
    ops = [s.operation for s in r.history]
    assert "load" in ops           # chained onto the QCData load step
    assert "bayes.capability" in ops


def test_bayes_capability_requires_attached_spec():
    with pytest.raises(mfgqc.MissingPrerequisiteError):
        _qc(spec=False).bayes_capability(seed=7, draws=2000)


def test_bayes_capability_registered_for_discovery():
    a = mfgqc.ANALYSES_BY_NAME["bayes_capability"]
    assert a.call == "QCData.bayes_capability"
    assert a.kind == "fluent"
    assert a.result_type == "BayesCapabilityResult"
    assert "spec" in a.requires


def test_bayes_submodule_and_result_are_exposed():
    assert hasattr(mfgqc, "bayes")
    assert "bayes" in mfgqc.__all__
    from mfgqc import BayesCapabilityResult as _BCR
    assert _BCR is BayesCapabilityResult


def test_hr4_no_em_dash_in_bayes_reports():
    """House Rule 4: no em-dash or en-dash in any .report() output text."""
    from mfgqc.bayes._results import fit_normal
    from mfgqc.bayes.capability import capability_from_values
    from mfgqc.bayes.priors import NormalPrior

    y = np.random.default_rng(0).normal(25.0, 0.5, 60)
    reports = [
        fit_normal(y, NormalPrior(25.0, 20, 20, 0.25), seed=1, draws=1000).report(),
        capability_from_values(y, lower=23.0, upper=27.0, seed=1, draws=2000).report(),
        _qc().bayes_capability(seed=1, draws=2000).report(),
    ]
    for rep in reports:
        assert "—" not in rep  # em-dash
        assert "–" not in rep  # en-dash
