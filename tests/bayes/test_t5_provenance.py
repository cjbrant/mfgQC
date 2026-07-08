"""T5 - provenance and immutability for mfgqc.bayes result objects.

Exercises the bayes Step.params schema (plan D2) and the data_sha256 mechanism
(plan D3) that make the digest data-sensitive, since the shared load Step records
only n_affected (data.py:654-665).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mfgqc._result import history_digest
from mfgqc.bayes._results import fit_normal
from mfgqc.bayes.priors import NormalPrior


def _data(seed: int = 0, n: int = 40) -> np.ndarray:
    return np.random.default_rng(seed).normal(25.0, 0.5, size=n)


def test_t5_1_result_is_immutable():
    """T5.1: bayes result objects are frozen dataclasses; field assignment raises."""
    r = fit_normal(_data(), NormalPrior(0.0, 1.0, 1.0, 1.0), seed=1, draws=1000)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.mun = 99.0


def test_t5_2_digest_changes_iff_inputs_change_and_reruns_are_identical():
    """T5.2: the provenance digest changes iff (prior, data, seed, draws) change,
    and is bit-identical on rerun. The data-sensitivity is the data_sha256 payoff:
    the shared load Step records only n_affected, so without it a 1e-9 value edit
    would leave the digest unchanged.
    """
    y = _data(seed=0)
    prior = NormalPrior(0.0, 1.0, 1.0, 1.0)
    base = fit_normal(y, prior, seed=1, draws=1000)
    d = base.provenance_digest()

    # bit-identical rerun
    assert fit_normal(y, prior, seed=1, draws=1000).provenance_digest() == d

    # data change (single value nudged by 1e-9) -> digest changes
    y2 = y.copy()
    y2[0] += 1e-9
    assert fit_normal(y2, prior, seed=1, draws=1000).provenance_digest() != d

    # data reorder is a data change too
    y3 = y[::-1].copy()
    assert fit_normal(y3, prior, seed=1, draws=1000).provenance_digest() != d

    # prior change
    assert fit_normal(y, NormalPrior(0.1, 1.0, 1.0, 1.0), seed=1, draws=1000).provenance_digest() != d

    # seed change
    assert fit_normal(y, prior, seed=2, draws=1000).provenance_digest() != d

    # draws change
    assert fit_normal(y, prior, seed=1, draws=2000).provenance_digest() != d


def test_t5_3_from_result_embeds_parent_and_tamper_breaks_child_verify():
    """T5.3: a prior built via NormalPrior.from_result carries the parent posterior
    forward. The child embeds the parent digest in its params AND prepends the
    parent history, so editing any recorded parent step breaks the child's verify.
    """
    parent = fit_normal(_data(seed=0), NormalPrior(0.0, 1.0, 1.0, 1.0), seed=1, draws=1000)
    parent_digest = parent.provenance_digest()

    child_prior = NormalPrior.from_result(parent)
    child = fit_normal(_data(seed=9), child_prior, seed=1, draws=1000)
    saved = child.provenance_digest()

    # parent digest is embedded in the child's analysis step params
    embedded = [s for s in child.history
                if s.operation == "bayes.normal_fit"
                and s.params.get("prior", {}).get("parent_digest") == parent_digest]
    assert embedded, "child analysis step must embed the parent digest"

    # child verifies against its own captured digest
    assert child.verify_provenance(saved)

    # tampering any recorded parent step changes the child's digest -> verify fails
    tampered_idx = next(i for i, s in enumerate(child.history) if s.operation == "load"
                        or s.operation == "bayes.normal_fit")
    bad = dataclasses.replace(
        child.history[tampered_idx],
        params={**child.history[tampered_idx].params, "_tamper": True},
    )
    bad_history = child.history[:tampered_idx] + (bad,) + child.history[tampered_idx + 1:]
    assert history_digest(bad_history) != saved
