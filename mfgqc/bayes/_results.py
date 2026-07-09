"""Bayes result objects, the provenance schema, and the normal-fit entry point.

Every bayes result is a frozen dataclass subclassing mfgqc._result.QCResult, so it
inherits report(), summary(), to_dict(), lineage(), provenance_digest(), and
verify_provenance(). The analysis Step records a params schema (plan D2) that
includes a data_sha256 fingerprint (plan D3): the shared load Step records only
n_affected (data.py:654-665), so the digest is otherwise blind to value edits.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from mfgqc._result import QCResult
from mfgqc.assumptions import AssumptionCheck
from mfgqc.data import Step

from .conjugate import mu_marginal, suffstats, update
from .guardrails import (
    prior_conflict_check,
    prior_weight_check,
    require_min_n,
    small_sample_check,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _assumption_step(a: AssumptionCheck) -> Step:
    """One provenance Step per assumption check, mirroring capability.py:124."""
    return Step(
        operation=f"assumption:{a.name}",
        params={
            "test": a.test, "passed": a.passed, "magnitude": a.magnitude,
            "reliability": a.reliability, "p_value": a.p_value, "statistic": a.statistic,
        },
        n_affected=None,
        timestamp=_now(),
    )


def data_digest(y) -> str:
    """SHA-256 over the measurement vector as canonical float64 C-order bytes.

    Recorded in the bayes Step.params so the provenance digest changes whenever
    the data change (the shared load Step hashes only the row count). Hashes the
    values as provided (order preserved, NaN not dropped) so any edit is caught.
    """
    a = np.ascontiguousarray(np.asarray(y, dtype="<f8"))
    return hashlib.sha256(a.tobytes()).hexdigest()


@dataclass(frozen=True, repr=False)
class BayesNormalResult(QCResult):
    """Posterior of the Normal-Inverse-chi2 conjugate fit (immutable)."""

    n: int
    mun: float
    kn: float
    nun: float
    sn2: float
    prior_family: str
    seed: int
    draws: int
    cred_level: float = 0.95
    assumptions: list = field(default_factory=list)
    history: tuple = field(default_factory=tuple)

    def _title(self) -> str:
        return "Bayesian Normal Fit"

    def _summary_lines(self) -> list[str]:
        conf = round(self.cred_level * 100)
        mu = mu_marginal(self.mun, self.kn, self.nun, self.sn2)
        lo = float(mu.ppf((1.0 - self.cred_level) / 2.0))
        hi = float(mu.ppf((1.0 + self.cred_level) / 2.0))
        return [
            f"n = {self.n}   posterior mean of mu = {self.mun:.5g}",
            f"mu {conf}% credible interval = ({lo:.5g}, {hi:.5g})",
            f"prior family = {self.prior_family}",
        ]

    def _render_standalone(self, fig, kind, **kwargs) -> None:
        from . import plotting
        if kind is None:
            plotting.normal_panels(fig, self)
            return
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs) -> None:
        from . import plotting
        if kind in (None, "mu", "mean"):
            plotting.normal_mu_axes(ax, self)
        elif kind in ("sigma", "sd"):
            plotting.normal_sigma_axes(ax, self)
        else:
            raise ValueError(f"unknown normal view kind={kind!r}; use None, 'mu', or 'sigma'.")


def fit_normal(y, prior, *, seed: int, draws: int, cred_level: float = 0.95,
               base_history: tuple = ()) -> BayesNormalResult:
    """Fit the Normal-Inverse-chi2 conjugate model to ``y`` under ``prior``.

    ``seed`` and ``draws`` are recorded in provenance even though the posterior
    itself is closed-form, because downstream Monte Carlo push-through draws
    depend on them (the reproducibility contract, T5.2). If ``prior`` was built
    via :meth:`NormalPrior.from_result`, the parent history is prepended so a
    later edit to any parent step breaks this result's verify (T5.3).
    """
    n, ybar, s2 = suffstats(y)
    require_min_n(n)
    mun, kn, nun, sn2 = update(prior.mu0, prior.k0, prior.nu0, prior.s20, n, ybar, s2)

    checks = [
        prior_weight_check(prior.k0, n),
        prior_conflict_check(prior, n, ybar),
        small_sample_check(n),
    ]

    parent_hist = tuple(getattr(prior, "_parent_history", ()) or ())
    base = tuple(base_history) or parent_hist

    step = Step(
        operation="bayes.normal_fit",
        params={
            "prior": prior.to_params(),
            "data_sha256": data_digest(y),
            "seed": int(seed),
            "draws": int(draws),
            "grid": None,
            "tests": [],
            "cred_level": float(cred_level),
        },
        n_affected=n,
        timestamp=_now(),
    )
    history = base + (step,) + tuple(_assumption_step(a) for a in checks)
    return BayesNormalResult(
        n=n, mun=mun, kn=kn, nun=nun, sn2=sn2,
        prior_family=prior.to_params()["family"],
        seed=int(seed), draws=int(draws), cred_level=float(cred_level),
        assumptions=checks, history=history,
    )
