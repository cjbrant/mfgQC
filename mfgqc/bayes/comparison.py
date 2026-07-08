"""Bayesian comparison of two fitted capability analyses (spec Algorithm D; Hoff sec 8.1).

Two independent posteriors are drawn under a common seed split via ``rng.spawn(2)``.
The two child streams are assigned to the two results by a canonical key (the
provenance digest), so swapping the arguments reuses the same draws and every
probability complements exactly (T1.9).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from mfgqc._result import QCResult
from mfgqc.data import Step

from .capability import _index_draws


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _draw_posterior(result, rng: np.random.Generator, draws: int) -> dict:
    """Draw (mu, sigma, ppk) from a fitted capability result's overall posterior,
    using the normative call order (chisquare then normal)."""
    sig2 = result.nun * result.sn2 / rng.chisquare(result.nun, draws)
    mu = rng.normal(result.mun, np.sqrt(sig2 / result.kn))
    sigma = np.sqrt(sig2)
    _, _, _, ppk = _index_draws(mu, sigma, result.spec)
    return {"mu": mu, "sigma": sigma, "ppk": ppk}


def _summary(arr, level: float) -> tuple:
    lo_p, hi_p = (1.0 - level) / 2.0, (1.0 + level) / 2.0
    med, lo, hi = np.quantile(arr, [0.5, lo_p, hi_p])
    return float(med), float(lo), float(hi)


@dataclass(frozen=True, repr=False)
class ComparisonResult(QCResult):
    """Posterior comparison of two capability analyses (immutable). Probabilities
    are stated as P(B better than A): mean larger, sd smaller, Ppk larger."""

    label_a: str
    label_b: str
    seed: int
    draws: int
    cred_level: float
    prob_mean_gt: float
    prob_sd_lt: float
    prob_ppk_gt: float
    delta_mean: tuple
    delta_ppk: tuple
    assumptions: list = field(default_factory=list)
    history: tuple = field(default_factory=tuple)

    def _title(self) -> str:
        return f"Bayesian Comparison ({self.label_b} vs {self.label_a})"

    def _summary_lines(self) -> list[str]:
        conf = round(self.cred_level * 100)
        dm, dml, dmh = self.delta_mean
        dp, dpl, dph = self.delta_ppk
        return [
            f"P({self.label_b} mean > {self.label_a} mean) = {self.prob_mean_gt:.3g}",
            f"P({self.label_b} sd < {self.label_a} sd)     = {self.prob_sd_lt:.3g}",
            f"P({self.label_b} Ppk > {self.label_a} Ppk)   = {self.prob_ppk_gt:.3g}",
            f"delta mean (B - A) = {dm:.4g}   {conf}% ({dml:.4g}, {dmh:.4g})",
            f"delta Ppk  (B - A) = {dp:.3g}   {conf}% ({dpl:.3g}, {dph:.3g})",
        ]


def compare(a, b, *, seed: int, draws: int = 100_000, cred_level: float = 0.95,
            labels: tuple = ("A", "B")) -> ComparisonResult:
    """Compare two fitted capability analyses. Probabilities are P(B better than A).

    The child seed assigned to each result is chosen by its provenance digest, so
    ``compare(a, b)`` and ``compare(b, a)`` reuse identical draws and every
    probability complements exactly.
    """
    da, db = a.provenance_digest(), b.provenance_digest()
    ss = np.random.default_rng(seed).spawn(2)
    rng_a, rng_b = (ss[0], ss[1]) if da <= db else (ss[1], ss[0])

    A = _draw_posterior(a, rng_a, draws)
    B = _draw_posterior(b, rng_b, draws)

    prob_mean_gt = float((B["mu"] > A["mu"]).mean())
    prob_sd_lt = float((B["sigma"] < A["sigma"]).mean())
    prob_ppk_gt = float((B["ppk"] > A["ppk"]).mean())

    step = Step(
        operation="bayes.compare",
        params={
            "parent_a": da, "parent_b": db,
            "seed": int(seed), "draws": int(draws), "cred_level": float(cred_level),
        },
        n_affected=None,
        timestamp=_now(),
    )
    return ComparisonResult(
        label_a=labels[0], label_b=labels[1],
        seed=int(seed), draws=int(draws), cred_level=float(cred_level),
        prob_mean_gt=prob_mean_gt, prob_sd_lt=prob_sd_lt, prob_ppk_gt=prob_ppk_gt,
        delta_mean=_summary(B["mu"] - A["mu"], cred_level),
        delta_ppk=_summary(B["ppk"] - A["ppk"], cred_level),
        assumptions=[], history=(step,),
    )
