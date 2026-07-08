"""Bayesian attribute capability: proportion (Beta-Binomial) and rate
(Gamma-Poisson) surfaces (spec Algorithm B; BDA3 sec 2.4 and 2.6; Hoff sec 3.1-3.2).

Both surfaces are closed-form: the posterior, its credible interval, tail
probabilities, and the posterior-predictive pmf are all analytic, so there is no
Monte Carlo and no seed. The proportion surface answers "how large is the defect
fraction, and P(fraction <= a target)?"; the rate surface answers the same for a
count rate and screens the counts for overdispersion (guardrail G4) before the
Poisson model is trusted. Index definitions and the reuse of the classical
dispersion check keep these consistent with the attribute charts in mfgqc.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from mfgqc.assumptions import check_dispersion
from mfgqc._result import QCResult
from mfgqc.data import Step

from ._results import _assumption_step, _now, data_digest
from .conjugate import beta_update, betabinom_pmf, gamma_update, nbinom_predictive

_JEFFREYS_BETA = (0.5, 0.5)      # noninformative proportion prior
_JEFFREYS_GAMMA = (0.5, 0.0)     # noninformative Poisson-rate prior


def _coerce_binary(data, mapping: dict | None) -> tuple:
    """Reduce an attribute column to (n_fail, n_trials). Accepts bool, {0,1}, or
    an explicit label ``mapping``; anything else raises (spec Algorithm B)."""
    arr = np.asarray(data)
    if mapping is not None:
        try:
            out = np.array([int(mapping[v]) for v in arr.tolist()], dtype=int)
        except KeyError as exc:
            raise ValueError(f"mapping has no entry for label {exc}.") from None
    elif arr.dtype == bool:
        out = arr.astype(int)
    else:
        vals = np.asarray(arr, dtype=float)
        vals = vals[~np.isnan(vals)]
        if not set(np.unique(vals).tolist()) <= {0.0, 1.0}:
            raise ValueError(
                "proportion data must be bool, {0,1}, or use an explicit mapping=.")
        out = vals.astype(int)
    return int(out.sum()), int(out.size)


# --------------------------------------------------------------------------- #
# Proportion (Beta-Binomial)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class BayesProportionResult(QCResult):
    """Posterior for a defect proportion, Beta(a_post, b_post) (immutable,
    closed-form). Everything derived is analytic; there are no Monte Carlo draws."""

    n_trials: int
    n_fail: int
    a_post: float
    b_post: float
    prior_family: str
    cred_level: float
    max_proportion: float | None = None
    assumptions: list = field(default_factory=list)
    history: tuple = field(default_factory=tuple)

    @property
    def mean(self) -> float:
        return self.a_post / (self.a_post + self.b_post)

    def _posterior(self):
        return stats.beta(self.a_post, self.b_post)

    def interval(self, level: float | None = None) -> tuple:
        level = self.cred_level if level is None else level
        d = self._posterior()
        return float(d.ppf((1.0 - level) / 2.0)), float(d.ppf((1.0 + level) / 2.0))

    def prob(self, threshold: float, direction: str = ">=") -> float:
        """Exact tail probability of the proportion (closed-form, no MCSE)."""
        d = self._posterior()
        return float(d.sf(threshold)) if direction == ">=" else float(d.cdf(threshold))

    def prob_within_spec(self) -> float | None:
        """P(proportion <= max_proportion), or None if no target was given."""
        if self.max_proportion is None:
            return None
        return self.prob(self.max_proportion, "<=")

    def predictive_pmf(self, k: int, m: int) -> float:
        """Posterior-predictive P(k failures in m future trials), Beta-Binomial."""
        return betabinom_pmf(k, m, self.a_post, self.b_post)

    def _title(self) -> str:
        return "Bayesian Proportion Capability"

    def _summary_lines(self) -> list[str]:
        conf = round(self.cred_level * 100)
        lo, hi = self.interval()
        lines = [
            f"n = {self.n_trials}   failures = {self.n_fail}   "
            f"posterior mean p = {self.mean:.4g}",
            f"p {conf}% credible interval = ({lo:.4g}, {hi:.4g})",
            f"prior family = {self.prior_family}",
        ]
        if self.max_proportion is not None:
            lines.append(
                f"P(p <= {self.max_proportion:.3g}) = {self.prob_within_spec():.3g}")
        return lines


def proportion_capability(data=None, *, n_fail: int | None = None,
                          n_trials: int | None = None,
                          max_proportion: float | None = None, prior=None,
                          mapping: dict | None = None, cred_level: float = 0.95,
                          base_history: tuple = ()) -> BayesProportionResult:
    """Bayesian proportion capability from raw attribute data or aggregate counts.

    Provide either ``data`` (bool, {0,1}, or labels with ``mapping``) or both
    ``n_fail`` and ``n_trials``. The default prior is Jeffreys Beta(0.5, 0.5);
    pass a :class:`BetaPrior` to override. If ``max_proportion`` is given the
    report and :meth:`~BayesProportionResult.prob_within_spec` show
    P(proportion <= that target).
    """
    if data is not None:
        n_fail, n_trials = _coerce_binary(data, mapping)
    if n_fail is None or n_trials is None:
        raise ValueError("provide either data= or both n_fail= and n_trials=.")
    if n_trials < 1 or n_fail < 0 or n_fail > n_trials:
        raise ValueError(f"invalid counts: n_fail={n_fail}, n_trials={n_trials}.")

    a0, b0 = (prior.a, prior.b) if prior is not None else _JEFFREYS_BETA
    a_post, b_post = beta_update(a0, b0, n_fail, n_trials)

    step = Step(
        operation="bayes.proportion_capability",
        params={
            "prior": {"family": "beta", "hyperparams": {"a": float(a0), "b": float(b0)}},
            "n_fail": int(n_fail), "n_trials": int(n_trials),
            "cred_level": float(cred_level),
            "max_proportion": None if max_proportion is None else float(max_proportion),
        },
        n_affected=int(n_trials),
        timestamp=_now(),
    )
    return BayesProportionResult(
        n_trials=int(n_trials), n_fail=int(n_fail),
        a_post=float(a_post), b_post=float(b_post), prior_family="beta",
        cred_level=float(cred_level),
        max_proportion=None if max_proportion is None else float(max_proportion),
        assumptions=[], history=tuple(base_history) + (step,),
    )


# --------------------------------------------------------------------------- #
# Rate (Gamma-Poisson)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class BayesRateResult(QCResult):
    """Posterior for a count rate, Gamma(a_post, rate=b_post) (immutable,
    closed-form). Carries the G4 overdispersion check in ``assumptions``."""

    k: int
    sum_y: float
    sum_x: float
    a_post: float
    b_post: float
    prior_family: str
    cred_level: float
    max_rate: float | None = None
    assumptions: list = field(default_factory=list)
    history: tuple = field(default_factory=tuple)

    @property
    def mean(self) -> float:
        return self.a_post / self.b_post

    def _posterior(self):
        return stats.gamma(self.a_post, scale=1.0 / self.b_post)

    def interval(self, level: float | None = None) -> tuple:
        level = self.cred_level if level is None else level
        d = self._posterior()
        return float(d.ppf((1.0 - level) / 2.0)), float(d.ppf((1.0 + level) / 2.0))

    def prob(self, threshold: float, direction: str = ">=") -> float:
        """Exact tail probability of the rate (closed-form, no MCSE)."""
        d = self._posterior()
        return float(d.sf(threshold)) if direction == ">=" else float(d.cdf(threshold))

    def prob_within_spec(self) -> float | None:
        """P(rate <= max_rate), or None if no target was given."""
        if self.max_rate is None:
            return None
        return self.prob(self.max_rate, "<=")

    def predictive_pmf(self, k: int, exposure: float = 1.0) -> float:
        """Posterior-predictive P(k counts over ``exposure``), negative-binomial."""
        return float(nbinom_predictive(self.a_post, self.b_post, exposure).pmf(k))

    def _title(self) -> str:
        return "Bayesian Rate Capability"

    def _summary_lines(self) -> list[str]:
        conf = round(self.cred_level * 100)
        lo, hi = self.interval()
        lines = [
            f"k = {self.k}   total count = {self.sum_y:.4g}   "
            f"total exposure = {self.sum_x:.4g}",
            f"posterior mean rate = {self.mean:.4g}",
            f"rate {conf}% credible interval = ({lo:.4g}, {hi:.4g})",
            f"prior family = {self.prior_family}",
        ]
        if self.max_rate is not None:
            lines.append(f"P(rate <= {self.max_rate:.3g}) = {self.prob_within_spec():.3g}")
        return lines


def rate_capability(counts, exposures=None, *, max_rate: float | None = None,
                    prior=None, cred_level: float = 0.95,
                    base_history: tuple = ()) -> BayesRateResult:
    """Bayesian rate capability from per-observation ``counts`` and ``exposures``.

    ``exposures`` defaults to one unit per observation. The default prior is
    Jeffreys Gamma(0.5, rate=0); pass a :class:`GammaPrior` to override. The G4
    guardrail runs the classical chi-square dispersion test (family Poisson) and
    is attached to ``assumptions``: it warns when the counts are overdispersed
    relative to the Poisson model that the rate posterior assumes.
    """
    counts = np.asarray(counts, dtype=float)
    if exposures is None:
        exposures = np.ones(counts.size)
    else:
        exposures = np.asarray(exposures, dtype=float)
    if counts.size == 0 or counts.size != exposures.size:
        raise ValueError("counts and exposures must be non-empty and equal length.")

    sum_y = float(counts.sum())
    sum_x = float(exposures.sum())
    a0, b0 = (prior.a, prior.b) if prior is not None else _JEFFREYS_GAMMA
    a_post, b_post = gamma_update(a0, b0, sum_y, sum_x)

    checks = [check_dispersion(counts, exposures, family="poisson")]

    step = Step(
        operation="bayes.rate_capability",
        params={
            "prior": {"family": "gamma", "hyperparams": {"a": float(a0), "b": float(b0)}},
            "sum_y": sum_y, "sum_x": sum_x, "k": int(counts.size),
            "data_sha256": data_digest(counts),
            "cred_level": float(cred_level),
            "max_rate": None if max_rate is None else float(max_rate),
        },
        n_affected=int(counts.size),
        timestamp=_now(),
    )
    history = tuple(base_history) + (step,) + tuple(_assumption_step(a) for a in checks)
    return BayesRateResult(
        k=int(counts.size), sum_y=sum_y, sum_x=sum_x,
        a_post=float(a_post), b_post=float(b_post), prior_family="gamma",
        cred_level=float(cred_level),
        max_rate=None if max_rate is None else float(max_rate),
        assumptions=checks, history=history,
    )
