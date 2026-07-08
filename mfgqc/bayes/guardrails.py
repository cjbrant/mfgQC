"""Bayesian guardrails (plan D5).

Each guardrail is an AssumptionCheck the analysis appends to its result, or a
raise for a contradictory request. They warn and recommend; they never switch
methods or mutate inputs (House Rule 5). This matches the existing idiom
(assumptions.py) - there is no warnings.warn and no Warning subclass.
"""
from __future__ import annotations

import math

from scipy import stats

from mfgqc.assumptions import AssumptionCheck
from mfgqc.assumptions import reliability as _reliability

_SMALL_SAMPLE_MIN = 8


def require_min_n(n: int, analysis: str = "bayes.normal_fit") -> None:
    """G3 (raise): the Normal fit needs n>=2 to form the sample variance."""
    if n < 2:
        raise ValueError(
            f"{analysis} needs n>=2 to form the sample variance; got n={n}. "
            f"Collect more data.")


def prior_weight_check(k0: float, n: int) -> AssumptionCheck:
    """G1: prior-weight disclosure. w = k0/(k0+n) is always reported; the
    recommendation fires only when the prior contributes more than half."""
    w = k0 / (k0 + n)
    passed = w <= 0.5
    return AssumptionCheck(
        name="prior_weight", test="prior weight w = k0/(k0+n)",
        statistic=float(w), p_value=None, passed=bool(passed),
        magnitude=float(w), magnitude_label="prior weight",
        reliability="ok", n=int(n),
        recommendation=None if passed else (
            f"The prior contributes {w:.0%} of the posterior; the report is prior "
            f"dominated. Widen the prior or collect more data."),
    )


def prior_conflict_check(prior, n: int, ybar: float) -> AssumptionCheck:
    """G2: prior-data conflict. Standardize the gap between the prior mean and the
    data mean by the prior-predictive scale, ybar ~ t_{nu0}(mu0, s20*(1/k0+1/n));
    warn when it exceeds five predictive SDs."""
    scale = math.sqrt(prior.s20 * (1.0 / prior.k0 + 1.0 / n))
    z = abs(ybar - prior.mu0) / scale
    passed = z < 5.0
    return AssumptionCheck(
        name="prior_data_conflict", test="prior predictive t on ybar",
        statistic=float(z), p_value=float(2.0 * stats.t.sf(z, prior.nu0)),
        passed=bool(passed), magnitude=float(z), magnitude_label="predictive SDs",
        reliability="ok", n=int(n),
        recommendation=None if passed else (
            f"The data mean is {z:.1f} prior-predictive SDs from the prior mean; "
            f"prior and data disagree. Reconsider the prior."),
    )


def small_sample_check(n: int, threshold: int = _SMALL_SAMPLE_MIN) -> AssumptionCheck:
    """G3 (warn): a small sample gives wide, prior-sensitive posteriors."""
    passed = n >= threshold
    return AssumptionCheck(
        name="small_sample", test=f"n >= {threshold}",
        statistic=float(n), p_value=None, passed=bool(passed),
        magnitude=float(n), magnitude_label="n",
        reliability=_reliability(n), n=int(n),
        recommendation=None if passed else (
            f"Only n={n}; posterior summaries are wide and prior sensitive. "
            f"Collect more data for a stable estimate."),
    )
