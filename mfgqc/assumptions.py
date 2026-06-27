"""Assumption checks: binary verdict + adjacent context (v2).

Philosophy: mfgQC surfaces assumptions and leaves the decision to the user -
"type hints, not decisions." So each check reports a BINARY ``passed`` driven by
the DIRECT test of the assumption at the conventional alpha (a single,
auditable threshold), and reports MAGNITUDE (practical impact) and RELIABILITY
(the test's resolving power at this n) as adjacent CONTEXT. The context never
flips the verdict - that is what prevents a derived quantity (e.g. a coincidental
small Cpk shift) from issuing a false all-clear on grossly non-normal data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

Reliability = str  # "ok" | "low_power" | "oversensitive"

ALPHA = 0.05
RELIABILITY_LOW_N = 20
RELIABILITY_HIGH_N = 5000
NDC_MIN = 5                  # AIAG adequacy rule
PROPORTION_MIN_EXPECTED = 5  # normal-approximation validity for proportions


@dataclass(frozen=True)
class AssumptionCheck:
    """Binary assumption verdict with practical-impact and reliability context.

    Attributes
    ----------
    name, test : str
        What was checked and the direct test used.
    statistic : float
        Test statistic.
    p_value : float or None
        P-value of the direct test where defined.
    passed : bool
        BINARY verdict from the direct test at alpha. The context fields below
        NEVER change this.
    magnitude : float or None
        Effect-size / practical-impact CONTEXT (e.g. variance ratio, Cpk impact).
    magnitude_label : str or None
        e.g. ``"variance ratio"``, ``"est. Cpk impact"``, ``"lag-1 autocorr"``.
    reliability : str
        ``"ok"`` / ``"low_power"`` / ``"oversensitive"`` - a fact about the test's
        resolving power at this n, not a judgment about the data.
    n : int
    recommendation : str or None
        Next step, populated when ``passed`` is False.
    """

    name: str
    test: str
    statistic: float
    p_value: float | None
    passed: bool
    magnitude: float | None
    magnitude_label: str | None
    reliability: Reliability
    n: int
    recommendation: str | None = None


def reliability(n: int, low: int = RELIABILITY_LOW_N, high: int | None = RELIABILITY_HIGH_N) -> Reliability:
    """n-aware caveat: small n underpowers the test, huge n over-detects."""
    if n < low:
        return "low_power"
    if high is not None and n > high:
        return "oversensitive"
    return "ok"


# --------------------------------------------------------------------------- #
# Anderson-Darling helpers
# --------------------------------------------------------------------------- #
def _anderson_darling_statistic(values: np.ndarray) -> float:
    xs = np.sort(np.asarray(values, dtype=float))
    n = xs.size
    sd = float(xs.std(ddof=1)) if n > 1 else 0.0
    if sd == 0:
        return float("nan")
    z = (xs - float(xs.mean())) / sd
    cdf = np.clip(stats.norm.cdf(z), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    s = np.sum((2 * i - 1) * (np.log(cdf) + np.log(1.0 - cdf[::-1])))
    return float(-n - s / n)


def _anderson_darling_pvalue(a2: float, n: int) -> float:
    if not np.isfinite(a2):
        return float("nan")
    a2_star = a2 * (1.0 + 0.75 / n + 2.25 / (n * n))
    if a2_star >= 13.0:
        return 0.0
    if a2_star >= 0.6:
        p = np.exp(1.2937 - 5.709 * a2_star + 0.0186 * a2_star**2)
    elif a2_star >= 0.34:
        p = np.exp(0.9177 - 4.279 * a2_star - 1.38 * a2_star**2)
    elif a2_star >= 0.2:
        p = 1.0 - np.exp(-8.318 + 42.796 * a2_star - 59.938 * a2_star**2)
    else:
        p = 1.0 - np.exp(-13.436 + 101.14 * a2_star - 223.73 * a2_star**2)
    return float(min(max(p, 0.0), 1.0))


# --------------------------------------------------------------------------- #
# 1. Normality
# --------------------------------------------------------------------------- #
_NORMALITY_ADVICE = {
    "capability": "for capability use a non-normal method (method='clements'/'johnson').",
    "hypothesis": "consider a non-parametric alternative.",
    "regression": ("consider a response transform, robust (HC) standard errors, or a "
                   "missing/nonlinear term; large-n inference is somewhat robust via the CLT."),
    "control": "consider a transform or a distribution-specific chart.",
    "generic": ("the appropriate remedy depends on the analysis (transform, non-parametric "
                "test, or a non-normal capability method)."),
}


def check_normality(values: np.ndarray, *, alpha: float = ALPHA,
                    cpk_impact: float | None = None,
                    context: str = "generic") -> AssumptionCheck:
    """Anderson-Darling normality. ``passed`` is the DIRECT AD test at alpha.

    ``cpk_impact`` (the relative change in the capability index between the normal
    and non-normal method) is recorded as CONTEXT when supplied; otherwise a
    skew-based distance is used as context. Context never changes ``passed``.

    ``context`` selects analysis-appropriate advice for the recommendation
    (``capability``/``hypothesis``/``regression``/``control``/``generic``) - the
    same failed test calls for different remedies in different analyses.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    a2 = _anderson_darling_statistic(x) if n >= 2 else float("nan")

    if not np.isfinite(a2):
        return AssumptionCheck("normality", "Anderson-Darling", float("nan"), None,
                               True, None, None, "low_power", n, None)

    p = _anderson_darling_pvalue(a2, n)
    passed = p >= alpha

    if cpk_impact is not None and np.isfinite(cpk_impact):
        magnitude, label = float(abs(cpk_impact)), "est. Cpk impact"
    else:
        magnitude, label = float(abs(stats.skew(x))), "skew"

    rec = None
    if not passed:
        advice = _NORMALITY_ADVICE.get(context, _NORMALITY_ADVICE["generic"])
        rec = f"Data are not normal (AD={a2:.3g}, p={p:.3g}); {advice}"
    return AssumptionCheck("normality", "Anderson-Darling", float(a2), float(p),
                           passed, magnitude, label, reliability(n, 20, 5000), n, rec)


# --------------------------------------------------------------------------- #
# 2. Independence (lag-1 autocorrelation)
# --------------------------------------------------------------------------- #
def check_independence(values: np.ndarray, *, alpha: float = ALPHA) -> AssumptionCheck:
    """Lag-1 autocorrelation test. ``passed`` from the test; |r| is context."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 4:
        return AssumptionCheck("independence", "lag-1 autocorrelation", float("nan"), None,
                               True, None, "lag-1 autocorr", "low_power", n, None)
    xc = x - x.mean()
    denom = float(np.sum(xc * xc))
    r1 = float(np.sum(xc[1:] * xc[:-1]) / denom) if denom > 0 else 0.0
    se = 1.0 / np.sqrt(n)
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(r1) / se)))
    passed = p >= alpha
    rec = None
    if not passed:
        rec = (f"Observations are autocorrelated (lag-1 r={r1:.2f}); control limits are "
               "unreliable - consider a time-series chart (e.g. EWMA).")
    return AssumptionCheck("independence", "lag-1 autocorrelation", r1, p,
                           passed, abs(r1), "lag-1 autocorr",
                           "low_power" if n < 30 else "ok", n, rec)


# --------------------------------------------------------------------------- #
# 3. Attribute over/under-dispersion
# --------------------------------------------------------------------------- #
def check_dispersion(counts: np.ndarray, sizes: np.ndarray, *, family: str,
                     alpha: float = ALPHA) -> AssumptionCheck:
    """Chi-square dispersion test vs binomial/Poisson. ``passed`` from the test;
    the dispersion ratio is context."""
    counts = np.asarray(counts, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    k = counts.size
    if k < 2:
        return AssumptionCheck("dispersion", "chi-square dispersion", float("nan"), None,
                               True, None, "dispersion ratio", "low_power", k, None)
    if family == "binomial":
        pbar = counts.sum() / sizes.sum()
        expected = sizes * pbar
        var = sizes * pbar * (1.0 - pbar)
    else:  # poisson
        lam = counts.sum() / sizes.sum()
        expected = sizes * lam
        var = expected.copy()
    dof = k - 1
    var = np.where(var <= 0, np.nan, var)
    chi2 = float(np.nansum((counts - expected) ** 2 / var))
    ratio = chi2 / dof if dof > 0 else float("nan")
    p = float(stats.chi2.sf(chi2, dof)) if dof > 0 else None
    passed = bool(p is not None and p >= alpha)
    rec = None
    if not passed:
        kind = "Over" if ratio > 1 else "Under"
        model = "binomial" if family == "binomial" else "Poisson"
        rec = (f"{kind}dispersion vs the {model} model (ratio {ratio:.2f}); standard limits "
               "are unreliable - consider a Laney p'/u' or dispersion-adjusted chart.")
    return AssumptionCheck("dispersion", "chi-square dispersion", ratio, p,
                           passed, ratio, "dispersion ratio",
                           "low_power" if k < 20 else "ok", k, rec)


# --------------------------------------------------------------------------- #
# 4. Homogeneity of variance across groups
# --------------------------------------------------------------------------- #
def check_homogeneity(groups: list[np.ndarray], *, alpha: float = ALPHA) -> AssumptionCheck:
    """Levene's test for equal variance. ``passed`` from Levene; variance ratio is context."""
    clean = [np.asarray(g, dtype=float) for g in groups]
    clean = [g[~np.isnan(g)] for g in clean]
    usable = [g for g in clean if g.size > 1]
    n_total = int(sum(g.size for g in usable))
    if len(usable) < 2:
        return AssumptionCheck("homogeneity_of_variance", "Levene", float("nan"), None,
                               True, None, "variance ratio", "low_power", n_total, None)
    variances = np.array([np.var(g, ddof=1) for g in usable])
    vmin = float(np.min(variances))
    ratio = float(np.max(variances) / vmin) if vmin > 0 else float("inf")
    stat, p = stats.levene(*usable, center="median")
    passed = float(p) >= alpha
    n_min = min(g.size for g in usable)
    rel = "low_power" if n_min < 10 else ("oversensitive" if n_total > RELIABILITY_HIGH_N else "ok")
    rec = None
    if not passed:
        rec = (f"Group variances differ (variance ratio {ratio:.3g}, Levene p={float(p):.3g}); "
               "for a two-sample t-test use Welch's; for ANOVA use Welch's ANOVA.")
    return AssumptionCheck("homogeneity_of_variance", "Levene", float(stat), float(p),
                           passed, ratio, "variance ratio", rel, n_total, rec)
