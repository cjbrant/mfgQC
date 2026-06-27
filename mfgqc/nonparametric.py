"""Remaining non-parametrics in the existing router idiom.

- ``test_medians``: Mood's median test for k samples - a location alternative to
  Kruskal-Wallis, more robust to outliers, less powerful (the trade-off is
  surfaced).
- ``test_repeated``: the blocked / repeated-measures case. Routes between
  repeated-measures ANOVA (with a Mauchly sphericity check surfaced) and the
  Friedman test when normality or sphericity fails, exactly as ``test_anova``
  routes to Kruskal-Wallis. Both report the route and why.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from . import assumptions as _assume
from .data import Step
from .hypothesis import HypothesisResult


def _now() -> datetime:
    return datetime.now(timezone.utc)


def test_medians(*groups, method=None, labels=None, alpha: float = 0.05) -> HypothesisResult:
    """Mood's median test for k>=2 samples (common-median null)."""
    arrs = [np.asarray(g, dtype=float) for g in groups]
    arrs = [a[~np.isnan(a)] for a in arrs]
    if len(arrs) < 2:
        raise ValueError("test_medians needs at least 2 groups.")
    labels = list(labels) if labels else [f"g{i+1}" for i in range(len(arrs))]
    stat, p, grand_median, _table = stats.median_test(*arrs)
    k = len(arrs)
    step = Step(operation="test_medians", params={"groups": k, "grand_median": float(grand_median)},
                n_affected=int(sum(a.size for a in arrs)), timestamp=_now())
    return HypothesisResult(
        h0="all groups share a common median", h1="at least one median differs",
        requested="medians", test_used="mood_median", statistic=float(stat),
        p_value=float(p), df=float(k - 1), effect_size=None, effect_name=None, ci=None,
        alternative="two-sided", routed=False,
        recommendation=("Mood's median test is robust to outliers but LESS powerful than "
                        "Kruskal-Wallis; prefer Kruskal-Wallis when a location-shift model holds."),
        selection_reason=None, _groups=tuple(arrs), _labels=tuple(labels),
        assumptions=[], history=(step,))


def _orthonormal_contrasts(k: int) -> np.ndarray:
    """A (k-1) x k orthonormal contrast matrix (rows orthogonal to 1)."""
    H = np.eye(k) - np.ones((k, k)) / k
    # take the first k-1 left singular vectors spanning the contrast space
    u, s, _ = np.linalg.svd(H)
    return u[:, : k - 1].T


def _mauchly(wide: np.ndarray):
    """Mauchly's test of sphericity on a subjects x conditions matrix.
    Returns (W, chi2, df, p)."""
    n, k = wide.shape
    M = _orthonormal_contrasts(k)
    Y = wide @ M.T                       # n x (k-1) contrast scores
    S = np.cov(Y, rowvar=False, ddof=1)
    S = np.atleast_2d(S)
    eig = np.linalg.eigvalsh(S)
    eig = eig[eig > 1e-12]
    p = k - 1
    if eig.size < p:
        return float("nan"), float("nan"), 0, float("nan")
    W = float(np.prod(eig) / (np.mean(eig) ** p))
    dfree = p * (p + 1) // 2 - 1
    f = 1 - (2 * p ** 2 + p + 2) / (6 * p * (n - 1))
    chi2 = -(n - 1) * f * np.log(W) if W > 0 else float("nan")
    pval = float(stats.chi2.sf(chi2, dfree)) if dfree > 0 and np.isfinite(chi2) else float("nan")
    return W, float(chi2), dfree, pval


def test_repeated(data: pd.DataFrame, *, subject: str, within: str, response: str,
                  method=None, alpha: float = 0.05) -> HypothesisResult:
    """Repeated-measures / blocked test. Routes RM-ANOVA <-> Friedman.

    Parameters
    ----------
    data : DataFrame
        Long form: one row per (subject, within-level) with the response.
    subject, within, response : str
        Column names for the blocking subject, the within factor, and the response.
    method : str or None
        Force ``"rm_anova"`` or ``"friedman"``; None routes by the assumptions.
    """
    for c in (subject, within, response):
        if c not in data.columns:
            raise ValueError(f"column {c!r} not found.")
    wide = data.pivot_table(index=subject, columns=within, values=response, aggfunc="mean").dropna()
    levels = list(wide.columns)
    mat = wide.to_numpy(dtype=float)
    n, k = mat.shape
    if k < 3:
        raise ValueError("test_repeated needs at least 3 within-conditions.")
    groups = tuple(mat[:, j] for j in range(k))

    # assumptions: normality of the within-subject differences, and sphericity
    resid = mat - mat.mean(axis=1, keepdims=True) - mat.mean(axis=0, keepdims=True) + mat.mean()
    norm = _assume.check_normality(resid.ravel(), context="hypothesis")
    W, m_chi2, m_df, m_p = _mauchly(mat)
    sph_passed = bool(np.isnan(m_p) or m_p >= alpha)
    sphericity = _assume.AssumptionCheck("sphericity", "Mauchly", float(m_chi2),
                                         None if np.isnan(m_p) else float(m_p),
                                         sph_passed, float(W), "Mauchly W",
                                         "low_power" if n < 10 else "ok", n,
                                         None if sph_passed else
                                         (f"Sphericity is violated (Mauchly W={W:.3g}, p={m_p:.3g}); "
                                          "use a Greenhouse-Geisser correction or the Friedman test."))
    checks = [norm, sphericity]

    routes = {None: "rm_anova", "rm_anova": "rm_anova", "friedman": "friedman"}
    if method not in routes:
        raise ValueError("method must be None, 'rm_anova', or 'friedman'.")
    routed = method is None
    if routed:
        run = "rm_anova" if (norm.passed and sph_passed) else "friedman"
        if not norm.passed:
            reason = "residuals are not normal: Friedman (non-parametric)"
        elif not sph_passed:
            reason = "sphericity violated: Friedman (avoids the sphericity assumption)"
        else:
            reason = "normal and spherical: repeated-measures ANOVA"
    else:
        run = method
        reason = f"forced {method}"

    if run == "rm_anova":
        from statsmodels.stats.anova import AnovaRM
        aov = AnovaRM(data, depvar=response, subject=subject, within=[within]).fit()
        row = aov.anova_table.iloc[0]
        stat = float(row["F Value"])
        p = float(row["Pr > F"])
        df = float(row["Num DF"])
        test_used = "rm_anova"
        h1 = "at least one condition mean differs"
    else:
        stat, p = stats.friedmanchisquare(*groups)
        stat, p = float(stat), float(p)
        df = float(k - 1)
        test_used = "friedman"
        h1 = "at least one condition distribution differs"

    rec = None if routed else f"forced {method}; routing would have chosen {('rm_anova' if (norm.passed and sph_passed) else 'friedman')}."
    step = Step(operation="test_repeated", params={"within": within, "test_used": test_used,
                                                   "routed": routed}, n_affected=n, timestamp=_now())
    return HypothesisResult(
        h0="all repeated conditions are equivalent", h1=h1, requested="repeated",
        test_used=test_used, statistic=stat, p_value=p, df=df, effect_size=None,
        effect_name=None, ci=None, alternative="two-sided", routed=routed,
        recommendation=rec, selection_reason=reason if routed else None,
        _groups=groups, _labels=tuple(str(x) for x in levels), assumptions=checks,
        history=(step,))
