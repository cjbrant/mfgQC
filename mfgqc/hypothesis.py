"""Hypothesis testing with assumption-driven routing.

Unlike the other modules, here the assumption check is the *routing logic*: it
determines which test is correct. By DEFAULT mfgQC checks the assumptions (normality
per group via Anderson-Darling, equal variance via Levene) FIRST and SELECTS the
appropriate test from the outcomes - pooled t / Welch's t / Mann-Whitney (and the
ANOVA / variance analogues). The result reports which test was selected and the
assumption outcomes that drove the choice. A user may force a specific test with
``method=`` (e.g. ``method='pooled'``); forcing still surfaces the assumption
checks so an overridden recommendation is visible. The provenance records the
selection.

Core statistics delegate to scipy so they match the reference exactly; mfgQC adds
the assumption routing, effect sizes, confidence intervals and the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import numpy as np
from scipy import stats

from . import assumptions as _assume
from ._result import QCResult
from .assumptions import AssumptionCheck
from .data import Step

_ALT = ("two-sided", "less", "greater")


@dataclass(frozen=True, repr=False)
class HypothesisResult(QCResult):
    """Result of a hypothesis test (immutable)."""

    h0: str
    h1: str
    requested: str
    test_used: str
    statistic: float
    p_value: float
    df: float | None
    effect_size: float | None
    effect_name: str | None
    ci: tuple[float, float] | None
    alternative: str
    routed: bool
    recommendation: str | None
    selection_reason: str | None = None
    _groups: tuple = field(repr=False, default=())
    _labels: tuple = field(repr=False, default=())
    _target: float | None = field(repr=False, default=None)
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def posthoc(self, method=None, control=None):
        """Routed pairwise multiple comparisons after a k-sample omnibus test.
        Routes by this result's assumptions and test route. See
        :func:`mfgqc.posthoc.compute`."""
        if len(self._groups) < 3 and control is None:
            raise ValueError("posthoc needs a k-sample (>=3) ANOVA-family result.")
        from .posthoc import compute
        labels = self._labels or tuple(f"g{i+1}" for i in range(len(self._groups)))
        return compute(self._groups, labels, self.assumptions, self.test_used,
                       method=method, control=control, base_history=self.history)

    def _title(self) -> str:
        return f"Hypothesis Test: {self.test_used}"

    def _summary_lines(self) -> list[str]:
        lines = [f"H0: {self.h0}", f"H1: {self.h1}  (alternative={self.alternative})", ""]
        if self.selection_reason:
            lines.append(f"selected {self.test_used}: {self.selection_reason}")
        elif self.requested != self.test_used:
            lines.append(f"requested {self.requested}; ran {self.test_used}")
        df = "" if self.df is None else f"  df={self.df:.4g}"
        lines.append(f"{self.test_used}: statistic={self.statistic:.4g}{df}  p={self.p_value:.4g}")
        if self.effect_size is not None:
            lines.append(f"effect size ({self.effect_name}) = {self.effect_size:.4g}")
        if self.ci is not None:
            lines.append(f"95% CI = ({self.ci[0]:.5g}, {self.ci[1]:.5g})")
        decision = "reject H0" if self.p_value < 0.05 else "fail to reject H0"
        lines.append(f"decision at alpha=0.05: {decision}")
        return lines

    def summary(self) -> dict:
        """Flat, JSON-serializable summary of the test outcome."""
        return {
            "test_used": self.test_used,
            "requested": self.requested,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "df": self.df,
            "effect_size": self.effect_size,
            "effect_name": self.effect_name,
            "ci_lower": None if self.ci is None else float(self.ci[0]),
            "ci_upper": None if self.ci is None else float(self.ci[1]),
            "alternative": self.alternative,
            "reject_h0": bool(self.p_value < 0.05),
        }

    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        self._render_axes(ax, kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        from . import plotting
        plotting.hypothesis_plot(ax, self)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _assumption_step(a: AssumptionCheck) -> Step:
    return Step(
        operation=f"assumption:{a.name}",
        params={"test": a.test, "passed": a.passed, "magnitude": a.magnitude,
                "reliability": a.reliability, "p_value": a.p_value, "statistic": a.statistic},
        n_affected=None, timestamp=_now(),
    )


def _history(base, op, params, checks):
    step = Step(operation=op, params=params, n_affected=None, timestamp=_now())
    return tuple(base) + (step,) + tuple(_assumption_step(a) for a in checks)


def _clean(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]


def _neutral(checks):
    """Strip the assumption checks' domain-specific recommendations.

    The normality/homogeneity helpers carry capability/ANOVA-flavored advice that
    is wrong in a hypothesis-test context. Here the actionable guidance is the
    result-level routing recommendation, so the per-check text is cleared (the
    pass/fail severity is kept).
    """
    return [replace(c, recommendation=None) for c in checks]


def _cohens_d_pooled(a, b) -> float:
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    sp = np.sqrt(sp2)
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else float("nan")


# --------------------------------------------------------------------------- #
# 1. One-sample mean vs target
# --------------------------------------------------------------------------- #
def test_mean(values, target, *, alternative="two-sided", auto=False, alpha=0.05, base_history=()):
    """One-sample t-test of the mean vs ``target``; Wilcoxon fallback if non-normal."""
    if alternative not in _ALT:
        raise ValueError(f"alternative must be one of {_ALT}.")
    x = _clean(values)
    checks = [_assume.check_normality(x)]
    appropriate = "t" if checks[0].passed else "wilcoxon"
    run = "wilcoxon" if (auto and appropriate == "wilcoxon") else "t"
    routed = run != "t"

    if run == "t":
        res = stats.ttest_1samp(x, target, alternative=alternative)
        statistic, p, df = float(res.statistic), float(res.pvalue), float(len(x) - 1)
        ci = res.confidence_interval(confidence_level=1 - alpha)
        ci = (float(ci.low), float(ci.high))
        effect = float((np.mean(x) - target) / np.std(x, ddof=1))
        effect_name, test_used = "Cohen's d", "one-sample t"
    else:
        res = stats.wilcoxon(x - target, alternative=alternative)
        statistic, p, df, ci = float(res.statistic), float(res.pvalue), None, None
        effect, effect_name, test_used = None, None, "Wilcoxon signed-rank"

    recommendation = None
    if appropriate == "wilcoxon" and run == "t":
        recommendation = (f"Data fail normality (Anderson-Darling p={checks[0].p_value:.3g}); "
                          "the Wilcoxon signed-rank test is more appropriate. Rerun with auto=True.")
    h0 = f"mean = {target}"
    h1 = {"two-sided": f"mean != {target}", "less": f"mean < {target}", "greater": f"mean > {target}"}[alternative]
    hist = _history(base_history, "test_mean",
                    {"target": target, "alternative": alternative, "test_used": test_used,
                     "p_value": p, "auto": auto}, checks)
    return HypothesisResult(
        h0=h0, h1=h1, requested="one-sample t", test_used=test_used,
        statistic=statistic, p_value=p, df=df, effect_size=effect, effect_name=effect_name,
        ci=ci, alternative=alternative, routed=routed, recommendation=recommendation,
        _groups=(x,), _labels=("sample",), _target=float(target),
        assumptions=_neutral(checks), history=hist,
    )


# --------------------------------------------------------------------------- #
# 2. Two independent samples
# --------------------------------------------------------------------------- #
_MEANS_METHODS = {"pooled": "student", "student": "student", "welch": "welch",
                  "mannwhitney": "mannwhitney", "mw": "mannwhitney"}


def test_means(a, b, *, alternative="two-sided", method=None, test=None, auto=None,
               alpha=0.05, labels=("A", "B"), base_history=()):
    """Two-sample comparison of means - ROUTES by default.

    Checks normality (Anderson-Darling per group) and equal variance (Levene),
    then SELECTS the test: both normal + equal variance -> pooled t; both normal +
    unequal variance -> Welch's t; either non-normal -> Mann-Whitney U. The result
    reports the choice and why. Force a test with ``method=`` (``'pooled'``,
    ``'welch'``, ``'mannwhitney'``); forcing still surfaces the assumption checks.
    """
    if alternative not in _ALT:
        raise ValueError(f"alternative must be one of {_ALT}.")
    forced_in = method if method is not None else test
    forced = None
    if forced_in is not None:
        forced = _MEANS_METHODS.get(forced_in)
        if forced is None:
            raise ValueError("method must be 'pooled', 'welch', or 'mannwhitney'.")
    a, b = _clean(a), _clean(b)
    ca, cb = _assume.check_normality(a), _assume.check_normality(b)
    cv = _assume.check_homogeneity([a, b])
    checks = [ca, cb, cv]
    normal = ca.passed and cb.passed

    if not normal:
        appropriate = "mannwhitney"
    elif not cv.passed:
        appropriate = "welch"
    else:
        appropriate = "student"
    run = forced if forced is not None else appropriate    # ROUTE by default
    routed = forced is None

    na, nb = len(a), len(b)
    if run in ("student", "welch"):
        equal_var = run == "student"
        res = stats.ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
        statistic, p = float(res.statistic), float(res.pvalue)
        df = float(na + nb - 2) if equal_var else float(res.df)
        ci = res.confidence_interval(confidence_level=1 - alpha)
        ci = (float(ci.low), float(ci.high))
        effect, effect_name = _cohens_d_pooled(a, b), "Cohen's d"
        test_used = "Student's t (pooled)" if equal_var else "Welch's t"
    else:  # mannwhitney
        res = stats.mannwhitneyu(a, b, alternative=alternative)
        statistic, p, df, ci = float(res.statistic), float(res.pvalue), None, None
        effect = float(1 - 2 * res.statistic / (na * nb))
        effect_name, test_used = "rank-biserial", "Mann-Whitney U"

    reasons = {
        "mannwhitney": f"normality failed (AD p={min(ca.p_value, cb.p_value):.3g}) "
                       "-> non-parametric Mann-Whitney U",
        "welch": f"both groups ~normal but unequal variance (Levene p={cv.p_value:.3g}) -> Welch's t",
        "student": "both groups ~normal with equal variance -> pooled Student's t",
    }
    selection_reason = reasons[run] if routed else None
    recommendation = None
    if forced is not None and forced != appropriate:
        recommendation = (f"You forced {test_used}, but the assumption checks indicate "
                          f"{reasons[appropriate]} would be more appropriate.")
    h1 = {"two-sided": f"mean({labels[0]}) != mean({labels[1]})",
          "less": f"mean({labels[0]}) < mean({labels[1]})",
          "greater": f"mean({labels[0]}) > mean({labels[1]})"}[alternative]
    hist = _history(base_history, "test_means",
                    {"alternative": alternative, "test_used": test_used, "appropriate": appropriate,
                     "p_value": p, "routed": routed, "method": forced}, checks)
    return HypothesisResult(
        h0=f"mean({labels[0]}) = mean({labels[1]})", h1=h1, requested="two-sample comparison",
        test_used=test_used, statistic=statistic, p_value=p, df=df,
        effect_size=effect, effect_name=effect_name, ci=ci, alternative=alternative,
        routed=routed, recommendation=recommendation, selection_reason=selection_reason,
        _groups=(a, b), _labels=tuple(labels), assumptions=_neutral(checks), history=hist,
    )


# --------------------------------------------------------------------------- #
# 3. Paired samples
# --------------------------------------------------------------------------- #
def test_paired(before, after, *, alternative="two-sided", auto=False, alpha=0.05,
                base_history=()):
    """Paired comparison; paired t with Wilcoxon-signed-rank fallback on the differences."""
    if alternative not in _ALT:
        raise ValueError(f"alternative must be one of {_ALT}.")
    before, after = _clean(before), _clean(after)
    if len(before) != len(after):
        raise ValueError("paired test requires equal-length before/after arrays.")
    diff = before - after
    checks = [_assume.check_normality(diff)]
    appropriate = "t" if checks[0].passed else "wilcoxon"
    run = "wilcoxon" if (auto and appropriate == "wilcoxon") else "t"
    routed = run != "t"

    if run == "t":
        res = stats.ttest_rel(before, after, alternative=alternative)
        statistic, p, df = float(res.statistic), float(res.pvalue), float(len(diff) - 1)
        rci = stats.ttest_1samp(diff, 0.0, alternative=alternative).confidence_interval(1 - alpha)
        ci = (float(rci.low), float(rci.high))
        effect = float(np.mean(diff) / np.std(diff, ddof=1))
        effect_name, test_used = "Cohen's d", "paired t"
    else:
        res = stats.wilcoxon(diff, alternative=alternative)
        statistic, p, df, ci = float(res.statistic), float(res.pvalue), None, None
        effect, effect_name, test_used = None, None, "Wilcoxon signed-rank"

    recommendation = None
    if appropriate == "wilcoxon" and run == "t":
        recommendation = (f"Differences fail normality (p={checks[0].p_value:.3g}); Wilcoxon "
                          "signed-rank is more appropriate. Rerun with auto=True.")
    hist = _history(base_history, "test_paired",
                    {"alternative": alternative, "test_used": test_used, "p_value": p, "auto": auto},
                    checks)
    return HypothesisResult(
        h0="mean(before - after) = 0", h1="mean difference != 0" if alternative == "two-sided"
        else f"mean difference {'<' if alternative == 'less' else '>'} 0",
        requested="paired t", test_used=test_used, statistic=statistic, p_value=p, df=df,
        effect_size=effect, effect_name=effect_name, ci=ci, alternative=alternative,
        routed=routed, recommendation=recommendation,
        _groups=(before, after), _labels=("before", "after"), assumptions=_neutral(checks), history=hist,
    )


# --------------------------------------------------------------------------- #
# 4. k-sample means (ANOVA)
# --------------------------------------------------------------------------- #
def _welch_anova(groups):
    k = len(groups)
    n = np.array([len(g) for g in groups], dtype=float)
    m = np.array([np.mean(g) for g in groups])
    v = np.array([np.var(g, ddof=1) for g in groups])
    w = n / v
    sw = w.sum()
    xbar = np.sum(w * m) / sw
    num = np.sum(w * (m - xbar) ** 2) / (k - 1)
    lam = np.sum((1 - w / sw) ** 2 / (n - 1))
    denom = 1 + (2 * (k - 2) / (k ** 2 - 1)) * lam
    F = num / denom
    df1 = k - 1
    df2 = (k ** 2 - 1) / (3 * lam)
    p = float(stats.f.sf(F, df1, df2))
    return float(F), float(df1), float(df2), p


_ANOVA_METHODS = {"anova": "anova", "classic": "anova", "f": "anova",
                  "welch": "welch", "kruskal": "kruskal"}


def test_anova(*groups, method=None, auto=None, alpha=0.05, labels=None, base_history=()):
    """One-way ANOVA - ROUTES by default (classic F / Welch's ANOVA / Kruskal-Wallis).

    All groups ~normal + equal variance -> classic one-way ANOVA; ~normal + unequal
    variance -> Welch's ANOVA; any non-normal -> Kruskal-Wallis. Force with
    ``method=`` ('anova', 'welch', 'kruskal')."""
    groups = [_clean(g) for g in groups]
    if len(groups) < 2:
        raise ValueError("ANOVA requires at least two groups.")
    forced = None
    if method is not None:
        forced = _ANOVA_METHODS.get(method)
        if forced is None:
            raise ValueError("method must be 'anova', 'welch', or 'kruskal'.")
    labels = tuple(labels) if labels is not None else tuple(f"g{i+1}" for i in range(len(groups)))
    normal_checks = [_assume.check_normality(g) for g in groups]
    cv = _assume.check_homogeneity(groups)
    all_normal = all(c.passed for c in normal_checks)
    # one combined normality check summary (worst case) plus homogeneity
    worst_norm = min(normal_checks, key=lambda c: (c.passed, c.p_value if c.p_value is not None else 1))
    checks = [worst_norm, cv]

    if not all_normal:
        appropriate = "kruskal"
    elif not cv.passed:
        appropriate = "welch"
    else:
        appropriate = "anova"
    run = forced if forced is not None else appropriate    # ROUTE by default
    routed = forced is None

    grand = np.concatenate(groups)
    if run == "anova":
        res = stats.f_oneway(*groups)
        statistic, p = float(res.statistic), float(res.pvalue)
        k = len(groups)
        df = float(k - 1)
        ss_between = sum(len(g) * (np.mean(g) - np.mean(grand)) ** 2 for g in groups)
        ss_total = float(np.sum((grand - np.mean(grand)) ** 2))
        effect = float(ss_between / ss_total) if ss_total > 0 else float("nan")
        effect_name, test_used = "eta^2", "one-way ANOVA"
    elif run == "welch":
        F, df1, df2, p = _welch_anova(groups)
        statistic, df = F, df2
        effect, effect_name, test_used = None, None, "Welch's ANOVA"
    else:  # kruskal
        res = stats.kruskal(*groups)
        statistic, p = float(res.statistic), float(res.pvalue)
        k, N = len(groups), len(grand)
        df = float(k - 1)
        effect = float((statistic - k + 1) / (N - k)) if N > k else float("nan")
        effect_name, test_used = "epsilon^2", "Kruskal-Wallis"

    a_reasons = {
        "kruskal": f"normality failed (AD p={worst_norm.p_value:.3g}) -> non-parametric Kruskal-Wallis",
        "welch": f"groups ~normal but unequal variance (Levene p={cv.p_value:.3g}) -> Welch's ANOVA",
        "anova": "groups ~normal with equal variance -> classic one-way ANOVA",
    }
    selection_reason = a_reasons[run] if routed else None
    recommendation = None
    if forced is not None and forced != appropriate:
        recommendation = (f"You forced {test_used}, but the assumption checks indicate "
                          f"{a_reasons[appropriate]} would be more appropriate.")
    hist = _history(base_history, "test_anova",
                    {"test_used": test_used, "appropriate": appropriate, "p_value": p,
                     "routed": routed, "groups": len(groups)}, checks)
    return HypothesisResult(
        h0="all group means are equal", h1="at least one group mean differs",
        requested="one-way comparison", test_used=test_used, statistic=statistic, p_value=p, df=df,
        effect_size=effect, effect_name=effect_name, ci=None, alternative="two-sided",
        routed=routed, recommendation=recommendation, selection_reason=selection_reason,
        _groups=tuple(groups), _labels=labels, assumptions=_neutral(checks), history=hist,
    )


# --------------------------------------------------------------------------- #
# 5. Variance
# --------------------------------------------------------------------------- #
def test_variance(*groups, method=None, auto=None, alpha=0.05, labels=None, base_history=()):
    """Compare variances - ROUTES by default. Two groups: F-test (normal) / Levene
    (robust); k groups: Bartlett (normal) / Levene. Any non-normal group routes to
    Levene. Force with ``method=`` ('f'/'bartlett'/'parametric' or 'levene')."""
    groups = [_clean(g) for g in groups]
    if len(groups) < 2:
        raise ValueError("variance test requires at least two groups.")
    labels = tuple(labels) if labels is not None else tuple(f"g{i+1}" for i in range(len(groups)))
    normal_checks = [_assume.check_normality(g) for g in groups]
    all_normal = all(c.passed for c in normal_checks)
    worst_norm = min(normal_checks, key=lambda c: (c.passed, c.p_value if c.p_value is not None else 1))
    checks = [worst_norm]

    parametric = "F-test" if len(groups) == 2 else "Bartlett"
    forced = None
    if method is not None:
        forced = {"f": parametric, "bartlett": parametric, "parametric": parametric,
                  "levene": "Levene"}.get(method)
        if forced is None:
            raise ValueError("method must be 'parametric' (F/Bartlett) or 'levene'.")
    appropriate = parametric if all_normal else "Levene"
    run = forced if forced is not None else appropriate    # ROUTE by default
    routed = forced is None

    if run == "F-test":
        a, b = groups
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        F = float(va / vb)
        dfn, dfd = len(a) - 1, len(b) - 1
        cdf = stats.f.cdf(F, dfn, dfd)
        p = float(2 * min(cdf, 1 - cdf))
        statistic, df, test_used = F, float(dfn), "F-test (variance ratio)"
    elif run == "Bartlett":
        res = stats.bartlett(*groups)
        statistic, p, df, test_used = float(res.statistic), float(res.pvalue), float(len(groups) - 1), "Bartlett"
    else:  # Levene
        res = stats.levene(*groups, center="median")
        statistic, p, df, test_used = float(res.statistic), float(res.pvalue), float(len(groups) - 1), "Levene"

    v_reasons = {
        parametric: f"all groups ~normal -> {parametric}",
        "Levene": f"normality failed (AD p={worst_norm.p_value:.3g}) -> Levene's robust test",
    }
    selection_reason = v_reasons[run] if routed else None
    recommendation = None
    if forced is not None and forced != appropriate:
        recommendation = (f"You forced {test_used}, but the assumption checks indicate "
                          f"{v_reasons[appropriate]} would be more appropriate.")
    hist = _history(base_history, "test_variance",
                    {"test_used": test_used, "appropriate": appropriate, "p_value": p, "routed": routed},
                    checks)
    return HypothesisResult(
        h0="all group variances are equal", h1="at least one variance differs",
        requested="variance comparison", test_used=test_used, statistic=statistic, p_value=p, df=df,
        effect_size=None, effect_name=None, ci=None, alternative="two-sided",
        routed=routed, recommendation=recommendation, selection_reason=selection_reason,
        _groups=tuple(groups), _labels=labels, assumptions=_neutral(checks), history=hist,
    )


# --------------------------------------------------------------------------- #
# 6. Proportions
# --------------------------------------------------------------------------- #
def _prop_validity(name, count, size, p_expected):
    mincell = float(min(size * p_expected, size * (1 - p_expected)))
    passed = mincell >= _assume.PROPORTION_MIN_EXPECTED
    rec = None
    if not passed:
        rec = (f"Normal approximation unreliable (min expected count {mincell:.1f} < 5); "
               "use an exact test (binomial / Fisher).")
    return AssumptionCheck(
        name=name, test="normal approximation (np, n(1-p) >= 5)",
        statistic=mincell, p_value=None, passed=passed,
        magnitude=mincell, magnitude_label="min expected count", reliability="ok",
        n=int(size), recommendation=rec,
    )


def test_proportion(x, n, p0, *, alternative="two-sided", auto=False, base_history=()):
    """One-proportion z-test; validity check recommends an exact binomial test if marginal."""
    if alternative not in _ALT:
        raise ValueError(f"alternative must be one of {_ALT}.")
    phat = x / n
    check = _prop_validity("normal_approximation", x, n, p0)
    run = "binomial" if (auto and not check.passed) else "z"
    routed = run != "z"

    if run == "z":
        se = np.sqrt(p0 * (1 - p0) / n)
        z = (phat - p0) / se
        if alternative == "two-sided":
            p = float(2 * stats.norm.sf(abs(z)))
        elif alternative == "greater":
            p = float(stats.norm.sf(z))
        else:
            p = float(stats.norm.cdf(z))
        statistic, df, test_used = float(z), None, "one-proportion z"
    else:
        res = stats.binomtest(x, n, p0, alternative=alternative)
        statistic, p, df, test_used = float(phat), float(res.pvalue), None, "exact binomial"

    recommendation = check.recommendation if (not check.passed and run == "z") else None
    hist = _history(base_history, "test_proportion",
                    {"x": x, "n": n, "p0": p0, "test_used": test_used, "p_value": p, "auto": auto},
                    [check])
    return HypothesisResult(
        h0=f"p = {p0}", h1=f"p != {p0}" if alternative == "two-sided" else f"p {'<' if alternative=='less' else '>'} {p0}",
        requested="one-proportion z", test_used=test_used, statistic=statistic, p_value=p, df=df,
        effect_size=float(phat - p0), effect_name="p_hat - p0", ci=None, alternative=alternative,
        routed=routed, recommendation=recommendation,
        _groups=(np.array([phat]),), _labels=("p_hat",), _target=float(p0),
        assumptions=_neutral([check]), history=hist,
    )


def test_proportions(x1, n1, x2, n2, *, alternative="two-sided", auto=False, base_history=()):
    """Two-proportion z-test; validity checks recommend Fisher's exact if marginal."""
    if alternative not in _ALT:
        raise ValueError(f"alternative must be one of {_ALT}.")
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    c1 = _prop_validity("normal_approximation_1", x1, n1, pooled)
    c2 = _prop_validity("normal_approximation_2", x2, n2, pooled)
    checks = [c1, c2]
    valid = c1.passed and c2.passed
    run = "fisher" if (auto and not valid) else "z"
    routed = run != "z"

    if run == "z":
        se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
        z = (p1 - p2) / se
        if alternative == "two-sided":
            p = float(2 * stats.norm.sf(abs(z)))
        elif alternative == "greater":
            p = float(stats.norm.sf(z))
        else:
            p = float(stats.norm.cdf(z))
        statistic, test_used = float(z), "two-proportion z"
    else:
        table = [[x1, n1 - x1], [x2, n2 - x2]]
        odds, p = stats.fisher_exact(table, alternative=alternative)
        statistic, test_used = float(odds), "Fisher's exact"

    recommendation = None
    if not valid and run == "z":
        recommendation = ("Normal approximation marginal in at least one group; Fisher's exact test "
                          "is more reliable. Rerun with auto=True.")
    hist = _history(base_history, "test_proportions",
                    {"x1": x1, "n1": n1, "x2": x2, "n2": n2, "test_used": test_used,
                     "p_value": p, "auto": auto}, checks)
    return HypothesisResult(
        h0="p1 = p2", h1="p1 != p2" if alternative == "two-sided" else f"p1 {'<' if alternative=='less' else '>'} p2",
        requested="two-proportion z", test_used=test_used, statistic=statistic, p_value=float(p), df=None,
        effect_size=float(p1 - p2), effect_name="p1 - p2", ci=None, alternative=alternative,
        routed=routed, recommendation=recommendation,
        _groups=(np.array([p1]), np.array([p2])), _labels=("p1", "p2"),
        assumptions=_neutral(checks), history=hist,
    )
