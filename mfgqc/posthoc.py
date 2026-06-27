"""Multiple comparisons (post-hoc), routed by the assumptions already on an
ANOVA-family result, inheriting the same surface-the-routing discipline.

After a significant omnibus test, ``.posthoc()`` selects:
- equal variances and normal residuals -> Tukey HSD (all pairwise, family-wise),
- unequal variances -> Games-Howell,
- a Kruskal-Wallis route -> Dunn's test with a stated p-adjustment,
- a ``control=`` argument -> Dunnett against that level.

Each pairwise difference is reported with its confidence interval and adjusted p,
and the family-wise method is stated. ``method=`` forces a choice; forcing still
surfaces the routing it overrode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from scipy import stats

from ._result import QCResult
from .assumptions import AssumptionCheck
from .data import Step

_VALID = ("tukey", "games-howell", "dunn", "dunnett")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Pair:
    a: str
    b: str
    diff: float
    ci: tuple[float, float]
    p_adj: float
    significant: bool


@dataclass(frozen=True, repr=False)
class PosthocResult(QCResult):
    """Pairwise multiple-comparison result (immutable)."""

    method: str
    family: str                    # family-wise control description
    pairs: tuple
    alpha: float
    routed: bool
    route_reason: str
    control: str | None = None
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"Post-hoc multiple comparisons ({self.method})"

    def _summary_lines(self) -> list[str]:
        lines = [f"family-wise control: {self.family}",
                 (f"routed: {self.route_reason}" if self.routed
                  else f"forced method={self.method} (overrode: {self.route_reason})"),
                 ""]
        head = f"{'pair':<20}{'diff':>12}{'95% CI low':>14}{'95% CI high':>14}{'p adj':>10}"
        lines.append(head)
        for p in self.pairs:
            star = " *" if p.significant else ""
            lo, hi = p.ci
            lines.append(f"{(p.a + ' - ' + p.b):<20}{p.diff:>12.4g}{lo:>14.4g}"
                         f"{hi:>14.4g}{p.p_adj:>10.3g}{star}")
        sig = [f"{p.a} vs {p.b}" for p in self.pairs if p.significant]
        lines += ["", f"significant pairs (p < {self.alpha}): {', '.join(sig) or '(none)'}"]
        return lines

    def summary(self) -> dict:
        out = {"method": self.method, "family": self.family, "alpha": self.alpha,
               "n_significant": sum(p.significant for p in self.pairs)}
        for p in self.pairs:
            key = f"{p.a}|{p.b}"
            out[f"diff[{key}]"] = p.diff
            out[f"p_adj[{key}]"] = p.p_adj
        return out

    def _render_standalone(self, fig, kind, **kwargs):
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        from . import palette as _pal
        pal = _pal.active()
        ys = np.arange(len(self.pairs))
        for y, p in zip(ys, self.pairs):
            lo, hi = p.ci
            col = pal.ooc if p.significant else pal.data
            ax.plot([lo, hi], [y, y], color=col, lw=2)
            ax.scatter([p.diff], [y], color=col, zorder=5)
        ax.axvline(0, color=pal.limit, ls="--", lw=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{p.a} - {p.b}" for p in self.pairs], fontsize=8)
        ax.set_xlabel("difference in means")
        ax.set_title(self._title())
        return ax


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def _route(assumptions, test_used, control) -> tuple[str, str]:
    if control is not None:
        return "dunnett", f"control={control!r} given: Dunnett against the control level"
    if "kruskal" in str(test_used).lower():
        return "dunn", "omnibus was the Kruskal-Wallis route (non-normal): Dunn's test"
    homo = next((a for a in assumptions if a.name == "homogeneity_of_variance"), None)
    if homo is not None and not homo.passed:
        return "games-howell", "unequal variances (Levene): Games-Howell"
    return "tukey", "equal variances and normal: Tukey HSD"


# --------------------------------------------------------------------------- #
# Pairwise engines
# --------------------------------------------------------------------------- #
def _stats(groups):
    means = np.array([g.mean() for g in groups])
    ns = np.array([g.size for g in groups])
    vars_ = np.array([g.var(ddof=1) for g in groups])
    return means, ns, vars_


def _tukey(groups, labels, alpha):
    means, ns, vars_ = _stats(groups)
    k = len(groups)
    N = ns.sum()
    df = N - k
    mse = float(np.sum((ns - 1) * vars_) / df)
    qcrit = stats.studentized_range.ppf(1 - alpha, k, df)
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = means[i] - means[j]
            se = np.sqrt(mse / 2.0 * (1 / ns[i] + 1 / ns[j]))
            q = abs(diff) / se
            p = float(stats.studentized_range.sf(q, k, df))
            half = qcrit * se
            pairs.append(Pair(labels[i], labels[j], float(diff),
                              (float(diff - half), float(diff + half)), p, p < alpha))
    return pairs, "Tukey HSD (family-wise across all pairs)"


def _games_howell(groups, labels, alpha):
    means, ns, vars_ = _stats(groups)
    k = len(groups)
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = means[i] - means[j]
            se = np.sqrt(vars_[i] / ns[i] + vars_[j] / ns[j])
            df = (vars_[i] / ns[i] + vars_[j] / ns[j]) ** 2 / (
                (vars_[i] / ns[i]) ** 2 / (ns[i] - 1) + (vars_[j] / ns[j]) ** 2 / (ns[j] - 1))
            q = np.sqrt(2) * abs(diff) / se
            p = float(stats.studentized_range.sf(q, k, df))
            half = stats.studentized_range.ppf(1 - alpha, k, df) * se / np.sqrt(2)
            pairs.append(Pair(labels[i], labels[j], float(diff),
                              (float(diff - half), float(diff + half)), p, p < alpha))
    return pairs, "Games-Howell (Welch-corrected, family-wise)"


def _holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def _dunn(groups, labels, alpha):
    all_vals = np.concatenate(groups)
    ranks = stats.rankdata(all_vals)
    N = all_vals.size
    # tie correction
    _, counts = np.unique(all_vals, return_counts=True)
    ties = float(np.sum(counts ** 3 - counts))
    sigma2 = (N * (N + 1) / 12.0) - ties / (12.0 * (N - 1))
    rbar, ns = [], []
    pos = 0
    for g in groups:
        rbar.append(ranks[pos:pos + g.size].mean()); ns.append(g.size); pos += g.size
    rbar = np.array(rbar); ns = np.array(ns)
    k = len(groups)
    raw_p, info = [], []
    for i in range(k):
        for j in range(i + 1, k):
            se = np.sqrt(sigma2 * (1 / ns[i] + 1 / ns[j]))
            z = (rbar[i] - rbar[j]) / se
            raw_p.append(float(2 * stats.norm.sf(abs(z))))
            info.append((i, j, float(rbar[i] - rbar[j])))
    adj = _holm(np.array(raw_p))
    pairs = [Pair(labels[i], labels[j], diff, (float("nan"), float("nan")),
                  float(adj[m]), adj[m] < alpha) for m, (i, j, diff) in enumerate(info)]
    return pairs, "Dunn's test (Holm-adjusted, rank-based)"


def _dunnett(groups, labels, control, alpha):
    if control not in labels:
        raise ValueError(f"control={control!r} is not one of the levels {labels}.")
    ci = labels.index(control)
    exp = [g for m, g in enumerate(groups) if m != ci]
    exp_labels = [labels[m] for m in range(len(labels)) if m != ci]
    res = stats.dunnett(*exp, control=groups[ci], alternative="two-sided")
    conf = res.confidence_interval(confidence_level=1 - alpha)
    pairs = []
    cm = groups[ci].mean()
    for m, lab in enumerate(exp_labels):
        diff = groups[labels.index(lab)].mean() - cm
        pairs.append(Pair(lab, control, float(diff),
                          (float(conf.low[m]), float(conf.high[m])),
                          float(res.pvalue[m]), res.pvalue[m] < alpha))
    return pairs, f"Dunnett vs control={control!r} (family-wise)"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def compute(groups, labels, assumptions, test_used, *, method=None, control=None,
            alpha: float = 0.05, base_history=()) -> PosthocResult:
    """Run the routed (or forced) pairwise comparisons. See module docstring."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    labels = [str(x) for x in labels]
    if len(groups) < 3 and control is None:
        raise ValueError("post-hoc needs at least 3 groups (or a control for Dunnett).")
    chosen, reason = _route(assumptions, test_used, control)
    routed = method is None
    if method is not None:
        if method not in _VALID:
            raise ValueError(f"method must be one of {_VALID}; got {method!r}.")
        reason = f"would have routed to {chosen} ({reason})"
        chosen = method

    if chosen == "tukey":
        pairs, family = _tukey(groups, labels, alpha)
    elif chosen == "games-howell":
        pairs, family = _games_howell(groups, labels, alpha)
    elif chosen == "dunn":
        pairs, family = _dunn(groups, labels, alpha)
    else:
        pairs, family = _dunnett(groups, labels, control, alpha)

    flag = AssumptionCheck("multiplicity", "family-wise control", float("nan"), None,
                           True, None, None, "ok", sum(g.size for g in groups),
                           None)
    step = Step(operation="posthoc", params={"method": chosen, "control": control,
                                              "routed": routed}, n_affected=None, timestamp=_now())
    return PosthocResult(method=chosen, family=family, pairs=tuple(pairs), alpha=alpha,
                         routed=routed, route_reason=reason, control=control,
                         assumptions=[flag], history=tuple(base_history) + (step,))
