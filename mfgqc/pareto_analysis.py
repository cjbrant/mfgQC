"""Pareto analysis and chi-square / contingency-table analysis.

Two descriptive/inferential tools for attribute (categorical) quality data:

1. :func:`pareto` ranks categories of defect/complaint counts to expose the
   "vital few" - the small set of categories that account for the bulk (default
   80%) of the total. Pure descriptive; no statistical assumptions.

2. :func:`contingency` / :func:`test_independence` run Pearson's chi-square test
   of independence on a contingency table (NO Yates continuity correction),
   reporting the statistic, p-value, degrees of freedom, the expected-count
   table, and Cramer's V as an effect size. The chi-square approximation's
   validity guardrail (min expected cell count >= 5) is surfaced as a flags-v2
   :class:`~mfgqc.assumptions.AssumptionCheck` - reported, never auto-switched.

Correctness anchors (verified against scipy):
- contingency [[30,20],[15,35]] (correction=False): chi2 = 9.091, p = 0.0026,
  dof = 1, Cramer's V = 0.3015.
- Pareto on {A:50, B:30, C:15, D:5}: cum% = 50, 80, 95, 100; vital_few = (A, B).
"""

from __future__ import annotations

from . import palette as _pal

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from ._result import QCResult
from .assumptions import AssumptionCheck
from .data import Step

# Minimum expected cell count for the chi-square approximation to be reliable
# (Cochran's rule of thumb).
_MIN_EXPECTED = 5.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Result objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class ParetoResult(QCResult):
    """A Pareto ranking of categories by count, with the vital-few cutoff."""

    categories: tuple
    counts: tuple
    cum_pct: tuple
    vital_few: tuple
    threshold: float = 0.80
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        return int(sum(self.counts))

    def _title(self) -> str:
        return "Pareto Analysis"

    def _summary_lines(self) -> list[str]:
        top = self.categories[0] if self.categories else None
        top_pct = (self.counts[0] / self.total * 100.0) if (self.counts and self.total) else 0.0
        lines = [
            f"total count   = {self.total}",
            f"categories    = {len(self.categories)}",
            f"vital few (<= {self.threshold * 100:.0f}% cumulative) = "
            f"{len(self.vital_few)}: {', '.join(str(c) for c in self.vital_few)}",
        ]
        if top is not None:
            lines.append(f"top category  = {top} ({top_pct:.3g}% of total)")
        lines.append("")
        lines.append("Ranked categories:")
        for cat, cnt, cum in zip(self.categories, self.counts, self.cum_pct):
            mark = "*" if cat in self.vital_few else " "
            lines.append(f"  {mark} {cat}: {cnt}  (cum {cum:.3g}%)")
        return lines

    def summary(self) -> dict:
        top = self.categories[0] if self.categories else None
        top_pct = (self.counts[0] / self.total * 100.0) if (self.counts and self.total) else 0.0
        return {
            "total": self.total,
            "n_categories": len(self.categories),
            "vital_few_count": len(self.vital_few),
            "top_category": top,
            "top_category_pct": round(top_pct, 4),
        }

    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        self._render_axes(ax, kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        cats = [str(c) for c in self.categories]
        x = np.arange(len(cats))
        # Descending bars (counts), coloring the vital few distinctly.
        colors = [_pal.active().center if c in self.vital_few else _pal.active().data for c in self.categories]
        ax.bar(x, self.counts, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_ylabel("count")
        ax.set_title(self._title())

        # Cumulative percentage line on a secondary axis.
        ax2 = ax.twinx()
        ax2.plot(x, self.cum_pct, color=_pal.active().ooc, marker="o", lw=1.5,
                 label="cumulative %")
        ax2.axhline(self.threshold * 100.0, color=_pal.active().muted, ls="--", lw=1.0,
                    label=f"{self.threshold * 100:.0f}% reference")
        ax2.set_ylabel("cumulative %")
        ax2.set_ylim(0, 105)
        ax2.legend(loc="center right", fontsize=8)


@dataclass(frozen=True, repr=False)
class ContingencyResult(QCResult):
    """Chi-square test of independence on a contingency table."""

    chi2: float
    p_value: float
    dof: int
    expected: np.ndarray
    cramers_v: float
    n: int
    observed: np.ndarray
    row_labels: tuple = ()
    col_labels: tuple = ()
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return "Chi-Square Test of Independence"

    def _summary_lines(self) -> list[str]:
        r, c = self.observed.shape
        return [
            f"table         = {r} x {c}   N = {self.n}",
            f"chi-square    = {self.chi2:.4g}   dof = {self.dof}",
            f"p-value       = {self.p_value:.4g}",
            f"Cramer's V    = {self.cramers_v:.4g}",
            f"min expected  = {float(np.min(self.expected)):.4g}",
        ]

    def summary(self) -> dict:
        return {
            "chi2": round(float(self.chi2), 6),
            "p_value": round(float(self.p_value), 6),
            "dof": int(self.dof),
            "cramers_v": round(float(self.cramers_v), 6),
            "n": int(self.n),
            "min_expected": round(float(np.min(self.expected)), 6),
        }

    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        self._render_axes(ax, kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        obs = np.asarray(self.observed, dtype=float)
        im = ax.imshow(obs, cmap="Blues", aspect="auto")
        r, c = obs.shape
        row_labels = [str(x) for x in (self.row_labels or range(r))]
        col_labels = [str(x) for x in (self.col_labels or range(c))]
        ax.set_xticks(np.arange(c))
        ax.set_yticks(np.arange(r))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        # Annotate observed (expected) in each cell.
        for i in range(r):
            for j in range(c):
                ax.text(j, i, f"{obs[i, j]:.0f}\n({self.expected[i, j]:.1f})",
                        ha="center", va="center", fontsize=8, color="black")
        ax.set_title(self._title())
        ax.figure.colorbar(im, ax=ax, label="observed count")


# --------------------------------------------------------------------------- #
# Pareto
# --------------------------------------------------------------------------- #
def pareto(data, category: str | None = None, count: str | None = None,
           *, threshold: float = 0.80) -> ParetoResult:
    """Pareto analysis: rank categories by count and find the vital few.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        Either a DataFrame (then ``category`` is required and names the column of
        labels; ``count`` optionally names a column of counts - if omitted, row
        frequencies of ``category`` are used), or a Series of counts indexed by
        category.
    category : str or None, optional
        Column of category labels (DataFrame input only).
    count : str or None, optional
        Column of counts (DataFrame input only). If ``None``, the frequency of
        each ``category`` value is counted.
    threshold : float, optional
        Cumulative-fraction cutoff defining the "vital few" (default 0.80).

    Returns
    -------
    ParetoResult
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1]; got {threshold!r}.")

    if isinstance(data, pd.Series):
        series = data.astype(float)
    elif isinstance(data, pd.DataFrame):
        if category is None:
            raise ValueError("pareto() on a DataFrame requires category=.")
        if category not in data.columns:
            raise ValueError(f"category column {category!r} not found in DataFrame.")
        if count is None:
            series = data[category].value_counts()
        else:
            if count not in data.columns:
                raise ValueError(f"count column {count!r} not found in DataFrame.")
            series = data.groupby(category)[count].sum()
    else:
        raise TypeError("data must be a pandas DataFrame or Series.")

    series = series.sort_values(ascending=False)
    categories = tuple(series.index)
    counts = tuple(int(round(v)) for v in series.to_numpy(dtype=float))

    total = float(sum(counts))
    if total <= 0:
        cum_pct: tuple = tuple(0.0 for _ in counts)
        vital_few: tuple = ()
    else:
        cum = np.cumsum(np.asarray(counts, dtype=float)) / total
        cum_pct = tuple(float(c * 100.0) for c in cum)
        # Vital few: categories up to AND INCLUDING the first one that reaches
        # the threshold cumulative fraction.
        vital_list = []
        for cat, frac in zip(categories, cum):
            vital_list.append(cat)
            if frac >= threshold:
                break
        vital_few = tuple(vital_list)

    step = Step(operation="pareto",
                params={"category": category, "count": count,
                        "threshold": threshold, "n_categories": len(categories)},
                n_affected=int(total), timestamp=_now())
    return ParetoResult(
        categories=categories, counts=counts, cum_pct=cum_pct,
        vital_few=vital_few, threshold=threshold,
        assumptions=[], history=(step,),
    )


# --------------------------------------------------------------------------- #
# Chi-square / contingency
# --------------------------------------------------------------------------- #
def _expected_count_check(expected: np.ndarray, n: int) -> AssumptionCheck:
    """Flags-v2 check: chi-square approximation needs min expected cell >= 5."""
    min_expected = float(np.min(expected))
    passed = min_expected >= _MIN_EXPECTED
    rec = None
    if not passed:
        rec = ("min expected cell count < 5; chi-square approximation may be "
               "invalid - consider Fisher's exact (2x2).")
    return AssumptionCheck(
        name="expected_count", test="min expected cell count >= 5",
        statistic=min_expected, p_value=None, passed=passed,
        magnitude=min_expected, magnitude_label="min expected count",
        reliability="ok", n=int(n), recommendation=rec,
    )


def _compute_contingency(observed: np.ndarray, row_labels: tuple, col_labels: tuple,
                         *, operation: str, params: dict) -> ContingencyResult:
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2:
        raise ValueError("contingency table must be 2-dimensional.")
    chi2, p, dof, expected = stats.chi2_contingency(obs, correction=False)
    n = int(round(float(obs.sum())))
    r, c = obs.shape
    k = min(r - 1, c - 1)
    cramers_v = float(np.sqrt(chi2 / (n * k))) if (n > 0 and k > 0) else float("nan")

    check = _expected_count_check(expected, n)
    step = Step(operation=operation, params=params, n_affected=n, timestamp=_now())
    assumption_step = Step(
        operation=f"assumption:{check.name}",
        params={"test": check.test, "passed": check.passed,
                "magnitude": check.magnitude, "reliability": check.reliability},
        n_affected=None, timestamp=_now(),
    )
    return ContingencyResult(
        chi2=float(chi2), p_value=float(p), dof=int(dof),
        expected=np.asarray(expected, dtype=float), cramers_v=cramers_v,
        n=n, observed=obs, row_labels=row_labels, col_labels=col_labels,
        assumptions=[check], history=(step, assumption_step),
    )


def contingency(table) -> ContingencyResult:
    """Chi-square test of independence on a contingency table of observed counts.

    Parameters
    ----------
    table : array-like or pandas.DataFrame
        A 2D table of observed counts (rows x columns).

    Returns
    -------
    ContingencyResult
        Chi-square statistic (NO Yates continuity correction), p-value, degrees
        of freedom, expected-count table, and Cramer's V. The min-expected-count
        guardrail is surfaced in ``assumptions``.
    """
    if isinstance(table, pd.DataFrame):
        observed = table.to_numpy(dtype=float)
        row_labels = tuple(table.index)
        col_labels = tuple(table.columns)
    else:
        observed = np.asarray(table, dtype=float)
        row_labels = ()
        col_labels = ()
    return _compute_contingency(
        observed, row_labels, col_labels,
        operation="contingency", params={"shape": list(observed.shape)},
    )


def test_independence(df: pd.DataFrame, row: str, col: str) -> ContingencyResult:
    """Chi-square test of independence between two categorical columns.

    Builds the contingency table from ``df[row]`` x ``df[col]`` via
    :func:`pandas.crosstab` and delegates to the same computation as
    :func:`contingency`.

    Parameters
    ----------
    df : pandas.DataFrame
        Tidy data, one row per observation.
    row, col : str
        Names of the two categorical columns to cross-tabulate.

    Returns
    -------
    ContingencyResult
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("test_independence() requires a pandas DataFrame.")
    for name in (row, col):
        if name not in df.columns:
            raise ValueError(f"column {name!r} not found in DataFrame.")
    table = pd.crosstab(df[row], df[col])
    return _compute_contingency(
        table.to_numpy(dtype=float), tuple(table.index), tuple(table.columns),
        operation="test_independence", params={"row": row, "col": col},
    )
