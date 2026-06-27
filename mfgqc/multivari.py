"""Multi-vari study: decompose observed variation into the positional, cyclical,
and temporal families across the named (nested) factors, with the multi-vari
chart and the variance components by family.

The factors are given outermost-to-innermost (e.g. ``["shift", "part"]`` or
``["shift", "part", "position"]``). Each named factor is a grouping level; the
spread within the innermost group is the positional (within-piece) family. The
component for a level is the variance of that level's means within its parent,
which is the practical multi-vari decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ._result import QCResult
from .data import QCData, Step

_FAMILY = ["temporal", "cyclical", "within-group"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, repr=False)
class MultivariResult(QCResult):
    """Multi-vari result (immutable): variance components by family."""

    factors: tuple
    components: dict                 # name -> variance component
    families: dict                   # family -> name (mapping)
    percents: dict                   # name -> percent of total
    total: float
    n: int
    response: str = "y"
    _frame: object = field(repr=False, default=None)
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"Multi-vari study: {self.response} across {', '.join(self.factors)}"

    def _summary_lines(self) -> list[str]:
        lines = [f"n = {self.n}   total variance = {self.total:.4g}", "",
                 f"{'family':<16}{'source':<16}{'variance':>12}{'% of total':>12}"]
        for fam, name in self.families.items():
            lines.append(f"{fam:<16}{name:<16}{self.components[name]:>12.4g}"
                         f"{self.percents[name]:>11.1f}%")
        dom = max(self.components, key=self.components.get)
        lines += ["", f"largest family: {dom} "
                  f"({[f for f, n in self.families.items() if n == dom][0]}, "
                  f"{self.percents[dom]:.0f}% of total) - focus improvement there."]
        return lines

    def summary(self) -> dict:
        out = {"response": self.response, "n": self.n, "total_variance": self.total}
        for name, v in self.components.items():
            out[f"var[{name}]"] = v
            out[f"pct[{name}]"] = self.percents[name]
        return out

    def _render_standalone(self, fig, kind, **kwargs):
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        from . import palette as _pal
        pal = _pal.active()
        df = self._frame
        outer, inner = self.factors[0], self.factors[-1]
        x = 0
        ticks, labels = [], []
        for ov in pd.unique(df[outer]):
            block = df[df[outer] == ov]
            xs, ys = [], []
            for iv in pd.unique(block[inner]):
                vals = block[block[inner] == iv][self.response].to_numpy(dtype=float)
                ax.scatter([x] * len(vals), vals, color=pal.data, s=14, zorder=4)
                ax.plot([x, x], [vals.min(), vals.max()], color=pal.data, lw=1)
                xs.append(x); ys.append(vals.mean()); x += 1
            ax.plot(xs, ys, color=pal.center, lw=1.5, marker="o", ms=4)
            ticks.append(np.mean(xs)); labels.append(str(ov)); x += 1
        ax.set_xticks(ticks); ax.set_xticklabels(labels)
        ax.set_xlabel(outer); ax.set_ylabel(self.response)
        ax.set_title(self._title())
        return ax


def _within_means_variance(df, group_cols, value):
    """Mean (over parent groups) of the variance of child-level means."""
    if len(group_cols) == 1:
        means = df.groupby(group_cols[0])[value].mean().to_numpy()
        return float(np.var(means, ddof=1)) if means.size > 1 else 0.0
    parent, child = group_cols[0], group_cols[1]
    vs = []
    for _pv, block in df.groupby(parent):
        means = block.groupby(child)[value].mean().to_numpy()
        if means.size > 1:
            vs.append(np.var(means, ddof=1))
    return float(np.mean(vs)) if vs else 0.0


def compute(qc: QCData, factors: list[str]) -> MultivariResult:
    """Multi-vari decomposition over the nested ``factors`` (outermost first)."""
    factors = list(factors)
    if not 2 <= len(factors) <= 3:
        raise ValueError("multivari takes 2 or 3 nested factors (outermost first).")
    response = qc.meta.measure
    frame = qc.frame[[response] + factors].apply(
        lambda s: pd.to_numeric(s, errors="coerce") if s.name == response else s).dropna()
    n = len(frame)

    components: dict = {}
    # outermost factor: variance of its level means
    components[factors[0]] = _within_means_variance(frame, [factors[0]], response)
    # middle factors: variance of their means within the parent
    for i in range(1, len(factors)):
        components[factors[i]] = _within_means_variance(frame, factors[:i + 1][-2:], response)
    # within-innermost-group spread (the positional / residual family)
    within = []
    for _key, block in frame.groupby(factors):
        v = block[response].to_numpy(dtype=float)
        if v.size > 1:
            within.append(np.var(v, ddof=1))
    components["within"] = float(np.mean(within)) if within else 0.0

    # family mapping (exactly the standard three; an empty residual is dropped)
    families = {"temporal": factors[0], "cyclical": factors[1]}
    if len(factors) == 3:
        families["positional"] = factors[2]
        if components.get("within", 0.0) > 1e-12:
            families["residual"] = "within"
        else:
            components.pop("within", None)
    else:
        families["positional"] = "within"

    total = sum(components.values()) or 1.0
    percents = {k: 100.0 * v / total for k, v in components.items()}

    step = Step(operation="multivari", params={"factors": factors, "components": components},
                n_affected=n, timestamp=_now())
    return MultivariResult(
        factors=tuple(factors), components=components, families=families,
        percents=percents, total=float(total), n=n, response=response,
        _frame=frame, assumptions=[], history=qc.history + (step,))
