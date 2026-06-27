"""System reliability (series, parallel, k-of-n, nested block diagrams) and the
ISO 281 rated bearing life (L10).

Series and parallel formulas assume independent component failures; that is
surfaced. L10 is a fleet rating at 90 percent reliability under the stated load
and speed, not a single-unit prediction; that is surfaced too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from scipy import stats

from .._result import QCResult
from ..assumptions import AssumptionCheck
from ..data import Step


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# System reliability
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class SystemReliabilityResult(QCResult):
    """Composed system reliability (immutable)."""

    structure: str
    reliability: float
    n_components: int
    component_r_repr: float
    _curve: object = field(repr=False, default=None)
    detail: str = ""
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"System reliability ({self.structure})"

    def _summary_lines(self) -> list[str]:
        return [f"structure: {self.structure}   components: {self.n_components}",
                self.detail,
                f"system reliability = {self.reliability:.6g}",
                "",
                "assumes the component failures are INDEPENDENT (series/parallel formulas); "
                "shared causes or common-mode failures break this."]

    def summary(self) -> dict:
        return {"structure": self.structure, "reliability": self.reliability,
                "n_components": self.n_components}

    def _render_standalone(self, fig, kind, **kwargs):
        from . import views
        views.system_view(self, fig, kind)

    def _render_axes(self, ax, kind, **kwargs):
        r = np.linspace(0, 1, 100)
        ax.plot(r, self._curve(r))
        return ax


def _indep_flag(n):
    return AssumptionCheck("independence", "series/parallel assumption", float("nan"), None,
                           True, None, None, "ok", n,
                           "the formula assumes independent component failures; check for "
                           "common-mode / shared-cause failures.")


def _result(structure, reliability, comps, curve, detail):
    n = len(comps) if hasattr(comps, "__len__") else int(comps)
    rep = float(np.mean(comps)) if hasattr(comps, "__len__") else float(reliability)
    step = Step(operation=f"reliability.{structure}",
                params={"reliability": float(reliability), "n": n}, n_affected=None, timestamp=_now())
    return SystemReliabilityResult(
        structure=structure, reliability=float(reliability), n_components=n,
        component_r_repr=rep, _curve=curve, detail=detail,
        assumptions=[_indep_flag(n)], history=(step,))


def series(components) -> SystemReliabilityResult:
    """Series system: R = product of component reliabilities (all must survive)."""
    comps = [float(c) for c in components]
    R = float(np.prod(comps))
    n = len(comps)
    return _result("series", R, comps, lambda r: np.asarray(r) ** n,
                   f"R = prod({', '.join(f'{c:g}' for c in comps)})")


def parallel(components) -> SystemReliabilityResult:
    """Parallel (redundant) system: R = 1 - product of unreliabilities."""
    comps = [float(c) for c in components]
    R = float(1 - np.prod([1 - c for c in comps]))
    n = len(comps)
    return _result("parallel", R, comps, lambda r: 1 - (1 - np.asarray(r)) ** n,
                   f"R = 1 - prod(1 - Ri)")


def k_of_n(k: int, n: int, reliability: float) -> SystemReliabilityResult:
    """k-out-of-n system (at least k of n identical components survive): binomial tail."""
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n; got k={k}, n={n}.")
    R = float(stats.binom.sf(k - 1, n, reliability))
    return _result("k_of_n", R, [reliability] * n,
                   lambda r: stats.binom.sf(k - 1, n, np.asarray(r)),
                   f"R = P(at least {k} of {n} survive), component R = {reliability:g}")


def system(blocks) -> SystemReliabilityResult:
    """Nested series/parallel block diagram. ``blocks`` is a float, a list (series),
    or a dict ``{'series': [...]}`` / ``{'parallel': [...]}`` nesting the above."""
    def _eval(b, r_override=None):
        if isinstance(b, (int, float)):
            return float(b) if r_override is None else float(r_override)
        if isinstance(b, list):
            return float(np.prod([_eval(x, r_override) for x in b]))
        if isinstance(b, dict):
            (op, items), = b.items()
            vals = [_eval(x, r_override) for x in items]
            if op == "series":
                return float(np.prod(vals))
            if op == "parallel":
                return float(1 - np.prod([1 - v for v in vals]))
            raise ValueError(f"block op must be 'series'/'parallel'; got {op!r}.")
        raise ValueError(f"unsupported block {b!r}.")

    def _count(b):
        if isinstance(b, (int, float)):
            return 1
        items = b if isinstance(b, list) else list(b.values())[0]
        return sum(_count(x) for x in items)

    R = _eval(blocks)
    n = _count(blocks)
    return _result("block-diagram", R, [R] * n,
                   lambda r: np.array([_eval(blocks, rr) for rr in np.atleast_1d(r)]),
                   "nested series/parallel composition")


# --------------------------------------------------------------------------- #
# ISO 281 bearing life (L10)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class BearingLifeResult(QCResult):
    """ISO 281 rated bearing life L10 (immutable)."""

    C: float
    P: float
    rpm: float
    exponent: float
    L10_revs_million: float
    L10_hours: float
    rated: dict
    kind: str
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"ISO 281 rated bearing life ({self.kind})"

    def _summary_lines(self) -> list[str]:
        lines = [f"C (dynamic rating) = {self.C:g}   P (equivalent load) = {self.P:g}   "
                 f"speed = {self.rpm:g} rpm   life exponent p = {self.exponent:g}",
                 f"L10 = {self.L10_revs_million:.5g} million revolutions = "
                 f"{self.L10_hours:.6g} hours", "",
                 "rated life at other reliabilities (ISO 281 a1 factor):"]
        for R, h in self.rated.items():
            lines.append(f"  L{int(round((1-R)*100)):>2} (R={R:.2f}): {h:.6g} hours")
        lines += ["",
                  "L10 is a FLEET rating: 90% of identical bearings reach it under the stated "
                  "constant load and speed. It is not a single-unit prediction and does not adapt "
                  "to condition data."]
        return lines

    def summary(self) -> dict:
        return {"C": self.C, "P": self.P, "rpm": self.rpm, "exponent": self.exponent,
                "L10_revs_million": self.L10_revs_million, "L10_hours": self.L10_hours}

    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        self._render_axes(ax, kind)

    def _render_axes(self, ax, kind, **kwargs):
        from .. import palette as _pal
        pal = _pal.active()
        Rs = sorted(self.rated)
        ax.plot([r for r in Rs], [self.rated[r] for r in Rs], marker="o", color=pal.center)
        ax.set_xlabel("reliability"); ax.set_ylabel("rated life (hours)")
        ax.set_title(self._title())
        return ax


def bearing_life(C: float, P: float, rpm: float, kind: str = "ball") -> BearingLifeResult:
    """ISO 281 basic rated life L10 from dynamic rating C, equivalent load P, speed rpm."""
    if C <= 0 or P <= 0 or rpm <= 0:
        raise ValueError(f"C, P and rpm must all be positive; got C={C}, P={P}, rpm={rpm}.")
    if kind not in ("ball", "roller"):
        raise ValueError("kind must be 'ball' or 'roller'.")
    p = 3.0 if kind == "ball" else 10.0 / 3.0
    L10_rev_million = (C / P) ** p
    L10_hours = (1e6 / (60.0 * rpm)) * L10_rev_million
    # ISO 281 reliability adjustment a1 (Weibull slope ~1.5)
    rated = {}
    for R in (0.90, 0.95, 0.96, 0.97, 0.98, 0.99):
        a1 = (np.log(1.0 / R) / np.log(1.0 / 0.90)) ** (1.0 / 1.5)
        rated[R] = float(a1 * L10_hours)
    flag = AssumptionCheck("constant_load_speed", "ISO 281 basis", float("nan"), None, True,
                           None, None, "ok", 0,
                           "L10 assumes constant load and speed; vary either and recompute.")
    step = Step(operation="bearing_life", params={"C": C, "P": P, "rpm": rpm, "kind": kind,
                                                  "L10_hours": L10_hours}, n_affected=None,
                timestamp=_now())
    return BearingLifeResult(C=float(C), P=float(P), rpm=float(rpm), exponent=p,
                             L10_revs_million=float(L10_rev_million), L10_hours=float(L10_hours),
                             rated=rated, kind=kind, assumptions=[flag], history=(step,))
