"""Kaplan-Meier: the nonparametric maximum-likelihood estimate of R(t).

The product-limit step function with Greenwood-variance confidence bounds, the
assumption-free baseline and the source of the empirical plotting positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from scipy import stats

from .._result import QCResult
from ..data import QCData, Step


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, repr=False)
class KaplanMeierResult(QCResult):
    """Kaplan-Meier survival estimate (immutable)."""

    times: np.ndarray
    survival: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    median_life: float
    median_ci: tuple
    n: int
    n_fail: int
    conf: float
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def R(self, t):
        idx = np.searchsorted(self.times, t, side="right") - 1
        return float(self.survival[idx]) if idx >= 0 else 1.0

    def _title(self) -> str:
        return f"Kaplan-Meier R(t): {self.n_fail} failures / {self.n} units"

    def _summary_lines(self) -> list[str]:
        ml = "not reached" if not np.isfinite(self.median_life) else f"{self.median_life:.5g}"
        lines = [f"n = {self.n}   failures = {self.n_fail}   "
                 f"suspensions = {self.n - self.n_fail}",
                 f"median life = {ml}"
                 + ("" if not np.isfinite(self.median_ci[0])
                    else f"   [{self.median_ci[0]:.5g}, {self.median_ci[1]:.5g}]"),
                 "",
                 f"{'t':>10}{'R(t)':>10}{'lower':>10}{'upper':>10}"]
        step = max(1, len(self.times) // 12)
        for i in range(0, len(self.times), step):
            lines.append(f"{self.times[i]:>10.4g}{self.survival[i]:>10.4f}"
                         f"{self.lower[i]:>10.4f}{self.upper[i]:>10.4f}")
        return lines

    def summary(self) -> dict:
        return {"n": self.n, "n_fail": self.n_fail, "median_life": self.median_life,
                "median_ci_low": self.median_ci[0], "median_ci_high": self.median_ci[1]}

    def _render_standalone(self, fig, kind, **kwargs):
        from . import views
        views.km_view(self, fig, kind)

    def _render_axes(self, ax, kind, **kwargs):
        from . import palette as _pal
        pal = _pal.active()
        ax.step(self.times, self.survival, where="post", color=pal.center, lw=2)
        ax.set_ylim(0, 1.02); ax.set_xlabel("time"); ax.set_ylabel("R(t)")
        return ax


def kaplan_meier(qc: QCData, *, conf: float = 0.95) -> KaplanMeierResult:
    """Kaplan-Meier R(t) from the time and event roles (event 1=failure, 0=suspension)."""
    frame = qc.frame
    tcol = qc.meta.roles.get("time", qc.meta.measure)
    ecol = qc.meta.roles.get("event")
    t = np.asarray(frame[tcol], dtype=float)
    e = (np.asarray(frame[ecol], dtype=float) if ecol else np.ones(t.size))
    ok = np.isfinite(t)
    t, e = t[ok], e[ok]
    n = t.size
    order = np.argsort(t)
    t, e = t[order], e[order]

    uniq = np.unique(t[e == 1])
    at_risk = np.array([np.sum(t >= ut) for ut in uniq], dtype=float)
    d = np.array([np.sum((t == ut) & (e == 1)) for ut in uniq], dtype=float)
    surv = np.cumprod(1 - d / at_risk)
    # Greenwood variance of S(t); guard the at_risk == d step (S hits 0, term undefined)
    denom = at_risk * (at_risk - d)
    term = np.where(denom > 0, d / np.where(denom > 0, denom, 1.0), 0.0)
    cum = np.cumsum(term)
    var = surv ** 2 * cum
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    se = np.sqrt(np.clip(var, 0, None))
    lower = np.clip(surv - z * se, 0, 1)
    upper = np.clip(surv + z * se, 0, 1)

    times = np.concatenate([[0.0], uniq])
    survival = np.concatenate([[1.0], surv])
    lo = np.concatenate([[1.0], lower]); up = np.concatenate([[1.0], upper])

    below = np.where(surv <= 0.5)[0]
    median_life = float(uniq[below[0]]) if below.size else float("inf")
    med_lo = med_hi = float("nan")
    if below.size:
        lo_idx = np.where(lower <= 0.5)[0]
        hi_idx = np.where(upper <= 0.5)[0]
        med_lo = float(uniq[hi_idx[0]]) if hi_idx.size else float("nan")
        med_hi = float(uniq[lo_idx[0]]) if lo_idx.size else float("nan")

    step = Step(operation="kaplan_meier", params={"n": n, "n_fail": int(np.sum(e == 1))},
                n_affected=n, timestamp=_now())
    return KaplanMeierResult(
        times=times, survival=survival, lower=lo, upper=up,
        median_life=median_life, median_ci=(med_lo, med_hi),
        n=n, n_fail=int(np.sum(e == 1)), conf=conf,
        assumptions=[], history=qc.history + (step,))
