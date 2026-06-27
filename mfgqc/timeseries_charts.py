"""Time-weighted control charts: EWMA and tabular CUSUM.

These charts complement the Shewhart charts in :mod:`mfgqc.control_charts`. Where a
Shewhart chart reacts only to the current point, EWMA and CUSUM accumulate
information across observations and so detect small, sustained shifts in the
process mean much faster.

Both charts assume the in-control mean ``mu0`` and standard deviation ``sigma``
are known (estimated from a stable phase-I baseline). When not supplied:

* ``mu0`` defaults to the spec target if one is set, else the sample mean.
* ``sigma`` is estimated the same way an I-chart estimates it: from the average
  moving range, ``sigma = MR-bar / d2`` with ``d2 = 1.128`` for moving ranges of
  size 2. This isolates short-term (within-process) variation and is robust to
  the very mean shifts these charts are designed to detect.

This estimation choice is surfaced as an :class:`~mfgqc.assumptions.AssumptionCheck`
note on every result, mirroring the "type hints, not decisions" philosophy.
"""

from __future__ import annotations

from . import palette as _pal

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from ._result import QCResult
from .assumptions import AssumptionCheck
from .control_charts import Violation
from .data import QCData, Step

_D2_N2 = 1.128  # d2 for moving ranges of size 2 (matches I-MR sigma estimate)

_OOC = _pal.active().ooc
_LIMIT = _pal.active().axis
_CENTER = _pal.active().target
_SERIES = _pal.active().center
_SERIES2 = _pal.active().amber


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _assumption_step(a: AssumptionCheck) -> Step:
    return Step(
        operation=f"assumption:{a.name}",
        params={"test": a.test, "passed": a.passed, "magnitude": a.magnitude,
                "reliability": a.reliability, "p_value": a.p_value, "statistic": a.statistic},
        n_affected=None, timestamp=_now(),
    )


def _estimate_sigma(x: np.ndarray) -> float:
    """Short-term sigma via the moving-range method (MR-bar / d2), as in an I-chart."""
    if x.size < 2:
        return 0.0
    mr = np.abs(np.diff(x))
    mrbar = float(mr.mean())
    return mrbar / _D2_N2 if mrbar > 0 else 0.0


def _baseline_note(mu0_source: str, sigma_source: str, mu0: float, sigma: float,
                   n: int) -> AssumptionCheck:
    """Surface that mu0/sigma are assumed estimated from a stable phase-I baseline."""
    rec = ("EWMA/CUSUM assume the in-control mu0 and sigma are known from a stable "
           f"phase-I baseline; here mu0={mu0:.5g} ({mu0_source}) and sigma={sigma:.5g} "
           f"({sigma_source}). Validate phase-I stability (e.g. an I-MR chart) before "
           "trusting these limits.")
    return AssumptionCheck(
        name="in_control_parameters",
        test="phase-I baseline (assumed)",
        statistic=float("nan"),
        p_value=None,
        passed=True,
        magnitude=None,
        magnitude_label=None,
        reliability="low_power" if n < 20 else "ok",
        n=n,
        recommendation=rec,
    )


def _resolve_params(qc: QCData, mu0, sigma):
    """Resolve mu0/sigma defaults and report where each value came from."""
    x = qc.values()
    x = x[~np.isnan(x)]
    if mu0 is None:
        target = qc.meta.limits.target
        if target is not None:
            mu0 = float(target)
            mu0_source = "spec target"
        else:
            mu0 = float(x.mean())
            mu0_source = "sample mean"
    else:
        mu0 = float(mu0)
        mu0_source = "user-specified"
    if sigma is None:
        sigma = _estimate_sigma(x)
        sigma_source = "MR-bar/d2 (d2=1.128)"
    else:
        sigma = float(sigma)
        sigma_source = "user-specified"
    return x, mu0, mu0_source, sigma, sigma_source


# --------------------------------------------------------------------------- #
# EWMA
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class EWMAResult(QCResult):
    """Result of an EWMA control-chart analysis (immutable)."""

    z: np.ndarray
    ucl: np.ndarray
    lcl: np.ndarray
    mu0: float
    sigma: float
    lam: float
    L: float
    center: float
    violations: list[Violation] = field(default_factory=list)
    labels: tuple = ()
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"EWMA Chart: lambda={self.lam:g}, L={self.L:g}"

    def _summary_lines(self) -> list[str]:
        lines = [
            f"EWMA: CL={self.center:.5g}  mu0={self.mu0:.5g}  sigma={self.sigma:.5g}",
            f"UCL: {self.ucl[0]:.5g} -> {self.ucl[-1]:.5g} (time-varying)",
            f"LCL: {self.lcl[0]:.5g} -> {self.lcl[-1]:.5g} (time-varying)",
            "",
        ]
        if self.violations:
            lines.append(f"Out-of-control signals: {len(self.violations)}")
            for v in self.violations:
                lines.append(f"  point {v.point}: {v.rule} - {v.description}")
        else:
            lines.append("Out-of-control signals: none (process in control)")
        return lines

    def summary(self) -> dict:
        """Flat summary dict (no nested values)."""
        return {
            "lam": float(self.lam),
            "L": float(self.L),
            "mu0": float(self.mu0),
            "sigma": float(self.sigma),
            "n_signals": len(self.violations),
        }

    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        self._render_axes(ax, kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        z = np.asarray(self.z, dtype=float)
        x = np.arange(1, len(z) + 1)
        ax.plot(x, z, marker="o", color=_SERIES, lw=1, ms=4, zorder=2, label="EWMA z")
        ax.axhline(self.center, color=_CENTER, lw=1.2, label="CL")
        ax.plot(x, np.asarray(self.ucl, dtype=float), color=_LIMIT, ls="--", lw=1,
                drawstyle="steps-mid", label="UCL")
        ax.plot(x, np.asarray(self.lcl, dtype=float), color=_LIMIT, ls="--", lw=1,
                drawstyle="steps-mid", label="LCL")
        viol_pts = [v.point for v in self.violations if 1 <= v.point <= len(z)]
        if viol_pts:
            idx = [p - 1 for p in viol_pts]
            ax.scatter(np.array(idx) + 1, z[idx], color=_OOC, zorder=3, s=45,
                       label="out of control")
        ax.set_ylabel("EWMA")
        ax.set_xlabel("observation")
        ax.set_title("EWMA chart")
        ax.legend(loc="best", fontsize=8)


def compute_ewma(qc: QCData, lam: float = 0.1, L: float = 2.7,
                 mu0: float | None = None, sigma: float | None = None) -> EWMAResult:
    """Compute an EWMA control chart.

    Parameters
    ----------
    qc : QCData
    lam : float, optional
        Smoothing constant ``0 < lambda <= 1`` (default 0.1).
    L : float, optional
        Control-limit width in sigma units (default 2.7).
    mu0 : float or None, optional
        In-control mean. Defaults to the spec target if set, else the sample mean.
    sigma : float or None, optional
        In-control standard deviation. Defaults to MR-bar/d2 (I-chart estimate).

    Returns
    -------
    EWMAResult
    """
    if not (0.0 < lam <= 1.0):
        raise ValueError(f"lam (lambda) must be in (0, 1]; got {lam!r}.")
    if L <= 0:
        raise ValueError(f"L must be positive; got {L!r}.")

    x, mu0, mu0_src, sigma, sigma_src = _resolve_params(qc, mu0, sigma)
    n = x.size
    if n == 0:
        raise ValueError("EWMA chart requires at least one observation.")

    # EWMA recursion: z_i = lam*x_i + (1-lam)*z_{i-1}, z_0 = mu0.
    z = np.empty(n, dtype=float)
    prev = mu0
    for i in range(n):
        prev = lam * x[i] + (1.0 - lam) * prev
        z[i] = prev

    # Time-varying limits: half-width_i = L*sigma*sqrt((lam/(2-lam))*(1-(1-lam)^(2i))).
    i_arr = np.arange(1, n + 1, dtype=float)
    factor = (lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2.0 * i_arr))
    halfwidth = L * sigma * np.sqrt(factor)
    ucl = mu0 + halfwidth
    lcl = mu0 - halfwidth

    violations = [
        Violation(point=i + 1, value=float(z[i]), chart="location",
                  rule="ewma", description="EWMA statistic beyond control limit")
        for i in range(n) if z[i] > ucl[i] or z[i] < lcl[i]
    ]

    checks = [_baseline_note(mu0_src, sigma_src, mu0, sigma, n)]
    step = Step(operation="ewma_chart",
                params={"lam": lam, "L": L, "mu0": mu0, "sigma": sigma},
                n_affected=n, timestamp=_now())
    history = qc.history + (step,) + tuple(_assumption_step(a) for a in checks)

    return EWMAResult(
        z=z, ucl=ucl, lcl=lcl, mu0=mu0, sigma=sigma, lam=lam, L=L, center=mu0,
        violations=violations, labels=tuple(range(1, n + 1)),
        assumptions=checks, history=history,
    )


# --------------------------------------------------------------------------- #
# CUSUM (tabular, two-sided)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class CUSUMResult(QCResult):
    """Result of a tabular two-sided CUSUM control-chart analysis (immutable)."""

    c_plus: np.ndarray
    c_minus: np.ndarray
    K: float
    H: float
    k: float
    h: float
    mu0: float
    sigma: float
    violations: list[Violation] = field(default_factory=list)
    labels: tuple = ()
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"CUSUM Chart: k={self.k:g}, h={self.h:g}"

    def _summary_lines(self) -> list[str]:
        lines = [
            f"CUSUM: mu0={self.mu0:.5g}  sigma={self.sigma:.5g}",
            f"K (reference)={self.K:.5g}  H (decision interval)={self.H:.5g}",
            "",
        ]
        if self.violations:
            lines.append(f"Out-of-control signals: {len(self.violations)}")
            for v in self.violations:
                lines.append(f"  point {v.point}: {v.rule} - {v.description}")
        else:
            lines.append("Out-of-control signals: none (process in control)")
        return lines

    def summary(self) -> dict:
        """Flat summary dict (no nested values)."""
        return {
            "k": float(self.k),
            "h": float(self.h),
            "K": float(self.K),
            "H": float(self.H),
            "mu0": float(self.mu0),
            "sigma": float(self.sigma),
            "n_signals": len(self.violations),
        }

    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        self._render_axes(ax, kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        cp = np.asarray(self.c_plus, dtype=float)
        cm = np.asarray(self.c_minus, dtype=float)
        x = np.arange(1, len(cp) + 1)
        # C+ above zero, C- plotted on its own (negative) track for readability.
        ax.plot(x, cp, marker="o", color=_SERIES, lw=1, ms=4, zorder=2, label="C+")
        ax.plot(x, -cm, marker="s", color=_SERIES2, lw=1, ms=4, zorder=2, label="C- (negated)")
        ax.axhline(0.0, color=_pal.active().muted, lw=0.8)
        ax.axhline(self.H, color=_LIMIT, ls="--", lw=1, label="+H")
        ax.axhline(-self.H, color=_LIMIT, ls="--", lw=1, label="-H")
        viol = {v.point for v in self.violations}
        if viol:
            idx = [p - 1 for p in viol if 1 <= p <= len(cp)]
            pts = np.array(idx)
            # Mark whichever arm crossed H.
            hi = pts[cp[pts] > self.H]
            lo = pts[cm[pts] > self.H]
            if hi.size:
                ax.scatter(hi + 1, cp[hi], color=_OOC, zorder=3, s=45, label="out of control")
            if lo.size:
                ax.scatter(lo + 1, -cm[lo], color=_OOC, zorder=3, s=45)
        ax.set_ylabel("cumulative sum")
        ax.set_xlabel("observation")
        ax.set_title("CUSUM chart")
        ax.legend(loc="best", fontsize=8)


def compute_cusum(qc: QCData, k: float = 0.5, h: float = 5,
                  mu0: float | None = None, sigma: float | None = None) -> CUSUMResult:
    """Compute a tabular two-sided CUSUM control chart.

    Parameters
    ----------
    qc : QCData
    k : float, optional
        Reference value (slack) in sigma units (default 0.5). ``K = k*sigma``.
    h : float, optional
        Decision interval in sigma units (default 5). ``H = h*sigma``.
    mu0 : float or None, optional
        In-control mean. Defaults to the spec target if set, else the sample mean.
    sigma : float or None, optional
        In-control standard deviation. Defaults to MR-bar/d2 (I-chart estimate).

    Returns
    -------
    CUSUMResult
    """
    if k < 0:
        raise ValueError(f"k must be non-negative; got {k!r}.")
    if h <= 0:
        raise ValueError(f"h must be positive; got {h!r}.")

    x, mu0, mu0_src, sigma, sigma_src = _resolve_params(qc, mu0, sigma)
    n = x.size
    if n == 0:
        raise ValueError("CUSUM chart requires at least one observation.")

    K = k * sigma
    H = h * sigma

    # C+_i = max(0, x_i - (mu0 + K) + C+_{i-1}); C-_i = max(0, (mu0 - K) - x_i + C-_{i-1}).
    c_plus = np.empty(n, dtype=float)
    c_minus = np.empty(n, dtype=float)
    cp_prev = 0.0
    cm_prev = 0.0
    for i in range(n):
        cp_prev = max(0.0, x[i] - (mu0 + K) + cp_prev)
        cm_prev = max(0.0, (mu0 - K) - x[i] + cm_prev)
        c_plus[i] = cp_prev
        c_minus[i] = cm_prev

    violations = []
    for i in range(n):
        if c_plus[i] > H:
            violations.append(Violation(
                point=i + 1, value=float(c_plus[i]), chart="location",
                rule="cusum_upper", description="C+ exceeds decision interval H (upward shift)"))
        elif c_minus[i] > H:
            violations.append(Violation(
                point=i + 1, value=float(c_minus[i]), chart="location",
                rule="cusum_lower", description="C- exceeds decision interval H (downward shift)"))

    checks = [_baseline_note(mu0_src, sigma_src, mu0, sigma, n)]
    step = Step(operation="cusum_chart",
                params={"k": k, "h": h, "K": K, "H": H, "mu0": mu0, "sigma": sigma},
                n_affected=n, timestamp=_now())
    history = qc.history + (step,) + tuple(_assumption_step(a) for a in checks)

    return CUSUMResult(
        c_plus=c_plus, c_minus=c_minus, K=K, H=H, k=k, h=h, mu0=mu0, sigma=sigma,
        violations=violations, labels=tuple(range(1, n + 1)),
        assumptions=checks, history=history,
    )
