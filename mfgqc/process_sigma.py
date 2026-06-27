"""Attributes capability: defect-rate to sigma (the Six Sigma Measure piece the
variables capability path does not cover).

Two input kinds: ``defectives`` (units pass or fail, binomial) and ``defects``
(counts per unit across opportunities, Poisson). Reports DPU, DPMO, yields, the
long-term Z from the observed rate, and the short-term Z.

Critical wedge point: the 1.5 sigma shift linking Z.lt and Z.st is a CONVENTION,
not a measurement. Both are reported, the shift is stated explicitly, and no
single "sigma level" is presented without saying which basis it is on. A small
sample (wide exact CI) is flagged and tempered the same way the assumption checks
are; a zero-defect sample reports the one-sided exact bound, never an infinite
sigma.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from scipy import stats

from ._result import QCResult
from .assumptions import AssumptionCheck
from .data import Step

SHIFT = 1.5            # the conventional long-term to short-term sigma shift
_VALID_KINDS = ("defectives", "defects")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _z_from_rate(p: float) -> float:
    """Long-term Z (sigma value) from a defect rate: Z.lt = Phi^-1(1 - p)."""
    p = min(max(p, 0.0), 1.0)
    return float(stats.norm.ppf(1.0 - p)) if 0.0 < p < 1.0 else (
        float("inf") if p == 0.0 else float("-inf"))


def _clopper_pearson(x: int, n: int, alpha: float) -> tuple[float, float]:
    """Exact binomial CI on the proportion (Clopper-Pearson)."""
    lo = 0.0 if x == 0 else float(stats.beta.ppf(alpha / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(stats.beta.ppf(1 - alpha / 2, x + 1, n - x))
    return lo, hi


def _poisson_exact(x: float, exposure: float, alpha: float) -> tuple[float, float]:
    """Exact Poisson CI on the per-unit rate (Garwood)."""
    lo = 0.0 if x == 0 else float(stats.chi2.ppf(alpha / 2, 2 * x) / 2.0 / exposure)
    hi = float(stats.chi2.ppf(1 - alpha / 2, 2 * (x + 1)) / 2.0 / exposure)
    return lo, hi


@dataclass(frozen=True, repr=False)
class ProcessSigmaResult(QCResult):
    """Attributes-capability result (immutable). Z is reported on two bases."""

    kind: str
    defects: float
    units: float
    opportunities: float
    dpu: float
    dpmo: float
    fty: float
    rty: float | None
    z_lt: float
    z_st: float
    shift: float
    z_lt_ci: tuple[float, float]
    z_st_ci: tuple[float, float]
    dpmo_ci: tuple[float, float]
    p_hat: float
    n: int
    zero_defect: bool = False
    one_sided_bound: float | None = None
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"Attributes capability ({self.kind}): process sigma"

    def _summary_lines(self) -> list[str]:
        lo, hi = self.dpmo_ci
        zlo, zhi = self.z_st_ci
        lines = [
            f"units = {self.units:g}   opportunities/unit = {self.opportunities:g}   "
            f"defects = {self.defects:g}",
            f"DPU = {self.dpu:.4g}   DPMO = {self.dpmo:.6g}   "
            f"({lo:.4g} to {hi:.6g}, exact {int((1-0.05)*100)}% CI)",
            f"first-time yield = {self.fty:.4%}"
            + (f"   rolled throughput yield = {self.rty:.4%}" if self.rty is not None else ""),
            "",
        ]
        if self.zero_defect:
            lines.append(
                f"zero defects observed: the point sigma is unbounded, so mfgQC reports the "
                f"one-sided exact upper bound on the rate (DPMO <= {self.dpmo_ci[1]:.4g}), "
                f"which gives Z.st >= {self.z_st_ci[0]:.3g}. Collect more units to tighten it.")
        else:
            lines.append(
                f"Z.lt (long-term, from the observed rate) = {self.z_lt:.3g}   "
                f"[{self.z_lt_ci[0]:.3g}, {self.z_lt_ci[1]:.3g}]")
            lines.append(
                f"Z.st (short-term sigma level)            = {self.z_st:.3g}   "
                f"[{zlo:.3g}, {zhi:.3g}]")
        lines.append("")
        lines.append(f"BASIS: Z.st = Z.lt + {self.shift} sigma. The {self.shift} sigma shift is a "
                     "CONVENTION linking long- and short-term, not a measured quantity; the "
                     '"sigma level" is the short-term basis.')
        return lines

    def summary(self) -> dict:
        return {"kind": self.kind, "dpu": self.dpu, "dpmo": self.dpmo, "fty": self.fty,
                "rty": self.rty, "z_lt": self.z_lt, "z_st": self.z_st, "shift": self.shift,
                "dpmo_ci_low": self.dpmo_ci[0], "dpmo_ci_high": self.dpmo_ci[1],
                "z_st_ci_low": self.z_st_ci[0], "z_st_ci_high": self.z_st_ci[1],
                "p_hat": self.p_hat, "n": self.n, "zero_defect": self.zero_defect}

    def _render_standalone(self, fig, kind, **kwargs):
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        from . import palette as _pal
        pal = _pal.active()
        zlo, zhi = self.z_st_ci
        ax.errorbar([0], [self.z_st], yerr=[[self.z_st - zlo], [zhi - self.z_st]],
                    fmt="o", color=pal.data, capsize=6, label="Z.st (sigma level)")
        ax.scatter([0], [self.z_lt], color=pal.amber, zorder=5, label="Z.lt (long-term)")
        for y in (3, 4, 5, 6):
            ax.axhline(y, color=pal.grid if hasattr(pal, "grid") else pal.limit, lw=0.5, ls=":")
        ax.set_xticks([])
        ax.set_ylabel("Z (sigma)")
        ax.set_title(self._title())
        ax.legend(loc="best", fontsize=8)
        return ax


def _adequacy(n: int, dpmo_ci: tuple[float, float], zero_defect: bool) -> AssumptionCheck:
    """Flag a rate estimate too unstable to trust: a wide exact CI or zero defects."""
    width = dpmo_ci[1] - dpmo_ci[0]
    rel = "low_power" if (n < 50 or zero_defect) else "ok"
    passed = not (zero_defect or n < 30)
    rec = None
    if not passed:
        rec = (f"The rate estimate is unstable (n={n}"
               + (", zero defects" if zero_defect else "")
               + f"); the exact CI spans {width:.4g} DPMO. Treat the sigma as provisional and "
               "collect more units before quoting a single number.")
    return AssumptionCheck("rate_stability", "exact CI width / n", float(width), None,
                           bool(passed), float(width), "DPMO CI width", rel, n, rec)


def compute(defects: float, units: float, opportunities: float = 1,
            kind: str = "defectives", *, alpha: float = 0.05) -> ProcessSigmaResult:
    """Attributes capability from defect counts. See module docstring.

    Parameters
    ----------
    defects : float
        Number of defects (``kind='defects'``) or defective units
        (``kind='defectives'``).
    units : float
        Number of units inspected.
    opportunities : float
        Defect opportunities per unit (defects kind). Default 1.
    kind : str
        ``'defectives'`` (binomial, pass/fail) or ``'defects'`` (Poisson, counts).
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}; got {kind!r}.")
    if units <= 0:
        raise ValueError("units must be positive.")
    if defects < 0:
        raise ValueError("defects must be non-negative.")
    n = int(round(units))

    if kind == "defectives":
        if defects > units:
            raise ValueError("defective units cannot exceed units inspected.")
        opportunities = 1
        p_hat = defects / units
        dpu = p_hat
        dpmo = p_hat * 1e6
        fty = 1.0 - p_hat
        rty = None
        rate_lo, rate_hi = _clopper_pearson(int(round(defects)), n, alpha)
    else:  # defects (Poisson)
        exposure = units * opportunities
        dpu = defects / units
        rate_per_opp = defects / exposure
        dpmo = rate_per_opp * 1e6
        p_hat = rate_per_opp
        fty = float(np.exp(-dpu))            # first-time yield, Poisson
        rty = float(np.exp(-dpu))            # single process step: RTY == FTY here
        rlo, rhi = _poisson_exact(defects, exposure, alpha)
        rate_lo, rate_hi = rlo, rhi

    zero_defect = defects == 0
    z_lt = _z_from_rate(p_hat)
    z_st = z_lt + SHIFT
    # CI endpoints: the higher rate is the WORSE (lower Z) end.
    z_lt_hi = _z_from_rate(rate_lo)          # better rate -> higher Z
    z_lt_lo = _z_from_rate(rate_hi)          # worse rate -> lower Z
    z_lt_ci = (z_lt_lo, z_lt_hi)
    z_st_ci = (z_lt_lo + SHIFT, z_lt_hi + SHIFT)
    dpmo_ci = (rate_lo * 1e6, rate_hi * 1e6)
    one_sided = rate_hi if zero_defect else None
    if zero_defect:
        # report the bound-based Z.st lower end as the headline; point Z is unbounded.
        z_lt = z_lt_lo
        z_st = z_lt_lo + SHIFT

    checks = [_adequacy(n, dpmo_ci, zero_defect)]
    step = Step(operation="process_sigma",
                params={"kind": kind, "defects": defects, "units": units,
                        "opportunities": opportunities, "dpmo": dpmo, "z_st": z_st},
                n_affected=n, timestamp=_now())
    return ProcessSigmaResult(
        kind=kind, defects=float(defects), units=float(units),
        opportunities=float(opportunities), dpu=float(dpu), dpmo=float(dpmo),
        fty=float(fty), rty=rty, z_lt=float(z_lt), z_st=float(z_st), shift=SHIFT,
        z_lt_ci=z_lt_ci, z_st_ci=z_st_ci, dpmo_ci=dpmo_ci, p_hat=float(p_hat),
        n=n, zero_defect=zero_defect, one_sided_bound=one_sided,
        assumptions=checks, history=(step,))
