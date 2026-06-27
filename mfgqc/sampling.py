"""Attribute acceptance-sampling plans (single sampling, by attributes).

This module builds and analyses single acceptance-sampling plans defined by a
sample size ``n`` and an acceptance number ``c``: inspect ``n`` units, accept the
lot if the number of defectives is ``<= c``, otherwise reject.

Three things are surfaced as guardrails, in the flags-v2 spirit (a binary verdict
plus adjacent context, never a silent decision):

1. The Pa MODEL choice and WHY. The default is the binomial. When a finite
   ``lot_size`` is given and the sampling fraction ``n/N`` exceeds 0.1, the
   hypergeometric model is selected automatically and the reason is reported.
   The Poisson model is only used when explicitly requested.
2. A flag when ``n/N`` is large enough that the binomial approximation to a finite
   lot is questionable.
3. The stated, non-data-checkable assumptions (random sampling, lot homogeneity,
   binary good/defective classification) are noted in the report text.

The operating-characteristic (OC) curve gives the probability of acceptance
``Pa(p)`` as a function of the incoming defective fraction ``p``. From it the
standard risk points are derived: the AQL (``Pa = 0.95``), the LTPD/RQL
(``Pa = 0.10``), the indifference quality (``Pa = 0.50``), the producer's risk
``alpha = 1 - Pa(AQL)`` and the consumer's risk ``beta = Pa(LTPD)``.

Correctness anchors (verified against scipy):
- Binomial OC, n=134, c=3: Pa(0.01)=0.9537, Pa(0.05)=0.0931.
- Derived risk points n=134,c=3: AQL=1.03%, LTPD=4.92%, indifference=2.73%.
- N=500, n=134, c=3, p=0.02: binomial Pa=0.719, hypergeometric Pa=0.734.
"""

from __future__ import annotations

from . import palette as _pal

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import numpy as np
from scipy import optimize, stats

from ._result import QCResult
from .assumptions import AssumptionCheck
from .data import Step

_VALID_MODELS = ("binomial", "hypergeometric", "poisson")

# Sampling fraction above which the binomial approximation to a finite lot is
# considered questionable and the hypergeometric model is preferred.
_FRACTION_THRESHOLD = 0.1


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Probability of acceptance
# --------------------------------------------------------------------------- #
def _pa(p: float, n: int, c: int, model: str, lot_size: int | None) -> float:
    """Probability of accepting a lot of incoming fraction defective ``p``.

    binomial:       P(X<=c), X ~ Binom(n, p)
    hypergeometric: P(X<=c), X ~ Hypergeom(N, D=round(N*p), n)
    poisson:        P(X<=c), X ~ Poisson(n*p)
    """
    p = float(np.clip(p, 0.0, 1.0))
    if model == "binomial":
        return float(stats.binom.cdf(c, n, p))
    if model == "poisson":
        return float(stats.poisson.cdf(c, n * p))
    if model == "hypergeometric":
        if lot_size is None:
            raise ValueError("hypergeometric model requires a lot_size.")
        N = int(lot_size)
        D = int(round(N * p))
        return float(stats.hypergeom.cdf(c, N, D, n))
    raise ValueError(f"model must be one of {_VALID_MODELS}; got {model!r}.")


def _pa_vec(p_grid: np.ndarray, n: int, c: int, model: str, lot_size: int | None) -> np.ndarray:
    return np.array([_pa(float(p), n, c, model, lot_size) for p in p_grid], dtype=float)


def _invert_pa(target: float, n: int, c: int, model: str, lot_size: int | None) -> float:
    """Return the incoming fraction defective ``p`` at which ``Pa(p) == target``.

    ``Pa`` is monotone decreasing in ``p``, so a bracketed root-find is robust.
    """
    f = lambda p: _pa(p, n, c, model, lot_size) - target
    lo, hi = 1e-12, 1.0 - 1e-12
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:  # degenerate (e.g. c >= n); clamp to the nearer end
        return 0.0 if abs(flo) < abs(fhi) else 1.0
    return float(optimize.brentq(f, lo, hi, xtol=1e-12, rtol=1e-12))


# --------------------------------------------------------------------------- #
# Model selection guardrail
# --------------------------------------------------------------------------- #
def _select_model(n: int, lot_size: int | None, model: str | None
                  ) -> tuple[str, list[AssumptionCheck]]:
    """Choose the Pa model and surface the choice + reasoning (flags v2)."""
    checks: list[AssumptionCheck] = []
    fraction = (n / lot_size) if lot_size else 0.0

    if model is not None:
        if model not in _VALID_MODELS:
            raise ValueError(f"model must be one of {_VALID_MODELS}; got {model!r}.")
        chosen = model
        reason = f"model explicitly requested ('{model}')"
    elif lot_size is not None and fraction > _FRACTION_THRESHOLD:
        chosen = "hypergeometric"
        reason = (f"lot_size given and n/N={fraction:.2f} > {_FRACTION_THRESHOLD} "
                  "-> hypergeometric (finite-lot exact)")
    else:
        chosen = "binomial"
        if lot_size is not None:
            reason = f"lot_size given but n/N={fraction:.2f} <= {_FRACTION_THRESHOLD} -> binomial"
        else:
            reason = "no lot_size -> binomial (infinite-lot default)"

    # 1) Surface the model choice itself.
    checks.append(AssumptionCheck(
        name="model_choice", test="Pa model selection",
        statistic=float(fraction), p_value=None,
        passed=True, magnitude=float(fraction), magnitude_label="sampling fraction n/N",
        reliability="ok", n=int(n), recommendation=reason,
    ))

    # 2) Flag when the binomial approximation to a finite lot is questionable.
    if chosen == "binomial" and lot_size is not None and fraction > _FRACTION_THRESHOLD:
        checks.append(AssumptionCheck(
            name="binomial_approximation", test="sampling fraction n/N <= 0.1",
            statistic=float(fraction), p_value=None,
            passed=False, magnitude=float(fraction), magnitude_label="sampling fraction n/N",
            reliability="ok", n=int(n),
            recommendation=(f"n/N={fraction:.2f} > {_FRACTION_THRESHOLD}: the binomial "
                            "approximation to this finite lot is questionable; consider "
                            "model='hypergeometric'."),
        ))

    return chosen, checks


_STATED_ASSUMPTIONS = (
    "Stated assumptions (not data-checkable): random sampling of the lot, "
    "lot homogeneity, and binary good/defective classification of each unit."
)


# --------------------------------------------------------------------------- #
# Result objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class OCCurveResult(QCResult):
    """Operating-characteristic curve: Pa(p) over a grid, with risk points marked."""

    p_grid: np.ndarray
    pa: np.ndarray
    n: int
    c: int
    model: str
    lot_size: int | None
    aql: float
    ltpd: float
    indifference: float
    alpha: float
    beta: float
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"OC Curve (n={self.n}, c={self.c}, model={self.model})"

    def _summary_lines(self) -> list[str]:
        return [
            f"n = {self.n}   c = {self.c}   model = {self.model}",
            f"AQL (Pa=0.95)          = {self.aql * 100:.3g}%",
            f"Indifference (Pa=0.50) = {self.indifference * 100:.3g}%",
            f"LTPD/RQL (Pa=0.10)     = {self.ltpd * 100:.3g}%",
            f"Producer risk alpha    = {self.alpha:.3g}",
            f"Consumer risk beta     = {self.beta:.3g}",
            "",
            _STATED_ASSUMPTIONS,
        ]

    def summary(self) -> dict:
        return {
            "n": self.n, "c": self.c, "model": self.model,
            "aql_pct": round(self.aql * 100, 4),
            "indifference_pct": round(self.indifference * 100, 4),
            "ltpd_pct": round(self.ltpd * 100, 4),
            "alpha": round(self.alpha, 4), "beta": round(self.beta, 4),
        }

    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        self._render_axes(ax, kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        ax.plot(self.p_grid * 100, self.pa, color=_pal.active().center, lw=2, label="Pa(p)")
        # Producer's risk region (good quality, around the AQL) and consumer's
        # risk region (bad quality, around the LTPD).
        ax.axvspan(0, self.aql * 100, color=_pal.active().target, alpha=0.08,
                   label="producer-risk region")
        ax.axvspan(self.ltpd * 100, float(self.p_grid.max()) * 100, color=_pal.active().ooc,
                   alpha=0.08, label="consumer-risk region")
        for val, name, color in (
            (self.aql, f"AQL ({self.aql * 100:.2f}%)", _pal.active().target),
            (self.indifference, f"Indiff ({self.indifference * 100:.2f}%)", _pal.active().muted),
            (self.ltpd, f"LTPD ({self.ltpd * 100:.2f}%)", _pal.active().ooc),
        ):
            ax.axvline(val * 100, color=color, ls="--", lw=1.2)
        ax.axhline(0.95, color=_pal.active().target, ls=":", lw=0.8)
        ax.axhline(0.10, color=_pal.active().ooc, ls=":", lw=0.8)
        ax.set_xlabel("Incoming fraction defective p (%)")
        ax.set_ylabel("Probability of acceptance  Pa")
        ax.set_ylim(0, 1.02)
        ax.set_title(self._title())
        ax.legend(loc="upper right", fontsize=8)


@dataclass(frozen=True, repr=False)
class AOQResult(QCResult):
    """Average-outgoing-quality curve (rectifying inspection) with the AOQL marked."""

    p_grid: np.ndarray
    aoq: np.ndarray
    aoql: float
    aoql_at_p: float
    n: int
    c: int
    model: str
    lot_size: int | None
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"AOQ Curve (n={self.n}, c={self.c}, model={self.model})"

    def _summary_lines(self) -> list[str]:
        finite = "rectifying, finite lot" if self.lot_size else "infinite lot"
        return [
            f"n = {self.n}   c = {self.c}   model = {self.model}   ({finite})",
            f"AOQL (max average outgoing quality) = {self.aoql * 100:.3g}%",
            f"  occurs at incoming p = {self.aoql_at_p * 100:.3g}%",
        ]

    def summary(self) -> dict:
        return {
            "n": self.n, "c": self.c, "model": self.model,
            "aoql_pct": round(self.aoql * 100, 4),
            "aoql_at_p_pct": round(self.aoql_at_p * 100, 4),
        }

    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        self._render_axes(ax, kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        ax.plot(self.p_grid * 100, self.aoq * 100, color=_pal.active().center, lw=2, label="AOQ(p)")
        ax.axhline(self.aoql * 100, color=_pal.active().ooc, ls="--", lw=1.2,
                   label=f"AOQL = {self.aoql * 100:.2f}%")
        ax.axvline(self.aoql_at_p * 100, color=_pal.active().ooc, ls=":", lw=0.8)
        ax.set_xlabel("Incoming fraction defective p (%)")
        ax.set_ylabel("Average outgoing quality (%)")
        ax.set_title(self._title())
        ax.legend(loc="upper right", fontsize=8)


@dataclass(frozen=True, repr=False)
class LotDisposition(QCResult):
    """Disposition of a single inspected lot: accept/reject and the basis."""

    decision: str
    defectives_found: int
    n: int
    c: int
    observed_rate: float
    pa_at_observed: float
    model: str
    lot_size: int | None
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"Lot Disposition: {self.decision.upper()}"

    def _summary_lines(self) -> list[str]:
        return [
            f"decision   = {self.decision}",
            f"defectives = {self.defectives_found}  (acceptance number c = {self.c})",
            f"n          = {self.n}",
            f"observed fraction defective = {self.observed_rate * 100:.3g}%",
            f"Pa at observed rate         = {self.pa_at_observed:.3g}",
        ]

    def summary(self) -> dict:
        return {
            "decision": self.decision,
            "found": self.defectives_found,
            "limit": self.c,
            "n": self.n,
            "observed_rate": round(self.observed_rate, 6),
            "pa_at_observed": round(self.pa_at_observed, 6),
        }

    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        self._render_axes(ax, kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        color = _pal.active().target if self.decision == "accept" else _pal.active().ooc
        ax.bar(["defectives found", "acceptance number c"],
               [self.defectives_found, self.c],
               color=[color, _pal.active().muted])
        ax.set_ylabel("count")
        ax.set_title(self._title())


@dataclass(frozen=True, repr=False)
class SamplingPlan(QCResult):
    """A single attribute acceptance-sampling plan (n, c) with derived risk points."""

    n: int
    c: int
    model: str
    lot_size: int | None
    aql: float
    ltpd: float
    alpha: float
    beta: float
    indifference_point: float
    # Optional provenance for inverse / standard plans.
    requested_aql: float | None = None
    requested_ltpd: float | None = None
    source: str | None = None
    code_letter: str | None = None
    re: int | None = None  # rejection number (Z1.4); accept number is c (= Ac)
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    # ---- derived curves & inspection ------------------------------------
    def _p_grid(self) -> np.ndarray:
        upper = max(self.ltpd * 2.0, 0.02)
        upper = float(min(upper, 1.0))
        return np.linspace(1e-4, upper, 300)

    def oc_curve(self) -> OCCurveResult:
        """Operating-characteristic curve for this plan."""
        grid = self._p_grid()
        pa = _pa_vec(grid, self.n, self.c, self.model, self.lot_size)
        step = Step(operation="oc_curve",
                    params={"n": self.n, "c": self.c, "model": self.model},
                    n_affected=None, timestamp=_now())
        return OCCurveResult(
            p_grid=grid, pa=pa, n=self.n, c=self.c, model=self.model,
            lot_size=self.lot_size, aql=self.aql, ltpd=self.ltpd,
            indifference=self.indifference_point, alpha=self.alpha, beta=self.beta,
            assumptions=list(self.assumptions), history=self.history + (step,),
        )

    def aoq_curve(self) -> AOQResult:
        """Average-outgoing-quality curve (rectifying inspection)."""
        grid = self._p_grid()
        pa = _pa_vec(grid, self.n, self.c, self.model, self.lot_size)
        if self.lot_size:
            factor = (self.lot_size - self.n) / self.lot_size
        else:
            factor = 1.0
        aoq = grid * pa * factor
        idx = int(np.argmax(aoq))
        step = Step(operation="aoq_curve",
                    params={"n": self.n, "c": self.c, "model": self.model,
                            "lot_size": self.lot_size},
                    n_affected=None, timestamp=_now())
        return AOQResult(
            p_grid=grid, aoq=aoq, aoql=float(aoq[idx]), aoql_at_p=float(grid[idx]),
            n=self.n, c=self.c, model=self.model, lot_size=self.lot_size,
            assumptions=list(self.assumptions), history=self.history + (step,),
        )

    def inspect(self, defectives_found: int) -> LotDisposition:
        """Disposition a lot given the number of defectives found in the sample."""
        d = int(defectives_found)
        if d < 0:
            raise ValueError("defectives_found must be non-negative.")
        decision = "accept" if d <= self.c else "reject"
        observed_rate = d / self.n if self.n else 0.0
        pa_at_observed = _pa(observed_rate, self.n, self.c, self.model, self.lot_size)
        step = Step(operation="inspect",
                    params={"defectives_found": d, "c": self.c, "decision": decision},
                    n_affected=self.n, timestamp=_now())
        return LotDisposition(
            decision=decision, defectives_found=d, n=self.n, c=self.c,
            observed_rate=observed_rate, pa_at_observed=pa_at_observed,
            model=self.model, lot_size=self.lot_size,
            assumptions=list(self.assumptions), history=self.history + (step,),
        )

    # ---- reporting -------------------------------------------------------
    def _title(self) -> str:
        src = f" [{self.source}]" if self.source else ""
        return f"Sampling Plan n={self.n}, c={self.c}{src}"

    def _summary_lines(self) -> list[str]:
        lines = [
            f"n = {self.n}   c = {self.c}   model = {self.model}",
        ]
        if self.lot_size is not None:
            lines.append(f"lot size N = {self.lot_size}   (n/N = {self.n / self.lot_size:.3g})")
        if self.code_letter is not None:
            lines.append(f"code letter = {self.code_letter}   Ac = {self.c}   Re = {self.re}")
        lines.extend([
            "",
            f"AQL (Pa=0.95)          = {self.aql * 100:.3g}%",
            f"Indifference (Pa=0.50) = {self.indifference_point * 100:.3g}%",
            f"LTPD/RQL (Pa=0.10)     = {self.ltpd * 100:.3g}%",
            f"Producer risk alpha    = {self.alpha:.3g}",
            f"Consumer risk beta     = {self.beta:.3g}",
        ])
        if self.requested_aql is not None or self.requested_ltpd is not None:
            lines.append("")
            lines.append("Requested vs achieved:")
            if self.requested_aql is not None:
                lines.append(f"  AQL  requested {self.requested_aql * 100:.3g}% "
                             f"-> achieved {self.aql * 100:.3g}%")
            if self.requested_ltpd is not None:
                lines.append(f"  LTPD requested {self.requested_ltpd * 100:.3g}% "
                             f"-> achieved {self.ltpd * 100:.3g}%")
        lines.extend(["", _STATED_ASSUMPTIONS])
        return lines

    def summary(self) -> dict:
        out = {
            "n": self.n, "c": self.c, "model": self.model,
            "lot_size": self.lot_size,
            "aql_pct": round(self.aql * 100, 4),
            "indifference_pct": round(self.indifference_point * 100, 4),
            "ltpd_pct": round(self.ltpd * 100, 4),
            "alpha": round(self.alpha, 4), "beta": round(self.beta, 4),
        }
        if self.code_letter is not None:
            out["code_letter"] = self.code_letter
            out["Ac"] = self.c
            out["Re"] = self.re
        return out

    def _render_standalone(self, fig, kind, **kwargs):
        # Delegate to the OC curve (the canonical view of a plan).
        self.oc_curve()._render_standalone(fig, kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        self.oc_curve()._render_axes(ax, kind, **kwargs)


# --------------------------------------------------------------------------- #
# Plan builders
# --------------------------------------------------------------------------- #
def _build_plan(n: int, c: int, model: str, lot_size: int | None,
                checks: list[AssumptionCheck], *, source: str | None = None,
                requested_aql: float | None = None, requested_ltpd: float | None = None,
                code_letter: str | None = None, re: int | None = None,
                history: tuple[Step, ...] = ()) -> SamplingPlan:
    aql = _invert_pa(0.95, n, c, model, lot_size)
    ltpd = _invert_pa(0.10, n, c, model, lot_size)
    indiff = _invert_pa(0.50, n, c, model, lot_size)
    alpha = 1.0 - _pa(aql, n, c, model, lot_size)
    beta = _pa(ltpd, n, c, model, lot_size)
    return SamplingPlan(
        n=n, c=c, model=model, lot_size=lot_size,
        aql=aql, ltpd=ltpd, alpha=alpha, beta=beta, indifference_point=indiff,
        requested_aql=requested_aql, requested_ltpd=requested_ltpd,
        source=source, code_letter=code_letter, re=re,
        assumptions=checks, history=history,
    )


def sampling_plan(n: int, c: int, *, lot_size: int | None = None,
                  model: str | None = None) -> SamplingPlan:
    """Build a single attribute sampling plan from a sample size and acceptance number.

    Parameters
    ----------
    n : int
        Sample size.
    c : int
        Acceptance number (accept if defectives <= c).
    lot_size : int or None, optional
        Finite lot size N. When given and ``n/N > 0.1`` the hypergeometric model
        is selected automatically (the choice is surfaced in ``assumptions``).
    model : {"binomial", "hypergeometric", "poisson"} or None, optional
        Force a Pa model. ``None`` (default) auto-selects: binomial, or
        hypergeometric when ``lot_size`` is given and ``n/N > 0.1``. Poisson is
        only used when explicitly requested.

    Returns
    -------
    SamplingPlan
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    if c < 0:
        raise ValueError("c must be >= 0.")
    if lot_size is not None and n > lot_size:
        raise ValueError(f"n ({n}) cannot exceed lot_size ({lot_size}).")
    chosen, checks = _select_model(n, lot_size, model)
    step = Step(operation="sampling_plan",
                params={"n": n, "c": c, "model": chosen, "lot_size": lot_size},
                n_affected=None, timestamp=_now())
    return _build_plan(n, c, chosen, lot_size, checks, source="single", history=(step,))


def find_plan(aql: float, ltpd: float, *, alpha: float = 0.05, beta: float = 0.10,
              lot_size: int | None = None) -> SamplingPlan:
    """Find the smallest single plan meeting both risk points.

    Searches for the plan ``(n, c)`` with the SMALLEST ``n`` whose OC curve gives
    ``Pa(aql) >= 1 - alpha`` (producer protected) AND ``Pa(ltpd) <= beta``
    (consumer protected). For each acceptance number ``c = 0, 1, 2, ...`` the
    n-range satisfying both constraints is found; the overall minimum ``n`` wins.

    Because integer ``(n, c)`` rarely hits the requested AQL/LTPD exactly, the
    ACHIEVED risk points are reported on the returned plan (``requested_aql`` /
    ``requested_ltpd`` record what was asked for).

    Parameters
    ----------
    aql, ltpd : float
        Acceptable and rejectable (tolerable) fraction defective, as fractions.
    alpha, beta : float, optional
        Producer's and consumer's risk (defaults 0.05, 0.10).
    lot_size : int or None, optional
        Finite lot size; model is auto-selected as in :func:`sampling_plan`.

    Returns
    -------
    SamplingPlan
    """
    if not (0 < aql < ltpd < 1):
        raise ValueError("require 0 < aql < ltpd < 1.")
    if not (0 < alpha < 1 and 0 < beta < 1):
        raise ValueError("alpha and beta must be in (0, 1).")

    # Model selection depends on n, which we don't know yet; pick per-candidate.
    best: tuple[int, int] | None = None
    max_c = 60
    for c in range(0, max_c + 1):
        found_n: int | None = None
        n = c + 1
        # Pa(aql) decreases and Pa(ltpd) decreases as n grows. The producer
        # constraint Pa(aql) >= 1-alpha sets an UPPER bound on n; the consumer
        # constraint Pa(ltpd) <= beta sets a LOWER bound. Scan up for the first n
        # meeting the consumer constraint, check the producer one.
        n_cap = best[0] if best is not None else 100000
        while n <= n_cap:
            model = _pick_model(n, lot_size)
            pa_ltpd = _pa(ltpd, n, c, model, lot_size)
            if pa_ltpd <= beta:
                pa_aql = _pa(aql, n, c, model, lot_size)
                if pa_aql >= 1 - alpha:
                    found_n = n
                break
            n += 1
        if found_n is not None and (best is None or found_n < best[0]):
            best = (found_n, c)

    if best is None:
        raise ValueError(
            f"no plan found for aql={aql}, ltpd={ltpd}, alpha={alpha}, beta={beta} "
            f"within c<={max_c}.")

    n, c = best
    chosen, checks = _select_model(n, lot_size, None)
    step = Step(operation="find_plan",
                params={"aql": aql, "ltpd": ltpd, "alpha": alpha, "beta": beta,
                        "n": n, "c": c, "model": chosen},
                n_affected=None, timestamp=_now())
    return _build_plan(n, c, chosen, lot_size, checks, source="find_plan",
                       requested_aql=aql, requested_ltpd=ltpd, history=(step,))


def _pick_model(n: int, lot_size: int | None) -> str:
    """Auto model rule, no side effects (used inside the search loop)."""
    if lot_size is not None and (n / lot_size) > _FRACTION_THRESHOLD:
        return "hypergeometric"
    return "binomial"


# --------------------------------------------------------------------------- #
# ANSI/ASQ Z1.4 single-sampling, general inspection level II, normal severity
# --------------------------------------------------------------------------- #
# Lot-size -> sample-size code letter (General Inspection Level II).
# (low, high inclusive) -> letter
_Z14_CODE_LETTERS = [
    (2, 8, "A"),
    (9, 15, "B"),
    (16, 25, "C"),
    (26, 50, "D"),
    (51, 90, "E"),
    (91, 150, "F"),
    (151, 280, "G"),
    (281, 500, "H"),
    (501, 1200, "J"),
    (1201, 3200, "K"),
    (3201, 10000, "L"),
    (10001, 35000, "M"),
    (35001, 150000, "N"),
    (150001, 500000, "P"),
    (500001, 10**12, "Q"),
]

# Code letter -> sample size n (single sampling), in ascending order.
_Z14_SAMPLE_SIZE = {
    "A": 2, "B": 3, "C": 5, "D": 8, "E": 13, "F": 20, "G": 32, "H": 50,
    "J": 80, "K": 125, "L": 200, "M": 315, "N": 500, "P": 800, "Q": 1250,
}
_Z14_LETTERS_ORDER = list(_Z14_SAMPLE_SIZE.keys())  # A..Q (I and O are skipped)

# Tabled AQLs (in percent defective).
_Z14_AQLS = [0.10, 0.15, 0.25, 0.40, 0.65, 1.0, 1.5, 2.5, 4.0, 6.5, 10.0]

# The single acceptance-number "staircase" that the published Z1.4 normal master
# table runs along its diagonals. Every on-table (Ac, Re) cell is read off this
# sequence; arrows (down/up) move to an adjacent code letter.
_Z14_AC_SEQUENCE = [0, 1, 2, 3, 5, 7, 10, 14, 21]


def _z14_cell(letter: str, aql: float):
    """Faithful Z1.4 single-sampling NORMAL (Ac, Re) for a (code letter, AQL).

    The published master table is a staircase: with letters indexed A=0,B=1,...
    (I/O skipped) and AQL columns indexed 0.10=0..10.0=10, the position on the
    acceptance-number sequence is ``p = col_index + letter_index - 11`` (anchored
    on code K / AQL 1.0 -> Ac=3). ``0 <= p <= 8`` is on-table; ``p < 0`` is a
    down-arrow (use the first plan below, a LARGER sample); ``p > 8`` is an
    up-arrow (use the first plan above, a SMALLER sample).

    Reproduces the published cells exactly, e.g. code K: 0.65->2, 1.0->3, 1.5->5,
    2.5->7, 4.0->10, 6.5->14, 10->21.
    """
    li = _Z14_LETTERS_ORDER.index(letter)
    ci = _Z14_AQLS.index(aql)
    p = ci + li - 11
    if 0 <= p <= len(_Z14_AC_SEQUENCE) - 1:
        ac = _Z14_AC_SEQUENCE[p]
        return ("ok", (ac, ac + 1))
    return ("down", None) if p < 0 else ("up", None)


def _code_letter(lot_size: int, level: str) -> str:
    if level != "II":
        raise NotImplementedError(
            f"only general inspection level II is tabled; got level={level!r}.")
    if lot_size < 2:
        raise ValueError("lot_size must be >= 2 for a Z1.4 plan.")
    for lo, hi, letter in _Z14_CODE_LETTERS:
        if lo <= lot_size <= hi:
            return letter
    raise ValueError(f"lot_size {lot_size} is out of the tabled range.")


def _resolve_z14(letter: str, aql: float) -> tuple[str, tuple[int, int]]:
    """Resolve a (letter, aql) cell, following Z1.4 arrows to the adjacent row.

    A down-arrow takes the first plan BELOW (larger code letter / sample); an
    up-arrow takes the first plan ABOVE (smaller). The resolved row supplies both
    the sample size and the (Ac, Re).
    """
    if letter not in _Z14_SAMPLE_SIZE:
        raise NotImplementedError(f"code letter {letter!r} is not tabled.")
    if aql not in _Z14_AQLS:
        raise NotImplementedError(f"AQL {aql} not tabled; available: {_Z14_AQLS}.")

    start = _Z14_LETTERS_ORDER.index(letter)
    status, plan = _z14_cell(letter, aql)
    if status == "ok":
        return letter, plan
    step = 1 if status == "down" else -1  # down-arrow -> larger letter
    i = start + step
    while 0 <= i < len(_Z14_LETTERS_ORDER):
        cand = _Z14_LETTERS_ORDER[i]
        s, plan = _z14_cell(cand, aql)
        if s == "ok":
            return cand, plan
        i += step
    raise NotImplementedError(f"no resolvable plan for letter {letter}, AQL {aql}.")


def z14_plan(lot_size: int, aql: float, *, level: str = "II",
             severity: str = "normal") -> SamplingPlan:
    """ANSI/ASQ Z1.4 single-sampling plan lookup (general inspection level II).

    Parameters
    ----------
    lot_size : int
        Lot or batch size N.
    aql : float
        Acceptable quality level, expressed in PERCENT defective (e.g. ``0.65``,
        ``1.0``, ``2.5``, ``4.0``).
    level : str, optional
        General inspection level. Only ``"II"`` is tabled.
    severity : {"normal", "tightened", "reduced"}, optional
        Only ``"normal"`` is tabled; the others raise ``NotImplementedError``.

    Returns
    -------
    SamplingPlan
        With ``c == Ac`` (acceptance number), ``re`` the rejection number, the
        sample-size ``code_letter``, and the OC-derived risk points.
    """
    if severity != "normal":
        raise NotImplementedError(
            f"only severity='normal' is tabled; got {severity!r}. "
            "(Tightened/reduced master tables are not encoded.)")

    letter = _code_letter(lot_size, level)
    resolved_letter, (ac, re) = _resolve_z14(letter, aql)
    n = _Z14_SAMPLE_SIZE[resolved_letter]

    # Z1.4 plans assume a finite lot; the OC curve uses the binomial (the Z1.4
    # convention) unless the sampling fraction is large.
    chosen, checks = _select_model(n, lot_size, None)
    checks.append(AssumptionCheck(
        name="z14_lookup", test="ANSI/ASQ Z1.4 single-sampling normal",
        statistic=float(n), p_value=None, passed=True,
        magnitude=float(ac), magnitude_label="acceptance number Ac",
        reliability="ok", n=int(n),
        recommendation=(f"lot {lot_size} -> code {letter}"
                        + ("" if resolved_letter == letter
                           else f" (arrow -> {resolved_letter})")
                        + f"; n={n}; AQL {aql}% -> Ac={ac}, Re={re}."),
    ))
    step = Step(operation="z14_plan",
                params={"lot_size": lot_size, "aql": aql, "level": level,
                        "severity": severity, "code_letter": resolved_letter,
                        "n": n, "Ac": ac, "Re": re},
                n_affected=None, timestamp=_now())
    return _build_plan(
        n, ac, chosen, lot_size, checks, source="Z1.4 normal",
        requested_aql=aql / 100.0, code_letter=resolved_letter, re=re, history=(step,),
    )


# =========================================================================== #
# Z1.9 — variables acceptance sampling (standard-deviation method, sigma unknown)
# =========================================================================== #
# Bridges acceptance sampling <-> capability: a lot is judged from the sample
# mean and SD via the quality index Q = (limit - xbar)/s and the estimated
# percent nonconforming. ASSUMES the characteristic is normal - that assumption
# is surfaced loudly, because the percent-nonconforming estimate is invalid
# otherwise (the mfgQC wedge).
#
# Inspection Level II lot-size -> sample-size code letter, and the SD-method
# sample size per letter (ANSI/ASQ Z1.9 / MIL-STD-414).
# Lot-size -> sample-size code letter (Inspection Level II). Anchored to the
# published Z1.9 cell: lot size 100 -> code letter F -> n = 10.
_Z19_CODE_LETTERS = [
    (2, 8, "B"), (9, 15, "C"), (16, 25, "D"), (26, 50, "E"), (51, 90, "E"),
    (91, 150, "F"), (151, 280, "G"), (281, 400, "H"), (401, 500, "I"),
    (501, 1200, "J"), (1201, 3200, "K"), (3201, 10000, "L"), (10001, 35000, "M"),
    (35001, 150000, "N"), (150001, 500000, "O"), (500001, 10**12, "P"),
]
_Z19_SAMPLE_SIZE = {
    "B": 3, "C": 4, "D": 5, "E": 7, "F": 10, "G": 15, "H": 20, "I": 25, "J": 30,
    "K": 40, "L": 50, "M": 75, "N": 100, "O": 150, "P": 200,
}


def _z19_code_letter(lot_size: int, level: str) -> str:
    if level != "II":
        raise NotImplementedError(f"only general inspection level II is tabled; got {level!r}.")
    if lot_size < 2:
        raise ValueError("lot_size must be >= 2 for a Z1.9 plan.")
    for lo, hi, letter in _Z19_CODE_LETTERS:
        if lo <= lot_size <= hi:
            return letter
    raise ValueError(f"lot_size {lot_size} is out of the tabled range.")


def _z19_k(n: int, aql: float, alpha: float = 0.05) -> float:
    """Form-1 acceptability constant via the standard normal-approximation design
    (the basis of the Z1.9 k table): accept if Q >= k where, at the AQL, the OC
    curve gives producer's risk ``alpha``. Reproduces the published normal-
    inspection k to ~0.02 for mid-range n; confirm exact cells against your Z1.9
    table copy for the smallest/largest code letters."""
    z_aql = float(stats.norm.ppf(1 - aql))      # quality index Q implied by the AQL
    z_a = float(stats.norm.ppf(1 - alpha))      # producer's risk
    return z_aql - z_a * np.sqrt(1.0 / n + z_aql ** 2 / (2 * n))


def _pct_nonconforming(q: float) -> float:
    """Normal-theory estimate of the fraction beyond a limit at quality index q."""
    return float(stats.norm.sf(q))


@dataclass(frozen=True, repr=False)
class Z19Disposition(QCResult):
    """Disposition of a lot under a Z1.9 variables plan (SD method)."""

    decision: str
    n: int
    xbar: float
    s: float
    lower: float | None
    upper: float | None
    QL: float | None
    QU: float | None
    k: float
    est_pct_lower: float | None
    est_pct_upper: float | None
    est_pct_total: float
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"Z1.9 Lot Disposition: {self.decision.upper()}"

    def _summary_lines(self) -> list[str]:
        lines = [
            f"n = {self.n}   xbar = {self.xbar:.5g}   s = {self.s:.4g}   k = {self.k:.3f}",
        ]
        if self.QL is not None:
            lines.append(f"QL = (xbar-LSL)/s = {self.QL:.3f}   est. % below LSL = {self.est_pct_lower*100:.3f}%")
        if self.QU is not None:
            lines.append(f"QU = (USL-xbar)/s = {self.QU:.3f}   est. % above USL = {self.est_pct_upper*100:.3f}%")
        lines.append(f"estimated total % nonconforming = {self.est_pct_total*100:.3f}%")
        lines.append(f"Decision: {self.decision} (Form 1: accept if every Q >= k = {self.k:.3f}).")
        return lines

    def summary(self) -> dict:
        return {
            "decision": self.decision, "n": self.n, "xbar": self.xbar, "s": self.s,
            "QL": self.QL, "QU": self.QU, "k": self.k,
            "est_pct_lower": self.est_pct_lower, "est_pct_upper": self.est_pct_upper,
            "est_pct_total": self.est_pct_total,
        }

    def _render_standalone(self, fig, kind, **kwargs):
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        import numpy as _np
        xs = _np.linspace(self.xbar - 4 * self.s, self.xbar + 4 * self.s, 200)
        ax.plot(xs, stats.norm.pdf(xs, self.xbar, self.s), color=_pal.active().center)
        for v, lab in ((self.lower, "LSL"), (self.upper, "USL")):
            if v is not None:
                ax.axvline(v, color=_pal.active().ooc, ls="--", lw=1.3, label=lab)
        ax.axvline(self.xbar, color=_pal.active().target, lw=1.2, label="xbar")
        ax.set_title(f"Z1.9 ({self.decision}) - fitted normal vs limits", fontsize=10)
        ax.legend(fontsize=8)
        return ax


@dataclass(frozen=True, repr=False)
class Z19Plan(QCResult):
    """A Z1.9 variables sampling plan (SD method, sigma unknown)."""

    n: int
    code_letter: str
    k: float
    M: float
    aql: float
    lot_size: int | None
    level: str
    severity: str
    lower: float | None = None
    upper: float | None = None
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"Z1.9 Variables Plan (code {self.code_letter})"

    def _summary_lines(self) -> list[str]:
        return [
            f"lot size {self.lot_size} -> code letter {self.code_letter}   "
            f"(level {self.level}, {self.severity})",
            f"n = {self.n}   AQL = {self.aql*100:.3g}%",
            f"Form 1 (k-method): accept if Q >= k = {self.k:.3f}",
            f"Form 2 (M-method): accept if est. % nonconforming <= M = {self.M*100:.3f}%",
            "method: standard deviation (sigma unknown). ASSUMES a normal characteristic.",
        ]

    def summary(self) -> dict:
        return {
            "n": self.n, "code_letter": self.code_letter, "k": self.k, "M": self.M,
            "aql": self.aql, "lot_size": self.lot_size, "level": self.level,
            "severity": self.severity,
        }

    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.02, 0.5, "\n".join(self._summary_lines()), fontsize=9,
                family="monospace", va="center")
        return ax

    def _render_axes(self, ax, kind, **kwargs):
        return self._render_standalone(ax.figure, kind, **kwargs)

    def inspect(self, sample, *, lower: float | None = None, upper: float | None = None) -> Z19Disposition:
        """Judge a lot from a sample (the measured characteristic values).

        Spec limits come from ``lower=``/``upper=`` here, or fall back to the limits
        set on the plan (``z19_plan(..., lower=, upper=)``). Computes xbar, s, the
        quality indices QL/QU, the normal-theory estimated percent nonconforming,
        and the Form-1 accept/reject decision. Surfaces a normality check - the
        percent-nonconforming estimate is unreliable if the characteristic is not
        normal."""
        from .assumptions import check_normality
        lower = lower if lower is not None else self.lower
        upper = upper if upper is not None else self.upper
        x = np.asarray(sample, dtype=float)
        x = x[~np.isnan(x)]
        n = x.size
        if n < 2:
            raise ValueError("Z1.9 inspect needs at least 2 sample values.")
        xbar = float(x.mean())
        s = float(x.std(ddof=1))
        QL = QU = None
        est_lo = est_hi = None
        accept = True
        if lower is not None:
            QL = (xbar - lower) / s
            est_lo = _pct_nonconforming(QL)
            accept = accept and QL >= self.k
        if upper is not None:
            QU = (upper - xbar) / s
            est_hi = _pct_nonconforming(QU)
            accept = accept and QU >= self.k
        if lower is None and upper is None:
            raise ValueError("Z1.9 inspect needs at least one spec limit (lower= or upper=).")
        est_total = (est_lo or 0.0) + (est_hi or 0.0)
        decision = "accept" if accept else "reject"

        checks = [check_normality(x)]
        if not checks[0].passed:
            checks[0] = replace(
                checks[0],
                recommendation="Z1.9 percent-nonconforming estimate assumes normality; this "
                "sample fails the normality test -> the estimate and decision are unreliable.")
        step = Step(operation="z19_inspect",
                    params={"decision": decision, "QL": QL, "QU": QU, "k": self.k,
                            "est_pct_total": est_total},
                    n_affected=n, timestamp=_now())
        return Z19Disposition(
            decision=decision, n=n, xbar=xbar, s=s, lower=lower, upper=upper,
            QL=QL, QU=QU, k=self.k, est_pct_lower=est_lo, est_pct_upper=est_hi,
            est_pct_total=est_total, assumptions=checks, history=(step,))


def z19_plan(lot_size: int, aql: float, *, level: str = "II", severity: str = "normal",
             lower: float | None = None, upper: float | None = None) -> Z19Plan:
    """ANSI/ASQ Z1.9 variables sampling plan (standard-deviation method, level II).

    Looks up the sample-size code letter and ``n`` from the lot size, and derives
    the Form-1 acceptability constant ``k`` (and Form-2 ``M``, the maximum allowable
    estimated percent nonconforming) for the AQL. ``aql`` is in percent (e.g. 1.0).
    Spec limits (``lower``/``upper``) may be attached here as plan metadata so
    :meth:`Z19Plan.inspect` can use them (or pass them to ``inspect`` directly).
    """
    if severity != "normal":
        raise NotImplementedError("only normal severity is implemented; "
                                  "tightened/reduced are documented extensions.")
    if not 0 < aql < 100:
        raise ValueError("aql is a percent in (0, 100), e.g. 1.0 for 1%.")
    letter = _z19_code_letter(lot_size, level)
    n = _Z19_SAMPLE_SIZE[letter]
    p = aql / 100.0
    k = _z19_k(n, p)
    M = _pct_nonconforming(k)   # boundary estimate: % nonconforming when Q == k
    from .assumptions import AssumptionCheck as _AC
    note = _AC("normality", "assumed (Z1.9)", float("nan"), None, True, None,
               "assumption", "ok", n,
               "Z1.9 assumes a normal characteristic; check normality on the sample "
               "(inspect() does this) - the % nonconforming estimate is invalid otherwise.")
    step = Step(operation="z19_plan",
                params={"lot_size": lot_size, "aql": p, "level": level,
                        "severity": severity, "code_letter": letter, "n": n, "k": k, "M": M},
                n_affected=None, timestamp=_now())
    return Z19Plan(n=n, code_letter=letter, k=k, M=M, aql=p, lot_size=lot_size,
                   level=level, severity=severity, lower=lower, upper=upper,
                   assumptions=[note], history=(step,))
