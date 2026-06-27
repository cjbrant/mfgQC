"""MTBF estimation with chi-square bounds, and reliability demonstration test
sizing (solve-for-the-missing-one, parallel to the power module)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from scipy import optimize, stats

from .._result import QCResult
from ..assumptions import AssumptionCheck
from ..data import QCData, Step


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# MTBF with chi-square bounds
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class MTBFResult(QCResult):
    """Constant-failure-rate MTBF with chi-square confidence bounds (immutable)."""

    mtbf: float
    lower: float
    upper: float
    total_time: float
    failures: int
    kind: str
    conf: float
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"MTBF ({self.kind}, constant failure rate)"

    def _summary_lines(self) -> list[str]:
        return [f"total test time = {self.total_time:g}   failures = {self.failures}   "
                f"test = {self.kind}",
                f"MTBF = {self.mtbf:.6g}   [{self.lower:.6g}, {self.upper:.6g}] "
                f"({int(self.conf*100)}% CI)",
                f"failure rate = {1/self.mtbf:.4g}", "",
                "EXACT only under a CONSTANT failure rate (exponential life). If a Weibull fit gives "
                "a shape far from 1, this MTBF is misleading - use the Weibull MTTF."]

    def summary(self) -> dict:
        return {"mtbf": self.mtbf, "lower": self.lower, "upper": self.upper,
                "total_time": self.total_time, "failures": self.failures, "kind": self.kind}

    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        ax.errorbar([0], [self.mtbf], yerr=[[self.mtbf - self.lower], [self.upper - self.mtbf]],
                    fmt="o", capsize=6)
        ax.set_xticks([]); ax.set_ylabel("MTBF"); ax.set_title(self._title())

    def _render_axes(self, ax, kind, **kwargs):
        ax.errorbar([0], [self.mtbf], yerr=[[self.mtbf - self.lower], [self.upper - self.mtbf]],
                    fmt="o", capsize=6)
        return ax


def mtbf(qc_or_T, failures=None, *, kind: str = "time_terminated", conf: float = 0.90):
    """MTBF = total time / failures, with chi-square bounds.

    Pass a QCData (total time and failures derived from the time/event roles) or a
    total test time with ``failures=``. ``kind`` is 'time_terminated' (test stopped
    at a time) or 'failure_terminated' (stopped at the r-th failure)."""
    if isinstance(qc_or_T, QCData):
        frame = qc_or_T.frame
        tcol = qc_or_T.meta.roles.get("time", qc_or_T.meta.measure)
        ecol = qc_or_T.meta.roles.get("event")
        t = np.asarray(frame[tcol], dtype=float)
        e = (np.asarray(frame[ecol], dtype=float) if ecol else np.ones(t.size))
        T = float(np.nansum(t)); r = int(np.nansum(e == 1))
        base_hist = qc_or_T.history
    else:
        T = float(qc_or_T); r = int(failures); base_hist = ()
    if r < 1:
        raise ValueError("MTBF needs at least one failure.")
    alpha = 1 - conf
    point = T / r
    if kind == "time_terminated":
        lower = 2 * T / stats.chi2.ppf(1 - alpha / 2, 2 * r + 2)
        upper = 2 * T / stats.chi2.ppf(alpha / 2, 2 * r)
    elif kind == "failure_terminated":
        lower = 2 * T / stats.chi2.ppf(1 - alpha / 2, 2 * r)
        upper = 2 * T / stats.chi2.ppf(alpha / 2, 2 * r)
    else:
        raise ValueError("kind must be 'time_terminated' or 'failure_terminated'.")
    flag = AssumptionCheck("constant_failure_rate", "exponential assumption", float("nan"), None,
                           True, None, None, "ok", r,
                           "MTBF assumes a constant failure rate; verify with a Weibull life_fit "
                           "(shape near 1) before trusting it.")
    step = Step(operation="mtbf", params={"mtbf": float(point), "kind": kind, "T": T, "r": r},
                n_affected=r, timestamp=_now())
    return MTBFResult(mtbf=float(point), lower=float(lower), upper=float(upper),
                      total_time=T, failures=r, kind=kind, conf=conf,
                      assumptions=[flag], history=base_hist + (step,))


# --------------------------------------------------------------------------- #
# Demonstration test sizing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class DemonstrationResult(QCResult):
    """Reliability demonstration test plan (immutable)."""

    solved_for: str
    reliability: float
    confidence: float
    n: int
    failures: int
    dist: str
    shape: float | None
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"Reliability demonstration test (solved for {self.solved_for})"

    def _summary_lines(self) -> list[str]:
        plan = "zero-failure (success-run)" if self.failures == 0 else f"allow {self.failures} failures"
        return [f"demonstrate reliability >= {self.reliability:.4f} at "
                f"{self.confidence:.0%} confidence",
                f"plan: n = {self.n} units, {plan}",
                f"assumed model: {self.dist}"
                + (f" (shape {self.shape})" if self.shape else ""), "",
                "a zero-failure result BOUNDS the reliability at the stated confidence; it does not "
                "prove it, and it demonstrates the assumed model as much as the hardware."]

    def summary(self) -> dict:
        return {"solved_for": self.solved_for, "reliability": self.reliability,
                "confidence": self.confidence, "n": self.n, "failures": self.failures,
                "dist": self.dist}

    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        ns = np.arange(1, max(self.n * 2, 10))
        ax.plot(ns, (1 - self.confidence) ** (1 / ns))
        ax.axhline(self.reliability, ls="--"); ax.axvline(self.n, ls="--")
        ax.set_xlabel("n"); ax.set_ylabel("demonstrable reliability"); ax.set_title(self._title())

    def _render_axes(self, ax, kind, **kwargs):
        return ax


def _binom_conf(n, f, R):
    """Confidence that the true reliability >= R given f failures in n (binomial)."""
    return float(1 - stats.binom.cdf(f, n, 1 - R))


def demonstration_test(reliability=None, confidence=None, n=None, failures: int = 0,
                       dist: str = "binomial", shape=None) -> DemonstrationResult:
    """Size a reliability demonstration test. Leave exactly one of ``reliability``,
    ``confidence``, ``n`` as None to solve for it. ``failures=0`` is the zero-failure
    (success-run) plan."""
    nones = [reliability is None, confidence is None, n is None]
    if sum(nones) != 1:
        raise ValueError("leave exactly one of reliability/confidence/n as None to solve for it.")

    if failures == 0:
        # success-run: confidence C = 1 - R^n
        if n is None:
            n = int(math.ceil(math.log(1 - confidence) / math.log(reliability)))
            solved = "n"
        elif reliability is None:
            reliability = (1 - confidence) ** (1.0 / n)
            solved = "reliability"
        else:
            confidence = 1 - reliability ** n
            solved = "confidence"
    else:
        if n is None:
            nn = failures + 1
            while _binom_conf(nn, failures, reliability) < confidence:
                nn += 1
            n = nn; solved = "n"
        elif reliability is None:
            reliability = float(optimize.brentq(
                lambda R: _binom_conf(n, failures, R) - confidence, 1e-6, 1 - 1e-9))
            solved = "reliability"
        else:
            confidence = _binom_conf(n, failures, reliability); solved = "confidence"

    flag = AssumptionCheck("assumed_model", "demonstration basis", float("nan"), None, True,
                           None, None, "ok", int(n),
                           f"the plan assumes a {dist} model; a zero-failure pass bounds, not proves, "
                           "the reliability.")
    step = Step(operation="demonstration_test",
                params={"solved_for": solved, "reliability": float(reliability),
                        "confidence": float(confidence), "n": int(n), "failures": failures},
                n_affected=None, timestamp=_now())
    return DemonstrationResult(solved_for=solved, reliability=float(reliability),
                              confidence=float(confidence), n=int(n), failures=failures,
                              dist=dist, shape=shape, assumptions=[flag], history=(step,))
