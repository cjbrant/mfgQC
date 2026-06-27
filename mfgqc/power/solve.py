"""Sample size and power: solve for the missing one of {effect, n, power}.

A pre-data planning engine, parallel to ``mfgqc.design``. Each function fixes
``alpha`` and the operating point of a test and solves for whichever of
``effect`` / ``n`` / ``power`` is left ``None``. Built on the noncentral t and F
distributions and the normal approximation for proportions (scipy only).

Surface, do not decide: the report states which quantity was solved for, the full
operating point, and the planning inputs that are not data-checkable (the effect
size is assumed, the variance is an estimate). When ``effect`` is the solved
quantity it is reported as the minimum detectable effect; mfgQC never picks an
effect size for you.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from scipy import optimize, stats

from .._result import QCResult
from ..assumptions import AssumptionCheck
from ..data import Step

_N_HI = 1e7


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _validate_power(power, alpha: float) -> None:
    if power is not None and not (alpha < power < 1.0):
        raise ValueError(
            f"cannot solve: target power {power:.3g} must lie strictly between "
            f"alpha={alpha:.3g} and 1.")


def _exactly_one_none(**kw) -> str:
    missing = [k for k, v in kw.items() if v is None]
    if len(missing) != 1:
        raise ValueError(
            f"exactly one of {list(kw)} must be left None to solve for it; "
            f"got {len(missing)} None ({missing or 'none'}). Fix alpha separately.")
    return missing[0]


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class PowerResult(QCResult):
    """Result of a sample-size / power solve (immutable, pre-data planning)."""

    test: str
    solved_for: str
    effect: float
    n: float
    power: float
    alpha: float
    kind: str = ""
    alternative: str = "two-sided"
    groups: int | None = None
    effect_label: str = "effect size"
    approximate: bool = False
    notes: tuple[str, ...] = ()
    _curve_x: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _curve_y: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _curve_xlabel: str = "n per group"
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    @property
    def n_total(self) -> float:
        """Total sample size across groups (n per group times the group count)."""
        g = self.groups if self.groups else (2 if self.kind == "two-sample" else 1)
        return self.n * g

    def _title(self) -> str:
        return f"Power / sample size ({self.test}): solved for {self.solved_for}"

    def _summary_lines(self) -> list[str]:
        n_disp = self.n if not float(self.n).is_integer() else int(self.n)
        lines = [
            f"solved for: {self.solved_for}",
            f"{self.effect_label} = {self.effect:.4g}   n = {n_disp} (per group)"
            f"   total n = {self.n_total:.6g}",
            f"power = {self.power:.4g}   alpha = {self.alpha:.4g}   "
            f"test = {self.kind or self.test}   alternative = {self.alternative}",
        ]
        if self.solved_for == "n":
            import math
            lines.append(f"recommended n = {math.ceil(self.n)} per group "
                         f"(smallest integer reaching the target power)")
        if self.solved_for == "effect":
            lines.append("this is the MINIMUM DETECTABLE effect at the given n and power; "
                         "mfgQC does not choose an effect size for you.")
        if self.approximate:
            lines.append("NOTE: normal approximation; it degrades for small n*p "
                         "(few expected successes).")
        for note in self.notes:
            lines.append(f"NOTE: {note}")
        return lines

    def summary(self) -> dict:
        return {"test": self.test, "solved_for": self.solved_for, "effect": self.effect,
                "n": self.n, "n_total": self.n_total, "power": self.power,
                "alpha": self.alpha, "kind": self.kind, "alternative": self.alternative,
                "approximate": self.approximate}

    def _render_standalone(self, fig, kind, **kwargs):
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        from .. import palette as _pal
        pal = _pal.active()
        if kind not in (None, "power_curve"):
            raise ValueError(f"unknown power view kind={kind!r}; use None or 'power_curve'.")
        ax.plot(self._curve_x, self._curve_y, color=pal.center, lw=2)
        ax.axhline(self.power, color=pal.limit, ls="--", lw=1)
        mark = self.n if self._curve_xlabel.startswith("n") else self.effect
        ax.axvline(mark, color=pal.ooc, ls="--", lw=1)
        ax.scatter([mark], [self.power], color=pal.ooc, zorder=5,
                   label=f"operating point ({mark:.3g}, {self.power:.3g})")
        ax.set_xlabel(self._curve_xlabel)
        ax.set_ylabel("power")
        ax.set_ylim(0, 1.02)
        ax.set_title(self._title())
        ax.legend(loc="lower right", fontsize=8)
        return ax


# --------------------------------------------------------------------------- #
# Power functions (the noncentral math)
# --------------------------------------------------------------------------- #
def _t_power(effect: float, n: float, alpha: float, kind: str, alternative: str) -> float:
    effect = abs(effect)
    if kind == "two-sample":
        df = 2.0 * n - 2.0
        ncp = effect * np.sqrt(n / 2.0)
    elif kind in ("one-sample", "paired"):
        df = n - 1.0
        ncp = effect * np.sqrt(n)
    else:
        raise ValueError(f"t_test kind must be one-sample/two-sample/paired; got {kind!r}.")
    if df <= 0:
        return float("nan")
    if alternative == "two-sided":
        tc = stats.t.ppf(1 - alpha / 2.0, df)
        return float(stats.nct.sf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp))
    tc = stats.t.ppf(1 - alpha, df)
    return float(stats.nct.sf(tc, df, ncp))


def _anova_power(f: float, k: int, n: float, alpha: float) -> float:
    df1 = k - 1
    df2 = k * (n - 1.0)
    if df2 <= 0:
        return float("nan")
    lam = f ** 2 * k * n
    fc = stats.f.ppf(1 - alpha, df1, df2)
    return float(stats.ncf.sf(fc, df1, df2, lam))


def _prop_power(p1: float, p2: float, n: float, alpha: float, kind: str) -> float:
    if kind == "two-sample":
        pbar = (p1 + p2) / 2.0
        z = stats.norm.ppf(1 - alpha / 2.0)
        num = abs(p1 - p2) * np.sqrt(n) - z * np.sqrt(2 * pbar * (1 - pbar))
        den = np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        return float(stats.norm.cdf(num / den)) if den > 0 else float("nan")
    # one-sample vs p2 (the null proportion)
    z = stats.norm.ppf(1 - alpha / 2.0)
    num = abs(p1 - p2) * np.sqrt(n) - z * np.sqrt(p2 * (1 - p2))
    den = np.sqrt(p1 * (1 - p1))
    return float(stats.norm.cdf(num / den)) if den > 0 else float("nan")


def _var_power(ratio: float, n: float, alpha: float) -> float:
    df = n - 1.0
    if df <= 0:
        return float("nan")
    f_hi = stats.f.ppf(1 - alpha / 2.0, df, df)
    f_lo = stats.f.ppf(alpha / 2.0, df, df)
    return float(stats.f.sf(f_hi / ratio, df, df) + stats.f.cdf(f_lo / ratio, df, df))


def _solve(func, target: float, lo: float, hi: float, what: str) -> float:
    """Root-find the operating-point variable that hits the target power. Power is
    monotone increasing in the variable, so bracket by doubling from ``lo`` until
    it crosses the target, stopping short of any NaN region (the noncentral
    distributions go numerically unstable at very large arguments)."""
    flo = func(lo)
    if np.isfinite(flo) and flo >= target:
        return float(lo)
    a, fa = lo, flo
    b = lo * 2.0
    while b <= hi:
        fb = func(b)
        if not np.isfinite(fb):
            break
        if fb >= target:
            return float(optimize.brentq(lambda v: func(v) - target, a, b))
        a, b = b, b * 2.0
    raise ValueError(
        f"cannot solve for {what}: the target power {target:.3g} is not reachable "
        "over the search range. Check the effect size, alpha, or target power.")


def _assumption(name: str, note: str) -> AssumptionCheck:
    # planning inputs are not data-checkable; record them as reliability context.
    return AssumptionCheck(name, "planning input", float("nan"), None, True, None,
                           None, "low_power", 0, note)


def _finish(test, solved_for, *, effect, n, power, alpha, kind="", alternative="two-sided",
            groups=None, effect_label="effect size", approximate=False, notes=(),
            curve_var="n"):
    # build the power curve over the solved (or natural) variable
    if curve_var == "n":
        xs = np.linspace(max(2.0, n * 0.25), n * 2.5, 80)
        ys = np.array([_curve_eval(test, effect, x, alpha, kind, alternative, groups) for x in xs])
        xlabel = "n per group"
    else:
        xs = np.linspace(max(1e-3, effect * 0.1), effect * 2.5, 80)
        ys = np.array([_curve_eval(test, e, n, alpha, kind, alternative, groups) for e in xs])
        xlabel = effect_label
    checks = [_assumption("effect_size", "the effect size is a planning assumption, not "
                          "measured from data."),
              _assumption("variance", "the variance / baseline rate is an estimate; revisit "
                          "the plan when phase-I data arrive.")]
    eff_scalar = abs(effect[0] - effect[1]) if isinstance(effect, tuple) else float(effect)
    step = Step(operation=f"power.{test}",
                params={"solved_for": solved_for, "effect": eff_scalar, "n": n,
                        "power": power, "alpha": alpha, "kind": kind}, n_affected=None,
                timestamp=_now())
    return PowerResult(test=test, solved_for=solved_for, effect=eff_scalar, n=float(n),
                       power=float(power), alpha=float(alpha), kind=kind,
                       alternative=alternative, groups=groups, effect_label=effect_label,
                       approximate=approximate, notes=tuple(notes),
                       _curve_x=xs, _curve_y=ys, _curve_xlabel=xlabel,
                       assumptions=checks, history=(step,))


def _curve_eval(test, effect, n, alpha, kind, alternative, groups):
    if test == "t_test":
        return _t_power(effect, n, alpha, kind, alternative)
    if test == "anova":
        return _anova_power(effect, groups, n, alpha)
    if test == "proportion":
        return _prop_power(effect[0], effect[1], n, alpha, kind) if isinstance(effect, tuple) \
            else _prop_power(*effect, n, alpha, kind)
    if test == "variance":
        return _var_power(effect, n, alpha)
    raise ValueError(test)


# --------------------------------------------------------------------------- #
# Public solve-for functions
# --------------------------------------------------------------------------- #
def t_test(effect=None, n=None, power=None, *, alpha: float = 0.05,
           kind: str = "two-sample", alternative: str = "two-sided") -> PowerResult:
    """Solve a one-sample / two-sample / paired t-test for effect, n, or power.

    Parameters
    ----------
    effect : float or None
        Cohen's d (standardized mean difference). Left None to solve for the
        minimum detectable d.
    n : float or None
        Sample size PER GROUP. Left None to solve for it.
    power : float or None
        Target power. Left None to solve for the achieved power.
    alpha : float
    kind : str
        ``"two-sample"`` (default), ``"one-sample"``, or ``"paired"``.
    alternative : str
        ``"two-sided"`` (default) or ``"one-sided"``.
    """
    solved = _exactly_one_none(effect=effect, n=n, power=power)
    _validate_power(power, alpha)
    pf = lambda e, nn: _t_power(e, nn, alpha, kind, alternative)
    if solved == "power":
        power = pf(effect, n)
    elif solved == "n":
        n = _solve(lambda nn: pf(effect, nn), power, 2.0001, _N_HI, "n")
    else:
        effect = _solve(lambda e: pf(e, n), power, 1e-6, 100.0, "effect")
    return _finish("t_test", solved, effect=effect, n=n, power=power, alpha=alpha,
                   kind=kind, alternative=alternative, effect_label="Cohen's d",
                   curve_var="effect" if solved == "effect" else "n")


def anova(groups: int, effect=None, n=None, power=None, *, alpha: float = 0.05) -> PowerResult:
    """Solve a one-way ANOVA (k = ``groups``) for Cohen's f, n per group, or power.

    Parameters
    ----------
    groups : int
        Number of groups k.
    effect : float or None
        Cohen's f. Left None to solve for the minimum detectable f.
    n : float or None
        Sample size per group. Left None to solve for it.
    power : float or None
    alpha : float
    """
    if groups < 2:
        raise ValueError(f"anova needs at least 2 groups; got {groups}.")
    solved = _exactly_one_none(effect=effect, n=n, power=power)
    _validate_power(power, alpha)
    pf = lambda f, nn: _anova_power(f, groups, nn, alpha)
    if solved == "power":
        power = pf(effect, n)
    elif solved == "n":
        n = _solve(lambda nn: pf(effect, nn), power, 2.0001, _N_HI, "n")
    else:
        effect = _solve(lambda f: pf(f, n), power, 1e-6, 100.0, "effect")
    return _finish("anova", solved, effect=effect, n=n, power=power, alpha=alpha,
                   kind="one-way", groups=groups, effect_label="Cohen's f",
                   curve_var="effect" if solved == "effect" else "n")


def proportion(p1=None, p2=None, n=None, power=None, *, alpha: float = 0.05,
               kind: str = "two-sample") -> PowerResult:
    """Solve a proportion test (normal approximation) for n or power.

    ``p1`` and ``p2`` are the two proportions (or, for ``kind="one-sample"``, the
    alternative and null proportions). Solve for ``n`` per group or ``power``.
    """
    if p1 is None or p2 is None:
        raise ValueError("proportion needs both p1 and p2 (the effect is fixed by them).")
    solved = _exactly_one_none(n=n, power=power)
    _validate_power(power, alpha)
    pf = lambda nn: _prop_power(p1, p2, nn, alpha, kind)
    if solved == "power":
        power = pf(n)
    else:
        n = _solve(pf, power, 2.0001, _N_HI, "n")
    eff = abs(p1 - p2)
    notes = (f"p1={p1:g}, p2={p2:g} (difference {eff:g}).",)
    return _finish("proportion", solved, effect=(p1, p2), n=n, power=power, alpha=alpha,
                   kind=kind, effect_label="|p1 - p2|", approximate=True, notes=notes,
                   curve_var="n")


def variance(ratio=None, n=None, power=None, *, alpha: float = 0.05) -> PowerResult:
    """Solve a two-variance F-test for the variance ratio, n per group, or power.

    ``ratio`` is the true variance ratio sigma1^2 / sigma2^2 to detect.
    """
    solved = _exactly_one_none(ratio=ratio, n=n, power=power)
    _validate_power(power, alpha)
    pf = lambda r, nn: _var_power(r, nn, alpha)
    if solved == "power":
        power = pf(ratio, n)
    elif solved == "n":
        n = _solve(lambda nn: pf(ratio, nn), power, 2.0001, _N_HI, "n")
    else:
        ratio = _solve(lambda r: pf(r, n), power, 1.0001, 1000.0, "ratio")
    return _finish("variance", solved, effect=ratio, n=n, power=power, alpha=alpha,
                   kind="two-variance F", effect_label="variance ratio",
                   curve_var="n")
