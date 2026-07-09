"""Bayesian decision helpers: sample-size assurance (spec Algorithm F) and
cost-optimal guardbanding (spec Algorithm K; BDA3 ch. 9-10; Hoff sec 4.3).

assurance() answers "how many future units should I measure so that the analysis
is likely to conclude the process is capable?" by predictive simulation from the
current posterior. The machinery is textbook predictive simulation; the term
"assurance" comes from the clinical-trials literature; the tests are
self-consistency only (monotonicity), not an external oracle.

guardband() chooses acceptance limits that minimize expected cost given a noisy
gauge: a part conforms on its true value but is accepted on its measurement, so
tightening the accept window trades scrap against escapes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from scipy import stats

from mfgqc._result import QCResult
from mfgqc.assumptions import AssumptionCheck
from mfgqc.data import Step

from ._results import _assumption_step
from .capability import _index_draws


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, repr=False)
class AssuranceResult(QCResult):
    """Assurance curve: the probability, per candidate sample size, that a future
    analysis concludes P(index >= threshold) exceeds the decision threshold."""

    quantity: str
    threshold: float
    decide_hi: float
    decide_lo: float
    n_grid: tuple
    assurance: tuple
    recommended_n: int | None
    seed: int
    sims: int
    assumptions: list = field(default_factory=list)
    history: tuple = field(default_factory=tuple)

    def _title(self) -> str:
        return "Bayesian Assurance (sample size)"

    def _summary_lines(self) -> list[str]:
        lines = [f"target: P({self.quantity} >= {self.threshold:.3g}) exceeds "
                 f"{self.decide_hi:.2f}"]
        for n, a in zip(self.n_grid, self.assurance):
            lines.append(f"  n = {n:>4}: assurance {a:.3g}")
        lines.append(f"recommended n = {self.recommended_n}")
        lines.append("Note: predictive-simulation machinery is textbook standard; the "
                     "assurance framing is from the clinical-trials literature; the "
                     "validation is self-consistency (monotonicity), not an oracle.")
        return lines

    def _render_standalone(self, fig, kind, **kwargs) -> None:
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs) -> None:
        if kind not in (None, "assurance"):
            raise ValueError(f"unknown assurance view kind={kind!r}; use None.")
        from . import plotting
        plotting.assurance_axes(ax, self)


def assurance(result, *, target: tuple = ("ppk", 1.33), decide: tuple = (0.9, 0.1),
              n_grid: tuple = (20, 50, 100, 200, 400), sims: int = 1000,
              inner_draws: int = 2000, seed: int) -> AssuranceResult:
    """Sample-size assurance from a fitted capability result's posterior predictive.

    For each candidate n: draw ``sims`` future datasets of size n from the current
    posterior predictive, refit each (noninformative), and record the fraction of
    sims whose P(index >= threshold) exceeds ``decide[0]``. Deterministic given seed.
    """
    quantity, thresh = target
    hi, lo = decide
    mun, kn, nun, sn2, spec = result.mun, result.kn, result.nun, result.sn2, result.spec
    rng = np.random.default_rng(seed)

    curve = []
    for n in n_grid:
        # one (mu, sigma) per sim from the current posterior
        sig2 = nun * sn2 / rng.chisquare(nun, sims)
        mu = rng.normal(mun, np.sqrt(sig2 / kn))
        y = rng.normal(mu[:, None], np.sqrt(sig2)[:, None], size=(sims, n))  # future data
        ybar = y.mean(1)
        s2 = y.var(1, ddof=1)
        # inner noninformative push-through per sim
        isig2 = (n - 1) * s2[:, None] / rng.chisquare(n - 1, size=(sims, inner_draws))
        imu = rng.normal(ybar[:, None], np.sqrt(isig2 / n))
        _, _, _, ppk = _index_draws(imu, np.sqrt(isig2), spec)
        p_capable = (ppk >= thresh).mean(axis=1)  # P(index>=thresh) per sim
        curve.append(float((p_capable >= hi).mean()))

    recommended = next((int(n) for n, a in zip(n_grid, curve) if a >= hi), None)
    step = Step(
        operation="bayes.assurance",
        params={"parent": result.provenance_digest(), "quantity": quantity,
                "threshold": float(thresh), "decide_hi": float(hi), "decide_lo": float(lo),
                "n_grid": list(n_grid), "sims": int(sims), "inner_draws": int(inner_draws),
                "seed": int(seed)},
        n_affected=None, timestamp=_now(),
    )
    return AssuranceResult(
        quantity=quantity, threshold=float(thresh), decide_hi=float(hi), decide_lo=float(lo),
        n_grid=tuple(n_grid), assurance=tuple(curve), recommended_n=recommended,
        seed=int(seed), sims=int(sims), assumptions=[], history=(step,),
    )


# --------------------------------------------------------------------------- #
# Guardband: cost-optimal acceptance limits (spec Algorithm K; BDA3 ch. 9)
# --------------------------------------------------------------------------- #
_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0  # golden-section ratio


@dataclass(frozen=True, repr=False)
class GuardbandResult(QCResult):
    """Cost-optimal acceptance limits under gauge measurement error (immutable).

    Conformance is judged on the true value against the spec; acceptance is judged
    on the noisy measurement against the reported limits. The optimum trades scrap
    (rejecting conforming parts) against escapes (accepting nonconforming parts)."""

    spec: object
    sigma_gauge: float
    c_scrap: float
    c_escape: float
    pp_sd: float
    a_lo: float | None
    a_hi: float | None
    scrap_pct: float
    escape_ppm: float
    expected_cost: float
    naive_a_lo: float | None
    naive_a_hi: float | None
    naive_scrap_pct: float
    naive_escape_ppm: float
    naive_expected_cost: float
    non_unimodal: bool
    fallback_used: bool
    seed: int
    ndraws: int
    grid: dict
    cred_level: float
    assumptions: list = field(default_factory=list)
    history: tuple = field(default_factory=tuple)

    def _title(self) -> str:
        return "Bayesian Guardband (acceptance limits)"

    def _summary_lines(self) -> list[str]:
        def fmt(v):
            return "n/a" if v is None else f"{v:.4g}"

        lines = [
            f"gauge sd = {self.sigma_gauge:.4g}   costs: scrap {self.c_scrap:.4g}, "
            f"escape {self.c_escape:.4g}",
            f"optimal accept limits = ({fmt(self.a_lo)}, {fmt(self.a_hi)})",
            f"  expected cost = {self.expected_cost:.4g}   scrap = {self.scrap_pct:.3g}%   "
            f"escape = {self.escape_ppm:.0f} ppm",
            f"naive limits (= spec) = ({fmt(self.naive_a_lo)}, {fmt(self.naive_a_hi)})",
            f"  expected cost = {self.naive_expected_cost:.4g}   scrap = {self.naive_scrap_pct:.3g}%   "
            f"escape = {self.naive_escape_ppm:.0f} ppm",
        ]
        if self.fallback_used:
            lines.append("Note: expected-cost surface was not unimodal; limits are the grid "
                         "global minimum, not a golden-section optimum.")
        return lines

    def _render_standalone(self, fig, kind, **kwargs) -> None:
        from . import plotting
        if kind is None:
            plotting.guardband_panels(fig, self)
            return
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs) -> None:
        from . import plotting
        if kind in (None, "cost"):
            plotting.guardband_cost_axes(ax, self)
        elif kind in ("limits", "acceptance"):
            plotting.guardband_limits_axes(ax, self)
        else:
            raise ValueError(f"unknown guardband view kind={kind!r}; use None, 'cost', "
                             f"or 'limits'.")


def _simpson_weights(n: int, dx: float) -> np.ndarray:
    """Composite Simpson weights for n points (n odd); trapezoid fallback if even."""
    w = np.ones(n)
    if n % 2 == 1:
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        return w * dx / 3.0
    w[0] = w[-1] = 0.5
    return w * dx


def _golden(f, a: float, b: float, tol: float) -> float:
    """Golden-section minimizer of a unimodal f on [a, b]."""
    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    fc, fd = f(c), f(d)
    while (b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _INV_PHI * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _INV_PHI * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def _coarse_min(f, a: float, b: float, n: int = 41) -> tuple:
    """Coarse scan of f on [a, b]: returns (bracket_lo, bracket_hi, unimodal) where
    the bracket surrounds the sampled minimum and ``unimodal`` is False if the
    sampled first-difference signs flip more than once."""
    xs = np.linspace(a, b, n)
    ys = np.array([f(v) for v in xs])
    signs = np.sign(np.diff(ys)[np.abs(np.diff(ys)) > 1e-15])
    unimodal = True if signs.size == 0 else int((np.diff(signs) != 0).sum()) <= 1
    k = int(np.argmin(ys))
    return xs[max(k - 1, 0)], xs[min(k + 1, n - 1)], unimodal


def _grid_argmin_2d(ec, lo_domain: tuple, hi_domain: tuple, n: int = 81) -> tuple:
    """Global minimizer of ec(a_lo, a_hi) over the 2-D coarse grid (a_lo < a_hi)."""
    best = None
    for vlo in np.linspace(lo_domain[0], lo_domain[1], n):
        for vhi in np.linspace(hi_domain[0], hi_domain[1], n):
            if vhi <= vlo:
                continue
            c = ec(vlo, vhi)
            if best is None or c < best[0]:
                best = (c, vlo, vhi)
    return best[1], best[2]


def _minimize_two_sided(ec, lo_domain: tuple, hi_domain: tuple, tol: float) -> tuple:
    """Coordinate golden-section on (a_lo, a_hi), each 1-D slice bracketed by a
    coarse-scan unimodality check. If any slice is non-unimodal, fall back to the
    global minimum of a 2-D coarse grid. Returns (a_lo, a_hi, non_unimodal,
    fallback_used).

    Note: for a single spec interval with a Gaussian gauge the true expected-cost
    surface is provably unimodal in each coordinate (scrap monotone decreasing,
    escape monotone increasing), so ``non_unimodal`` is a defensive guard against
    pathological posteriors, not a case the normal model reaches."""
    a_lo, a_hi = lo_domain[1], hi_domain[0]
    non_unimodal = False
    for _ in range(10):
        la, lb, uni1 = _coarse_min(lambda v: ec(v, a_hi), lo_domain[0], min(lo_domain[1], a_hi - tol))
        new_lo = _golden(lambda v: ec(v, a_hi), la, lb, tol)
        ha, hb, uni2 = _coarse_min(lambda v: ec(new_lo, v), max(hi_domain[0], new_lo + tol), hi_domain[1])
        new_hi = _golden(lambda v: ec(new_lo, v), ha, hb, tol)
        non_unimodal = non_unimodal or (not uni1) or (not uni2)
        converged = abs(new_lo - a_lo) < tol and abs(new_hi - a_hi) < tol
        a_lo, a_hi = new_lo, new_hi
        if converged:
            break
    fallback_used = False
    if non_unimodal:
        fallback_used = True
        a_lo, a_hi = _grid_argmin_2d(ec, lo_domain, hi_domain)
    return a_lo, a_hi, non_unimodal, fallback_used


def guardband(result, *, sigma_gauge: float, c_scrap: float, c_escape: float,
              grid: int = 4001, ndraws: int = 4000, seed: int,
              cred_level: float = 0.95) -> GuardbandResult:
    """Choose acceptance limits minimizing expected cost under gauge error.

    ``result`` is a fitted capability result exposing posterior draws
    ``_draws['mu']`` and ``_draws['sigma']`` and a ``spec``. The true value
    x has posterior predictive mixture density g(x) = mean_i N(x; mu_i, sigma_i);
    the measurement is m = x + e, e ~ N(0, sigma_gauge^2). Because m is Gaussian
    given x, the acceptance probability A(x) is closed form and the (x, m) integral
    collapses to a single 1-D quadrature over x. Minimized by coordinate
    golden-section with a coarse-scan unimodality check and a 2-D grid fallback.
    """
    if sigma_gauge <= 0:
        raise ValueError(f"sigma_gauge must be positive; got {sigma_gauge}.")
    if c_scrap < 0 or c_escape < 0:
        raise ValueError("costs must be non-negative.")
    spec = result.spec
    if not spec.has_any():
        raise ValueError("guardband requires at least one spec limit.")

    mu = np.asarray(result._draws["mu"], dtype=float)
    sigma = np.asarray(result._draws["sigma"], dtype=float)
    pp_sd = float(math.sqrt(np.mean(sigma ** 2) + np.var(mu)))
    if sigma_gauge >= pp_sd:
        raise ValueError(
            f"gauge dominates process: sigma_gauge ({sigma_gauge:.4g}) >= posterior "
            f"predictive sd ({pp_sd:.4g}). The measurement noise swamps the process; "
            f"improve the gauge (MSA) before guardbanding.")

    rng = np.random.default_rng(seed)
    if mu.size > ndraws:
        idx = rng.choice(mu.size, ndraws, replace=False)
        mu, sigma = mu[idx], sigma[idx]
    n_used = mu.size

    lsl, usl = spec.lower, spec.upper
    pred_mean = float(mu.mean())
    # Union the spec-anchored and predictive-mean-anchored ranges so the quadrature
    # grid always covers the density, even for a process centered far outside spec.
    x_lo = min(lsl if lsl is not None else math.inf, pred_mean - 8.0 * pp_sd) - 8.0 * sigma_gauge - 8.0 * pp_sd
    x_hi = max(usl if usl is not None else -math.inf, pred_mean + 8.0 * pp_sd) + 8.0 * sigma_gauge + 8.0 * pp_sd
    x = np.linspace(x_lo, x_hi, grid)
    # force LSL/USL onto exact nodes to keep the hard conformance edges sharp
    for lim in (lsl, usl):
        if lim is not None:
            x[np.argmin(np.abs(x - lim))] = lim
    dx = (x_hi - x_lo) / (grid - 1)
    w = _simpson_weights(grid, dx)

    # predictive mixture density g(x) over the (subsampled) draw set
    g = np.exp(-0.5 * ((x[:, None] - mu[None, :]) / sigma[None, :]) ** 2)
    g = (g / (sigma[None, :] * math.sqrt(2.0 * math.pi))).mean(axis=1)

    conforming = np.ones(grid, dtype=bool)
    if lsl is not None:
        conforming &= x >= lsl
    if usl is not None:
        conforming &= x <= usl
    wg = w * g

    def accept_prob(a_lo, a_hi):
        p = np.ones(grid)
        if a_hi is not None:
            p = p * stats.norm.cdf((a_hi - x) / sigma_gauge)
        if a_lo is not None:
            p = p - stats.norm.cdf((a_lo - x) / sigma_gauge)
            p = np.clip(p, 0.0, 1.0) if a_hi is not None else stats.norm.sf((a_lo - x) / sigma_gauge)
        return p

    def costs(a_lo, a_hi):
        A = accept_prob(a_lo, a_hi)
        scrap = float((wg * conforming * (1.0 - A)).sum())
        escape = float((wg * (~conforming) * A).sum())
        return scrap, escape, c_scrap * scrap + c_escape * escape

    def ec(a_lo, a_hi):
        return costs(a_lo, a_hi)[2]

    m_mid = pred_mean if (lsl is None or usl is None) else 0.5 * (lsl + usl)
    lo_domain = (x_lo, m_mid)
    hi_domain = (m_mid, x_hi)

    tol_a = max(dx, 1e-4 * pp_sd)
    fallback_used = False

    if lsl is not None and usl is not None:
        a_lo, a_hi, non_unimodal, fallback_used = _minimize_two_sided(ec, lo_domain, hi_domain, tol_a)
    elif usl is not None:
        a_lo = None
        lo_b, hi_b, uni = _coarse_min(lambda v: ec(None, v), x_lo, x_hi)
        non_unimodal = not uni
        a_hi = _golden(lambda v: ec(None, v), lo_b, hi_b, tol_a)
    else:
        a_hi = None
        lo_b, hi_b, uni = _coarse_min(lambda v: ec(v, None), x_lo, x_hi)
        non_unimodal = not uni
        a_lo = _golden(lambda v: ec(v, None), lo_b, hi_b, tol_a)

    scrap, escape, ec_opt = costs(a_lo, a_hi)
    nscrap, nescape, ec_naive = costs(lsl, usl)

    checks = []
    if non_unimodal:
        checks.append(AssumptionCheck(
            name="guardband_unimodal", test="expected-cost surface unimodal",
            statistic=0.0, p_value=None, passed=False, magnitude=None,
            magnitude_label=None, reliability="ok", n=int(n_used),
            recommendation=("expected-cost surface is not unimodal; reported limits are the "
                            "grid global minimum, not a golden-section optimum. Inspect the "
                            "cost curve.")))

    step = Step(
        operation="bayes.guardband",
        params={
            "parent": result.provenance_digest() if hasattr(result, "provenance_digest") else None,
            "sigma_gauge": float(sigma_gauge), "c_scrap": float(c_scrap),
            "c_escape": float(c_escape), "grid": int(grid), "ndraws": int(n_used),
            "seed": int(seed), "x_bounds": [float(x_lo), float(x_hi)],
            "cred_level": float(cred_level),
        },
        n_affected=None, timestamp=_now(),
    )
    grid_meta = {"points": int(grid), "x_lo": float(x_lo), "x_hi": float(x_hi)}
    return GuardbandResult(
        spec=spec, sigma_gauge=float(sigma_gauge), c_scrap=float(c_scrap),
        c_escape=float(c_escape), pp_sd=pp_sd,
        a_lo=None if a_lo is None else float(a_lo), a_hi=None if a_hi is None else float(a_hi),
        scrap_pct=100.0 * scrap, escape_ppm=1e6 * escape, expected_cost=ec_opt,
        naive_a_lo=lsl, naive_a_hi=usl, naive_scrap_pct=100.0 * nscrap,
        naive_escape_ppm=1e6 * nescape, naive_expected_cost=ec_naive,
        non_unimodal=non_unimodal, fallback_used=fallback_used, seed=int(seed),
        ndraws=int(n_used), grid=grid_meta, cred_level=float(cred_level),
        assumptions=checks, history=(step,) + tuple(_assumption_step(a) for a in checks),
    )
