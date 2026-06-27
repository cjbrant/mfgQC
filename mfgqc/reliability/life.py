"""Life-data analysis: fit a life distribution to failure times with censoring.

Censoring is first class: each exact failure contributes f(t), each right-censored
suspension contributes R(t), each interval-censored unit F(hi)-F(lo), each
left-censored unit F(t). Maximum likelihood is primary; rank regression on the
probability plot (median ranks adjusted for suspensions) is the secondary cross-
check and the recommended fallback at few failures, where the Weibull-shape MLE
is biased.

Surface, do not decide: the fit reports how well the chosen distribution fits
(probability-plot correlation) and the competing distributions' AIC, so the
distribution is chosen on evidence, not silently. MTTF comes from the fitted
distribution, never the sample mean of censored data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from scipy import optimize, stats

from .._result import QCResult
from ..assumptions import AssumptionCheck
from ..data import QCData, Step

_DISTS = ("exponential", "weibull", "lognormal", "normal")


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Frozen scipy distribution from named parameters
# --------------------------------------------------------------------------- #
def _frozen(dist: str, p):
    if dist == "exponential":
        return stats.expon(scale=p[0])
    if dist == "weibull":
        return stats.weibull_min(p[0], scale=p[1])
    if dist == "lognormal":
        return stats.lognorm(p[1], scale=np.exp(p[0]))
    if dist == "normal":
        return stats.norm(loc=p[0], scale=p[1])
    raise ValueError(f"dist must be one of {_DISTS}; got {dist!r}.")


def _param_names(dist: str) -> tuple:
    return {"exponential": ("scale",), "weibull": ("shape", "scale"),
            "lognormal": ("mu", "sigma"), "normal": ("mu", "sigma")}[dist]


def _start(dist: str, fails, right):
    allt = np.concatenate([fails, right]) if right.size else fails
    m = float(np.mean(allt)); s = float(np.std(allt) + 1e-9)
    logm = float(np.mean(np.log(np.clip(allt, 1e-9, None))))
    logs = float(np.std(np.log(np.clip(allt, 1e-9, None))) + 1e-3)
    return {"exponential": [m], "weibull": [1.0, m],
            "lognormal": [logm, logs], "normal": [m, s]}[dist]


def _negloglik(p, dist, fails, right, left, ilo, ihi):
    if dist in ("exponential", "weibull") and np.any(np.asarray(p) <= 0):
        return 1e12
    if dist in ("lognormal", "normal") and p[-1] <= 0:
        return 1e12
    d = _frozen(dist, p)
    ll = 0.0
    if fails.size:
        ll += np.sum(d.logpdf(fails))
    if right.size:
        ll += np.sum(d.logsf(right))
    if left.size:
        ll += np.sum(d.logcdf(left))
    if ilo.size:
        ll += np.sum(np.log(np.clip(d.cdf(ihi) - d.cdf(ilo), 1e-300, None)))
    return -ll if np.isfinite(ll) else 1e12


def _fit_mle(dist, fails, right, left, ilo, ihi):
    p0 = _start(dist, fails, right)
    res = optimize.minimize(_negloglik, p0, args=(dist, fails, right, left, ilo, ihi),
                            method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    return res.x, -res.fun


def _hessian(f, x, eps=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        for j in range(n):
            xi = np.array(x, dtype=float)
            hi = eps * max(1.0, abs(x[i])); hj = eps * max(1.0, abs(x[j]))
            xi[i] += hi; xi[j] += hj; fpp = f(xi)
            xi = np.array(x, float); xi[i] += hi; xi[j] -= hj; fpm = f(xi)
            xi = np.array(x, float); xi[i] -= hi; xi[j] += hj; fmp = f(xi)
            xi = np.array(x, float); xi[i] -= hi; xi[j] -= hj; fmm = f(xi)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * hi * hj)
    return H


def _wald_ci(negll, x, alpha):
    H = _hessian(negll, x)
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(x), np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    return se, [(x[i] - z * se[i], x[i] + z * se[i]) for i in range(len(x))]


def _lr_ci(negll, x, ll_max, alpha):
    """Likelihood-ratio profile interval for each parameter."""
    crit = stats.chi2.ppf(1 - alpha, 1) / 2.0
    out = []
    for i in range(len(x)):
        def profile(val):
            if len(x) == 1:
                return -negll([val]) - ll_max + crit
            others0 = [x[j] for j in range(len(x)) if j != i]

            def nll_fixed(rest):
                p = list(rest)
                p.insert(i, val)
                return negll(p)
            r = optimize.minimize(nll_fixed, others0, method="Nelder-Mead")
            return -r.fun - ll_max + crit
        se_guess = max(abs(x[i]) * 0.5, 1e-3)
        lo = _root(profile, x[i], x[i] - 10 * se_guess)
        hi = _root(profile, x[i], x[i] + 10 * se_guess)
        out.append((lo, hi))
    return out


def _root(f, a, b):
    try:
        fa, fb = f(a), f(b)
        if fa * fb > 0:
            return float("nan")
        return float(optimize.brentq(f, min(a, b), max(a, b)))
    except (ValueError, RuntimeError):
        return float("nan")


# --------------------------------------------------------------------------- #
# Plotting positions / rank regression
# --------------------------------------------------------------------------- #
def _plot_positions(times, events):
    """Median-rank plotting positions adjusted for suspensions (Johnson method)."""
    order = np.argsort(times)
    t = np.asarray(times)[order]; e = np.asarray(events)[order]
    n = t.size
    rank = 0.0
    prev = 0.0
    positions, ft = [], []
    reverse_rank = n
    for i in range(n):
        if e[i] == 1:
            inc = (n + 1 - prev) / (1 + reverse_rank)
            rank = prev + inc
            prev = rank
            F = (rank - 0.3) / (n + 0.4)        # Benard median-rank approximation
            positions.append(F); ft.append(t[i])
        reverse_rank -= 1
    return np.array(ft), np.array(positions)


def _linearize(dist, t, F):
    if dist == "weibull":
        return np.log(t), np.log(-np.log(1 - F))
    if dist == "exponential":
        return t, -np.log(1 - F)
    if dist == "lognormal":
        return np.log(t), stats.norm.ppf(F)
    return t, stats.norm.ppf(F)                  # normal


def _fit_rankreg(dist, t, F):
    x, y = _linearize(dist, t, F)
    sl, ic, r, _p, _se = stats.linregress(x, y)
    if dist == "weibull":
        shape = sl; scale = np.exp(-ic / sl)
        return [shape, scale], r
    if dist == "exponential":
        return [1.0 / sl], r                     # slope = 1/theta
    if dist == "lognormal":
        sigma = 1.0 / sl; mu = -ic * sigma
        return [mu, sigma], r
    sigma = 1.0 / sl; mu = -ic * sigma
    return [mu, sigma], r


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class LifeFitResult(QCResult):
    """Life-distribution fit with censoring (immutable)."""

    dist: str
    method: str
    params: dict
    param_ci: dict
    mttf: float
    b10: float
    b50: float
    aic: float
    loglik: float
    n_fail: int
    n_susp: int
    ppcc: float
    competing_aic: dict
    conf: float
    _frozen: object = field(repr=False, default=None)
    _times: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _events: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def R(self, t):
        return float(self._frozen.sf(t))

    def hazard(self, t):
        return float(self._frozen.pdf(t) / max(self._frozen.sf(t), 1e-300))

    def _title(self) -> str:
        return f"Life fit ({self.dist}, {self.method}): R(t) and percentiles"

    def _summary_lines(self) -> list[str]:
        lines = [f"n = {self.n_fail + self.n_susp}   failures = {self.n_fail}   "
                 f"suspensions (right-censored) = {self.n_susp}",
                 f"method = {self.method} (MLE primary; rank regression secondary)", ""]
        for nm, v in self.params.items():
            lo, hi = self.param_ci[nm]
            lines.append(f"{nm:<10}= {v:>12.5g}   [{lo:.5g}, {hi:.5g}]  ({int(self.conf*100)}% CI)")
        lines += ["",
                  f"MTTF = {self.mttf:.5g} (from the fitted distribution, not the sample mean)",
                  f"B10 (10% fail) = {self.b10:.5g}   B50 (median life) = {self.b50:.5g}",
                  f"AIC = {self.aic:.5g}   probability-plot correlation = {self.ppcc:.4f}",
                  "",
                  "competing fits (AIC, lower is better): "
                  + ", ".join(f"{d}={a:.1f}" for d, a in sorted(self.competing_aic.items(),
                                                                key=lambda kv: kv[1]))]
        return lines

    def summary(self) -> dict:
        out = {"dist": self.dist, "method": self.method, "mttf": self.mttf,
               "b10": self.b10, "b50": self.b50, "aic": self.aic, "ppcc": self.ppcc,
               "n_fail": self.n_fail, "n_susp": self.n_susp}
        for nm, v in self.params.items():
            out[nm] = v
        return out

    def _render_standalone(self, fig, kind, **kwargs):
        from . import views
        views.life_view(self, fig, kind or "probability_plot")

    def _render_axes(self, ax, kind, **kwargs):
        from . import views
        views.life_axes(self, ax, kind or "survival")


def life_fit(qc: QCData, dist: str = "weibull", method: str = "mle", *,
             conf: float = 0.95) -> LifeFitResult:
    """Fit a life distribution to the measure/time with the event-role censoring.

    ``event`` role: 1 = exact failure, 0 = right-censored suspension. ``dist`` is
    exponential/weibull/lognormal/normal; ``method`` is 'mle' (primary) or
    'rankreg' (rank regression on the probability plot)."""
    if dist not in _DISTS:
        raise ValueError(f"dist must be one of {_DISTS}; got {dist!r}.")
    frame = qc.frame
    tcol = qc.meta.roles.get("time", qc.meta.measure)
    ecol = qc.meta.roles.get("event")
    times = np.asarray(frame[tcol], dtype=float)
    events = (np.asarray(frame[ecol], dtype=float) if ecol else np.ones(times.size))
    ok = np.isfinite(times)
    times, events = times[ok], events[ok]
    fails = times[events == 1]
    right = times[events == 0]
    left = np.array([]); ilo = np.array([]); ihi = np.array([])
    n_fail, n_susp = fails.size, right.size
    if n_fail < 2:
        raise ValueError(
            f"too few exact failures to fit ({n_fail}); a life distribution needs at least 2. "
            "Collect more failures or report the nonparametric life_table instead.")

    negll = lambda p: _negloglik(p, dist, fails, right, left, ilo, ihi)
    if method == "mle":
        params_v, ll = _fit_mle(dist, fails, right, left, ilo, ihi)
        ci_list = _lr_ci(negll, params_v, ll, 1 - conf)
    elif method == "rankreg":
        t_pp, F_pp = _plot_positions(times, events)
        params_v, _r = _fit_rankreg(dist, t_pp, F_pp)
        ll = -negll(params_v)
        _, ci_list = _wald_ci(negll, params_v, 1 - conf)
    else:
        raise ValueError("method must be 'mle' or 'rankreg'.")

    names = _param_names(dist)
    params = {nm: float(params_v[i]) for i, nm in enumerate(names)}
    param_ci = {nm: (float(ci_list[i][0]), float(ci_list[i][1])) for i, nm in enumerate(names)}
    frozen = _frozen(dist, params_v)
    k = len(params_v)
    aic = 2 * k - 2 * ll

    # probability-plot correlation (adequacy) and competing AIC
    t_pp, F_pp = _plot_positions(times, events)
    xx, yy = _linearize(dist, t_pp, F_pp)
    ppcc = float(abs(np.corrcoef(xx, yy)[0, 1])) if t_pp.size > 2 else float("nan")
    competing = {}
    for d in _DISTS:
        pv, llc = _fit_mle(d, fails, right, left, ilo, ihi)
        competing[d] = 2 * len(pv) - 2 * llc

    checks = _adequacy(dist, params, n_fail, n_susp, ppcc, competing, fails, right, left, ilo, ihi)
    step = Step(operation="life_fit",
                params={"dist": dist, "method": method, "params": params, "mttf": float(frozen.mean())},
                n_affected=times.size, timestamp=_now())
    return LifeFitResult(
        dist=dist, method=method, params=params, param_ci=param_ci,
        mttf=float(frozen.mean()), b10=float(frozen.ppf(0.10)), b50=float(frozen.ppf(0.50)),
        aic=float(aic), loglik=float(ll), n_fail=n_fail, n_susp=n_susp, ppcc=ppcc,
        competing_aic=competing, conf=conf, _frozen=frozen, _times=times, _events=events,
        assumptions=checks, history=qc.history + (step,))


def _adequacy(dist, params, n_fail, n_susp, ppcc, competing, fails, right, left, ilo, ihi):
    checks = []
    # distribution adequacy: probability-plot correlation + AIC vs the best competitor
    best = min(competing, key=competing.get)
    passed = bool(ppcc >= 0.95)
    rec = None if passed else (
        f"The {dist} probability-plot correlation is {ppcc:.3f} (< 0.95); the {dist} may not be the "
        f"best fit - the lowest-AIC distribution here is {best}. Compare the competing AIC before "
        "committing to a distribution.")
    checks.append(AssumptionCheck("distribution_fit", "prob-plot correlation", float(ppcc), None,
                                  passed, float(ppcc), "PPCC", "ok", n_fail + n_susp, rec))
    # few-failures temper (Weibull shape MLE bias)
    low = n_fail < 10
    checks.append(AssumptionCheck("failure_count", "exact failures", float(n_fail), None,
                                  not low, float(n_fail), "failures",
                                  "low_power" if low else "ok", n_fail + n_susp,
                                  None if not low else
                                  (f"Only {n_fail} exact failures; the Weibull-shape MLE is biased at "
                                   "few failures - cross-check with method='rankreg'.")))
    # constant-rate cross-flag: if Weibull shape is far from 1, an exponential MTBF would mislead
    if dist == "weibull" and "shape" in params:
        sh = params["shape"]
        far = abs(sh - 1.0) > 0.5
        kind = "wear-out" if sh > 1 else "infant-mortality"
        checks.append(AssumptionCheck("constant_failure_rate", "Weibull shape vs 1",
                                      float(sh), None, not far, float(sh), "shape", "ok",
                                      n_fail + n_susp,
                                      None if not far else
                                      (f"Weibull shape = {sh:.2f} ({kind}); the failure rate is NOT "
                                       "constant, so a blind MTBF/exponential is misleading - use the "
                                       "Weibull MTTF.")))
    return checks
