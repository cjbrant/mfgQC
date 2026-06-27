"""Regression additions: automated model selection, logistic regression, and
non-linear least squares. Built on top of the existing OLS engine where it fits;
statsmodels for the logistic fit (a secondary engine, not reimplemented).

Surface, do not decide: automated selection inflates the apparent significance of
retained terms (stated in the report); logistic refuses legibly on complete
separation rather than returning unstable estimates; non-linear regression
reports the same residual diagnostics as OLS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from . import assumptions as _assume
from ._result import QCResult
from .assumptions import AssumptionCheck
from .data import QCData, Step
from .regression import RegressionResult, compute_regression


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Model selection (on the OLS engine)
# --------------------------------------------------------------------------- #
def _aic_bic(reg: RegressionResult) -> tuple[float, float]:
    n = reg.n
    p = len(reg.predictors) + 1
    ss_res = float(np.sum(reg._resid ** 2))
    if ss_res <= 0:
        ss_res = 1e-300
    ll = -0.5 * n * (np.log(2 * np.pi * ss_res / n) + 1)
    aic = 2 * p - 2 * ll
    bic = p * np.log(n) - 2 * ll
    return float(aic), float(bic)


def _score(qc: QCData, preds: list[str], criterion: str) -> float:
    if not preds:
        # intercept-only: score from the total SS
        y = qc.frame[qc.meta.measure].to_numpy(dtype=float)
        y = y[~np.isnan(y)]
        ss = float(np.sum((y - y.mean()) ** 2)); n = y.size
        ll = -0.5 * n * (np.log(2 * np.pi * ss / n) + 1)
        return float(2 - 2 * ll) if criterion == "aic" else float(np.log(n) - 2 * ll)
    reg = compute_regression(qc, preds)
    aic, bic = _aic_bic(reg)
    return aic if criterion == "aic" else bic


def select(qc: QCData, candidates: list[str], *, direction: str = "forward",
           criterion: str = "aic", alpha_in: float = 0.05,
           alpha_out: float = 0.10) -> RegressionResult:
    """Forward / backward / stepwise selection. Returns the chosen OLS fit with
    the selection path recorded; ``criterion`` is ``'aic'``, ``'bic'``, or ``'p'``."""
    candidates = list(candidates)
    path: list[str] = []
    if direction not in ("forward", "backward", "stepwise"):
        raise ValueError("direction must be forward/backward/stepwise.")

    if criterion == "p":
        chosen = _select_by_p(qc, candidates, direction, alpha_in, alpha_out, path)
    else:
        chosen = _select_by_ic(qc, candidates, direction, criterion, path)

    if not chosen:
        raise ValueError("selection removed every predictor; no model to report.")
    import dataclasses
    reg = compute_regression(qc, chosen)
    note = ("automated selection inflates the apparent significance of the retained terms; "
            "treat their p-values as optimistic and confirm on fresh data.")
    step = Step(operation="regression.select",
                params={"direction": direction, "criterion": criterion,
                        "selected": chosen, "path": path}, n_affected=reg.n, timestamp=_now())
    return dataclasses.replace(reg, history=reg.history + (step,),
                               selection_path=tuple(path), selection_note=note)


def _select_by_ic(qc, candidates, direction, criterion, path):
    if direction == "backward":
        current = list(candidates)
    else:
        current = []
    best = _score(qc, current, criterion)
    improved = True
    while improved:
        improved = False
        if direction in ("forward", "stepwise"):
            for c in [x for x in candidates if x not in current]:
                s = _score(qc, current + [c], criterion)
                if s < best - 1e-9:
                    best, add = s, c; improved = True
            if improved:
                current.append(add); path.append(f"+{add} ({criterion}={best:.2f})")
                continue
        if direction in ("backward", "stepwise"):
            for c in list(current):
                s = _score(qc, [x for x in current if x != c], criterion)
                if s < best - 1e-9:
                    best, drop = s, c; improved = True
            if improved:
                current.remove(drop); path.append(f"-{drop} ({criterion}={best:.2f})")
    return current


def _select_by_p(qc, candidates, direction, alpha_in, alpha_out, path):
    current = list(candidates) if direction == "backward" else []
    changed = True
    while changed:
        changed = False
        if direction in ("forward", "stepwise"):
            best_p, best_c = 1.0, None
            for c in [x for x in candidates if x not in current]:
                reg = compute_regression(qc, current + [c])
                p = reg.p_value[c]
                if p < best_p:
                    best_p, best_c = p, c
            if best_c is not None and best_p < alpha_in:
                current.append(best_c); path.append(f"+{best_c} (p={best_p:.3g})"); changed = True
        if direction in ("backward", "stepwise") and current:
            reg = compute_regression(qc, current)
            worst = max(current, key=lambda c: reg.p_value[c])
            if reg.p_value[worst] > alpha_out:
                current.remove(worst); path.append(f"-{worst} (p={reg.p_value[worst]:.3g})")
                changed = True
    return current


# --------------------------------------------------------------------------- #
# Logistic regression
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class LogisticResult(QCResult):
    """Binary logistic regression result (immutable)."""

    terms: tuple
    coef: dict
    se: dict
    z: dict
    p_value: dict
    odds_ratio: dict
    odds_ratio_ci: dict
    deviance: float
    null_deviance: float
    pseudo_r2: float
    aic: float
    auc: float
    classification: dict
    n: int
    response: str = "y"
    converged: bool = True
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"Logistic regression: P({self.response}=1) ~ {' + '.join(self.terms[1:])}"

    def _summary_lines(self) -> list[str]:
        c = self.classification
        acc = (c["TP"] + c["TN"]) / max(1, sum(c.values()))
        lines = [f"n = {self.n}   deviance = {self.deviance:.4g}   "
                 f"pseudo R^2 (McFadden) = {self.pseudo_r2:.4g}   AIC = {self.aic:.4g}",
                 f"ROC AUC = {self.auc:.4g}   accuracy @0.5 = {acc:.3f}",
                 "",
                 f"{'term':<14}{'coef':>10}{'odds ratio':>12}{'OR 95% CI':>22}{'p':>10}"]
        for t in self.terms:
            lo, hi = self.odds_ratio_ci[t]
            lines.append(f"{t:<14}{self.coef[t]:>10.4g}{self.odds_ratio[t]:>12.4g}"
                         f"{f'[{lo:.3g}, {hi:.3g}]':>22}{self.p_value[t]:>10.3g}")
        lines += ["",
                  f"classification @0.5: TP={c['TP']} FP={c['FP']} TN={c['TN']} FN={c['FN']}"]
        return lines

    def summary(self) -> dict:
        out = {"n": self.n, "deviance": self.deviance, "pseudo_r2": self.pseudo_r2,
               "aic": self.aic, "auc": self.auc}
        for t in self.terms:
            out[f"coef[{t}]"] = self.coef[t]
            out[f"odds_ratio[{t}]"] = self.odds_ratio[t]
            out[f"p[{t}]"] = self.p_value[t]
        return out

    def _render_standalone(self, fig, kind, **kwargs):
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        from . import palette as _pal
        pal = _pal.active()
        fpr, tpr = self._roc
        ax.plot(fpr, tpr, color=pal.center, lw=2, label=f"ROC (AUC={self.auc:.3f})")
        ax.plot([0, 1], [0, 1], color=pal.limit, ls="--", lw=1)
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
        ax.set_title(self._title()); ax.legend(loc="lower right", fontsize=8)
        return ax

    _roc: tuple = field(repr=False, default=(np.array([0, 1]), np.array([0, 1])))


def logistic(qc: QCData, on) -> LogisticResult:
    """Binary logistic regression of the measure on ``on`` (a column or list)."""
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    preds = [on] if isinstance(on, str) else list(on)
    response = qc.meta.measure
    frame = qc.frame
    cols = [response] + preds
    sub = frame[cols].apply(pd.to_numeric, errors="coerce").dropna()
    y = sub[response].to_numpy(dtype=float)
    if set(np.unique(y).tolist()) - {0.0, 1.0}:
        raise ValueError(f"logistic needs a 0/1 binary response; {response!r} has other values.")
    X = sm.add_constant(sub[preds].to_numpy(dtype=float))
    terms = ("const",) + tuple(preds)
    import warnings as _warnings
    separated = False
    try:
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            res = sm.Logit(y, X).fit(disp=0)
        separated = any("separ" in str(w.message).lower() for w in caught)
    except (PerfectSeparationError, np.linalg.LinAlgError) as e:
        raise ValueError(
            "logistic regression refused: the data are completely separated, so the maximum-"
            "likelihood estimates are infinite. A predictor perfectly splits the classes - drop "
            f"it or use penalized regression. ({type(e).__name__})") from e
    converged = bool(res.mle_retvals.get("converged", True))
    if separated or not converged or np.max(np.abs(res.params)) > 15:
        raise ValueError(
            "logistic regression refused: the data are completely (or quasi-) separated, so the "
            "estimates are unstable / infinite (a predictor nearly splits the classes). Drop the "
            "separating predictor or use penalized (Firth / L2) regression.")

    coef = dict(zip(terms, res.params))
    se = dict(zip(terms, res.bse))
    zvals = dict(zip(terms, res.tvalues))
    pvals = dict(zip(terms, res.pvalues))
    ci = res.conf_int()
    odds = {t: float(np.exp(coef[t])) for t in terms}
    or_ci = {t: (float(np.exp(ci[i][0])), float(np.exp(ci[i][1]))) for i, t in enumerate(terms)}

    prob = res.predict(X)
    pred = (prob >= 0.5).astype(int)
    c = {"TP": int(np.sum((pred == 1) & (y == 1))), "FP": int(np.sum((pred == 1) & (y == 0))),
         "TN": int(np.sum((pred == 0) & (y == 0))), "FN": int(np.sum((pred == 0) & (y == 1)))}
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan")
    fpr, tpr, _ = roc_curve(y, prob)

    converged = bool(res.mle_retvals.get("converged", True))
    sep_flag = AssumptionCheck("separation", "max |coef| / se", float(np.max(np.abs(res.tvalues))),
                               None, True, None, None, "ok", len(y),
                               None)
    if any(abs(coef[t]) > 15 for t in terms):
        sep_flag = AssumptionCheck("separation", "max |coef|", float(max(abs(v) for v in coef.values())),
                                   None, False, None, None, "ok", len(y),
                                   "Very large coefficients suggest quasi-separation; the odds ratios "
                                   "are unstable - check for a predictor that nearly splits the classes.")
    conv_flag = AssumptionCheck("convergence", "MLE convergence", 1.0 if converged else 0.0, None,
                                converged, None, None, "ok", len(y),
                                None if converged else "the optimizer did not converge; estimates are unreliable.")

    step = Step(operation="logistic", params={"on": preds, "response": response,
                                              "pseudo_r2": float(res.prsquared)},
                n_affected=len(y), timestamp=_now())
    return LogisticResult(
        terms=terms, coef=coef, se=se, z=zvals, p_value=pvals, odds_ratio=odds,
        odds_ratio_ci=or_ci, deviance=float(-2 * res.llf), null_deviance=float(-2 * res.llnull),
        pseudo_r2=float(res.prsquared), aic=float(res.aic), auc=auc, classification=c,
        n=len(y), response=response, converged=converged,
        assumptions=[sep_flag, conv_flag], history=qc.history + (step,), _roc=(fpr, tpr))


# --------------------------------------------------------------------------- #
# Non-linear least squares
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class NonlinearResult(QCResult):
    """Non-linear least-squares fit (immutable)."""

    param_names: tuple
    params: dict
    se: dict
    ci: dict
    r_squared: float
    resid_std_err: float
    n: int
    response: str = "y"
    predictor: str = "x"
    _x: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _y: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _resid: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"Non-linear regression: {self.response} ~ f({self.predictor})"

    def _summary_lines(self) -> list[str]:
        lines = [f"n = {self.n}   R^2 = {self.r_squared:.4g}   "
                 f"resid std err = {self.resid_std_err:.4g}", "",
                 f"{'param':<12}{'estimate':>12}{'std err':>12}{'95% CI':>24}"]
        for p in self.param_names:
            lo, hi = self.ci[p]
            lines.append(f"{p:<12}{self.params[p]:>12.5g}{self.se[p]:>12.4g}"
                         f"{f'[{lo:.4g}, {hi:.4g}]':>24}")
        return lines

    def summary(self) -> dict:
        out = {"n": self.n, "r_squared": self.r_squared, "resid_std_err": self.resid_std_err}
        for p in self.param_names:
            out[f"param[{p}]"] = self.params[p]
            out[f"se[{p}]"] = self.se[p]
        return out

    def _render_standalone(self, fig, kind, **kwargs):
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        from . import palette as _pal
        pal = _pal.active()
        order = np.argsort(self._x)
        ax.scatter(self._x, self._y, s=18, color=pal.data, label="data")
        ax.plot(self._x[order], (self._y - self._resid)[order], color=pal.center, lw=2, label="fit")
        ax.set_xlabel(self.predictor); ax.set_ylabel(self.response)
        ax.set_title(self._title()); ax.legend(fontsize=8)
        return ax


def nls(qc: QCData, on: str, model, start, *, alpha: float = 0.05) -> NonlinearResult:
    """Non-linear least squares: ``model(x, *params)`` fit by ``scipy.curve_fit``
    from initial ``start``. Reports parameters with CIs and OLS-style residuals."""
    import inspect
    response = qc.meta.measure
    sub = qc.frame[[response, on]].apply(pd.to_numeric, errors="coerce").dropna()
    x = sub[on].to_numpy(dtype=float)
    y = sub[response].to_numpy(dtype=float)
    start = list(start)
    names = list(inspect.signature(model).parameters)[1:][: len(start)]
    popt, pcov = optimize.curve_fit(model, x, y, p0=start, maxfev=20000)
    resid = y - model(x, *popt)
    n = y.size
    dof = max(1, n - len(popt))
    se = np.sqrt(np.diag(pcov))
    tcrit = stats.t.ppf(1 - alpha / 2, dof)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    params = {nm: float(popt[i]) for i, nm in enumerate(names)}
    se_d = {nm: float(se[i]) for i, nm in enumerate(names)}
    ci = {nm: (float(popt[i] - tcrit * se[i]), float(popt[i] + tcrit * se[i]))
          for i, nm in enumerate(names)}
    checks = [_assume.check_normality(resid, context="regression")]
    step = Step(operation="nls", params={"on": on, "params": params}, n_affected=n, timestamp=_now())
    return NonlinearResult(
        param_names=tuple(names), params=params, se=se_d, ci=ci, r_squared=float(r2),
        resid_std_err=float(np.sqrt(ss_res / dof)), n=n, response=response, predictor=on,
        _x=x, _y=y, _resid=resid, assumptions=checks, history=qc.history + (step,))
