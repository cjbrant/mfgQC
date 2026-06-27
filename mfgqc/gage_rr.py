"""Gage R&R (Measurement System Analysis).

Two methods: ``anova`` (default, includes the part x operator interaction and
pools it into error when not significant at AIAG's alpha = 0.25, i.e. p > 0.25,
per AIAG MSA 4th ed.) and ``xbar_r`` (Average &
Range with the AIAG K-factors). Correctness anchor (watch-list #3): the ANOVA
expected-mean-squares decomposition and ``ndc = 1.41 * PV / GRR`` (truncated).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from . import assumptions as _assume
from ._result import QCResult
from .assumptions import AssumptionCheck
from .constants import gage_k1, gage_k2, gage_k3
from .data import QCData, Step

_VALID_METHODS = ("anova", "xbar_r")
# AIAG MSA 4th ed.: pool the part x operator interaction into error (repeatability)
# when its F-test is NOT significant at alpha = 0.25 (i.e. p > 0.25); retain it
# otherwise. (Earlier this used 0.05, which pools too readily versus the standard.)
_INTERACTION_POOL_ALPHA = 0.25


@dataclass(frozen=True, repr=False)
class GageRRResult(QCResult):
    """Result of a gage R&R analysis (immutable)."""

    method: str
    n_parts: int
    n_operators: int
    n_trials: int
    # standard deviations (study-variation components)
    ev: float
    av: float
    interaction: float
    grr: float
    pv: float
    tv: float
    # variance components
    var_repeat: float
    var_oper: float
    var_interaction: float
    var_part: float
    var_grr: float
    var_total: float
    # percentages keyed by component name
    pct_study: dict
    pct_contrib: dict
    pct_tol: dict | None
    ndc: int
    verdict: str
    pooled: bool
    anova_table: dict | None
    alpha: float = 0.10
    ev_ci: tuple[float, float] | None = None
    av_ci: tuple[float, float] | None = None
    grr_ci: tuple[float, float] | None = None
    pv_ci: tuple[float, float] | None = None
    _crossed: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"Gage R&R (method={self.method})"

    def _summary_lines(self) -> list[str]:
        conf = round((1.0 - self.alpha) * 100)
        ci_for = {"EV": self.ev_ci, "AV": self.av_ci, "GRR": self.grr_ci, "PV": self.pv_ci}
        has_ci = any(ci_for.values())
        lines = [
            f"Design: {self.n_parts} parts x {self.n_operators} operators x {self.n_trials} trials",
            f"Verdict: {self.verdict}   ndc = {self.ndc}",
            "",
        ]
        if has_ci:
            lines.append(f"{'component':<16}{'std dev':>10}"
                         f"{('lower ' + str(conf) + '%'):>12}{('upper ' + str(conf) + '%'):>12}"
                         f"{'%study var':>12}{'%contrib':>11}")
        else:
            lines.append(f"{'component':<16}{'std dev':>12}{'%study var':>12}{'%contrib':>12}")
        rows = [
            ("Repeatability(EV)", self.ev, "EV"),
            ("Reproducib.(AV)", self.av, "AV"),
            ("GRR", self.grr, "GRR"),
            ("Part(PV)", self.pv, "PV"),
            ("Total(TV)", self.tv, "TV"),
        ]
        for label, sd, key in rows:
            sv = self.pct_study.get(key)
            ct = self.pct_contrib.get(key)
            sv_s = "" if sv is None else f"{sv:.2f}%"
            ct_s = "" if ct is None else f"{ct:.2f}%"
            if has_ci:
                ci = ci_for.get(key)
                lo_s = "" if ci is None else f"{ci[0]:.3f}"
                hi_s = "" if ci is None else f"{ci[1]:.3f}"
                lines.append(f"{label:<16}{sd:>10.5g}{lo_s:>12}{hi_s:>12}{sv_s:>12}{ct_s:>11}")
            else:
                lines.append(f"{label:<16}{sd:>12.5g}{sv_s:>12}{ct_s:>12}")
        if self.method == "anova":
            lines.append("")
            lines.append("interaction " + ("pooled into error (not significant)"
                                           if self.pooled else "retained (significant)"))
        if self.pct_tol is not None:
            lines.append("")
            lines.append(f"%tolerance (GRR) = {self.pct_tol.get('GRR'):.2f}%")
        return lines

    def summary(self) -> dict:
        """Flat {label: value} dict of the headline numbers (dashboard-ready)."""
        def lo(ci): return None if ci is None else ci[0]
        def hi(ci): return None if ci is None else ci[1]
        return {
            "method": self.method,
            "n_parts": self.n_parts,
            "n_operators": self.n_operators,
            "n_trials": self.n_trials,
            "EV": self.ev, "EV_CI_low": lo(self.ev_ci), "EV_CI_high": hi(self.ev_ci),
            "AV": self.av, "AV_CI_low": lo(self.av_ci), "AV_CI_high": hi(self.av_ci),
            "GRR": self.grr, "GRR_CI_low": lo(self.grr_ci), "GRR_CI_high": hi(self.grr_ci),
            "PV": self.pv, "PV_CI_low": lo(self.pv_ci), "PV_CI_high": hi(self.pv_ci),
            "TV": self.tv,
            "pct_study_GRR": self.pct_study.get("GRR"),
            "pct_contrib_GRR": self.pct_contrib.get("GRR"),
            "pct_tol_GRR": None if self.pct_tol is None else self.pct_tol.get("GRR"),
            "ndc": self.ndc,
            "verdict": self.verdict,
            "confidence": round((1.0 - self.alpha) * 100),
        }

    def _render_standalone(self, fig, kind, **kwargs):
        from . import plotting
        plotting.gage_panels(fig, self)

    def _render_axes(self, ax, kind, **kwargs):
        from . import plotting
        plotting.gage_components_bar(ax, self)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _assumption_step(a: AssumptionCheck) -> Step:
    return Step(
        operation=f"assumption:{a.name}",
        params={"test": a.test, "passed": a.passed, "magnitude": a.magnitude,
                "reliability": a.reliability, "p_value": a.p_value, "statistic": a.statistic},
        n_affected=None, timestamp=_now(),
    )


def _verdict(pct_grr: float, ndc: int) -> str:
    if pct_grr < 10 and ndc >= 5:
        return "acceptable"
    if pct_grr > 30 or ndc < 2:
        return "unacceptable"
    return "marginal (conditionally acceptable)"


def _to_array(crossed) -> tuple[np.ndarray, list, list, int]:
    """Build a balanced (parts x operators x trials) array from the crossed frame."""
    df = crossed.frame
    parts = list(crossed.parts)
    operators = list(crossed.operators)
    r = crossed.n_replicates
    arr = np.full((len(parts), len(operators), r), np.nan)
    pidx = {p: i for i, p in enumerate(parts)}
    oidx = {o: j for j, o in enumerate(operators)}
    for (p, o), grp in df.groupby(["part", "operator"], sort=False):
        vals = grp["value"].to_numpy(dtype=float)
        arr[pidx[p], oidx[o], : len(vals)] = vals
    return arr, parts, operators, r


def _ndc(pv: float, grr: float) -> int:
    if grr <= 0:
        return 0
    return max(1, int(1.41 * (pv / grr)))  # AIAG truncates to integer


def _mls_variance_ci(pos: list, neg: list, alpha: float) -> tuple[float, float]:
    """Modified Large Sample (Ting-Burdick-Graybill) CI for a VARIANCE that is a
    linear combination ``gamma = sum(pos c*MS) - sum(neg c*MS)`` of mean squares.

    Each entry is ``(coef>0, MS, df)``. Reproduces the AIAG MSA Table III-B.9
    confidence limits (Burdick-Larsen). Returns the (lower, upper) variance bounds.
    """
    a2 = alpha / 2.0

    def G(f):
        return 1.0 - f / stats.chi2.ppf(1 - a2, f)

    def H(f):
        return f / stats.chi2.ppf(a2, f) - 1.0

    point = sum(c * ms for c, ms, _ in pos) - sum(c * ms for c, ms, _ in neg)
    v_l = sum((G(f) * c * ms) ** 2 for c, ms, f in pos) + sum((H(f) * c * ms) ** 2 for c, ms, f in neg)
    v_u = sum((H(f) * c * ms) ** 2 for c, ms, f in pos) + sum((G(f) * c * ms) ** 2 for c, ms, f in neg)
    # Cross terms appear only between a positive-set MS and a negative-set MS.
    for c_i, ms_i, f_i in pos:
        for c_j, ms_j, f_j in neg:
            f_up = stats.f.ppf(1 - a2, f_i, f_j)
            f_lo = stats.f.ppf(a2, f_i, f_j)
            g_ij = ((f_up - 1.0) ** 2 - G(f_i) ** 2 * f_up ** 2 - H(f_j) ** 2) / f_up
            h_ij = ((1.0 - f_lo) ** 2 - H(f_i) ** 2 * f_lo ** 2 - G(f_j) ** 2) / f_lo
            v_l += g_ij * c_i * c_j * ms_i * ms_j
            v_u += h_ij * c_i * c_j * ms_i * ms_j
    lo = max(point - np.sqrt(max(v_l, 0.0)), 0.0)
    hi = max(point + np.sqrt(max(v_u, 0.0)), 0.0)
    return lo, hi


def _gage_cis(anova_table: dict, pooled: bool, p: int, o: int, r: int, alpha: float) -> dict:
    """ANOVA-method component CIs (std-dev scale) via the MLS method, matching AIAG
    Table III-B.9. Returns {'EV','AV','GRR','PV': (lo, hi)}."""
    ms_o = anova_table["operator"]["ms"]
    f_o = anova_table["operator"]["df"]
    ms_p = anova_table["parts"]["ms"]
    f_p = anova_table["parts"]["df"]
    ss_int, f_int = anova_table["interaction"]["ss"], anova_table["interaction"]["df"]
    ms_int = anova_table["interaction"]["ms"]
    ss_eq, f_eq = anova_table["equipment"]["ss"], anova_table["equipment"]["df"]
    pr, orr = p * r, o * r

    if pooled:
        # No-interaction model: pooled error.
        ms_e = (ss_int + ss_eq) / (f_int + f_eq)
        f_e = f_int + f_eq
        comps = {
            "EV": ([(1.0, ms_e, f_e)], []),
            "AV": ([(1.0 / pr, ms_o, f_o)], [(1.0 / pr, ms_e, f_e)]),
            "GRR": ([(1.0 / pr, ms_o, f_o), ((pr - 1.0) / pr, ms_e, f_e)], []),
            "PV": ([(1.0 / orr, ms_p, f_p)], [(1.0 / orr, ms_e, f_e)]),
        }
    else:
        # Interaction retained: full model with MS_PO and MS_E separate.
        ms_e, f_e = anova_table["equipment"]["ms"], f_eq
        comps = {
            "EV": ([(1.0, ms_e, f_e)], []),
            "AV": ([(1.0 / pr, ms_o, f_o), ((p - 1.0) / pr, ms_int, f_int)],
                   [(1.0 / r, ms_e, f_e)]),
            "GRR": ([(1.0 / pr, ms_o, f_o), ((p - 1.0) / pr, ms_int, f_int),
                     ((r - 1.0) / r, ms_e, f_e)], []),
            "PV": ([(1.0 / orr, ms_p, f_p)], [(1.0 / orr, ms_int, f_int)]),
        }
    out = {}
    for key, (pos, neg) in comps.items():
        lo, hi = _mls_variance_ci(pos, neg, alpha)
        out[key] = (float(np.sqrt(lo)), float(np.sqrt(hi)))  # variance -> std dev
    return out


def compute(qc: QCData, *, method: str = "anova", alpha: float = 0.10) -> GageRRResult:
    """Compute a gage R&R study.

    Parameters
    ----------
    qc : QCData
        Requires ``part``, ``operator`` and ``replicate`` roles.
    method : str, optional
        ``"anova"`` (default) or ``"xbar_r"``.
    alpha : float, optional
        Significance level for the component confidence limits (default 0.10 ->
        90% CL, the AIAG convention; note this differs from capability's 95%).
        CIs are computed for the ``anova`` method (MLS / Burdick-Larsen); the
        ``xbar_r`` method reports point estimates only.

    Returns
    -------
    GageRRResult
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}; got {method!r}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}.")
    crossed = qc.crossed()
    if not crossed.balanced:
        df = crossed.frame
        per_op = {op: int(round(len(sub) / max(sub["part"].nunique(), 1)))
                  for op, sub in df.groupby("operator", sort=False)}
        detail = ", ".join(f"{op}: {t} trials/part" for op, t in per_op.items())
        raise ValueError(
            "gage R&R requires a balanced design (equal trials in every "
            f"part x operator cell). Trials per part by operator: {detail}."
        )
    arr, parts, operators, r = _to_array(crossed)
    p, o = len(parts), len(operators)
    if r < 2:
        raise ValueError("gage R&R requires at least 2 trials per part/operator cell.")

    spec = qc.meta.limits
    tol = (spec.upper - spec.lower) if (spec.lower is not None and spec.upper is not None) else None

    checks: list[AssumptionCheck] = []

    if method == "anova":
        result_core = _anova(arr, p, o, r, checks)
    else:
        result_core = _xbar_r(arr, p, o, r, checks)

    (var_repeat, var_oper, var_interaction, var_part,
     ev, av, interaction, grr, pv, anova_table, pooled) = result_core

    var_grr = var_repeat + var_oper + var_interaction
    var_total = var_grr + var_part
    tv = float(np.sqrt(var_total))
    grr = float(np.sqrt(var_grr))  # ensure consistency with components

    def pct(sd):
        return 100.0 * sd / tv if tv > 0 else float("nan")

    pct_study = {"EV": pct(ev), "AV": pct(av), "GRR": pct(grr), "PV": pct(pv), "TV": 100.0}
    if interaction > 0:
        pct_study["INT"] = pct(interaction)
    pct_contrib = {
        "EV": 100.0 * var_repeat / var_total,
        "AV": 100.0 * var_oper / var_total,
        "GRR": 100.0 * var_grr / var_total,
        "PV": 100.0 * var_part / var_total,
        "TV": 100.0,
    }
    if var_interaction > 0:
        pct_contrib["INT"] = 100.0 * var_interaction / var_total

    pct_tol = None
    if tol:
        pct_tol = {k: 100.0 * 6.0 * v / tol for k, v in
                   {"EV": ev, "AV": av, "GRR": grr, "PV": pv}.items()}

    ndc = _ndc(pv, grr)
    verdict = _verdict(pct_study["GRR"], ndc)

    # Component confidence limits (MLS / Burdick-Larsen) - ANOVA method only.
    ev_ci = av_ci = grr_ci = pv_ci = None
    if method == "anova" and anova_table is not None:
        cis = _gage_cis(anova_table, pooled, p, o, r, alpha)
        ev_ci, av_ci, grr_ci, pv_ci = cis["EV"], cis["AV"], cis["GRR"], cis["PV"]

    # ndc adequacy guardrail (AIAG ndc >= 5 rule drives passed; ndc value is context)
    ndc_ok = ndc >= _assume.NDC_MIN
    checks.append(AssumptionCheck(
        name="ndc_adequacy", test="ndc (AIAG ndc>=5)",
        statistic=float(ndc), p_value=None, passed=ndc_ok,
        magnitude=float(ndc), magnitude_label="ndc", reliability="ok", n=p * o * r,
        recommendation=None if ndc_ok else (
            f"ndc = {ndc} (< 5 AIAG): the measurement system cannot reliably "
            "distinguish parts. Improve gage resolution/repeatability."),
    ))

    step = Step(
        operation="gage_rr",
        params={"method": method, "pooled": pooled, "ndc": ndc,
                "pct_grr": pct_study["GRR"], "verdict": verdict},
        n_affected=int(np.sum(~np.isnan(arr))),
        timestamp=_now(),
    )
    history = qc.history + (step,) + tuple(_assumption_step(a) for a in checks)

    return GageRRResult(
        method=method, n_parts=p, n_operators=o, n_trials=r,
        ev=ev, av=av, interaction=interaction, grr=grr, pv=pv, tv=tv,
        var_repeat=var_repeat, var_oper=var_oper, var_interaction=var_interaction,
        var_part=var_part, var_grr=var_grr, var_total=var_total,
        pct_study=pct_study, pct_contrib=pct_contrib, pct_tol=pct_tol,
        ndc=ndc, verdict=verdict, pooled=pooled, anova_table=anova_table,
        alpha=alpha, ev_ci=ev_ci, av_ci=av_ci, grr_ci=grr_ci, pv_ci=pv_ci,
        _crossed=crossed.frame, assumptions=checks, history=history,
    )


def _anova(arr, p, o, r, checks):
    grand = float(np.nanmean(arr))
    part_means = np.nanmean(arr, axis=(1, 2))
    oper_means = np.nanmean(arr, axis=(0, 2))
    cell_means = np.nanmean(arr, axis=2)  # (p, o)

    ss_total = float(np.nansum((arr - grand) ** 2))
    ss_parts = float(o * r * np.sum((part_means - grand) ** 2))
    ss_oper = float(p * r * np.sum((oper_means - grand) ** 2))
    ss_cells = float(r * np.sum((cell_means - grand) ** 2))
    ss_int = ss_cells - ss_parts - ss_oper
    ss_equip = ss_total - ss_cells

    df_parts, df_oper = p - 1, o - 1
    df_int = (p - 1) * (o - 1)
    df_equip = p * o * (r - 1)
    df_total = p * o * r - 1

    ms_parts = ss_parts / df_parts
    ms_oper = ss_oper / df_oper
    ms_int = ss_int / df_int
    ms_equip = ss_equip / df_equip

    # AIAG table: F-ratios against the equipment (error) mean square.
    f_parts = ms_parts / ms_equip
    f_oper = ms_oper / ms_equip
    f_int = ms_int / ms_equip
    p_int = float(stats.f.sf(f_int, df_int, df_equip))

    anova_table = {
        "operator": {"df": df_oper, "ss": ss_oper, "ms": ms_oper, "f": f_oper},
        "parts": {"df": df_parts, "ss": ss_parts, "ms": ms_parts, "f": f_parts},
        "interaction": {"df": df_int, "ss": ss_int, "ms": ms_int, "f": f_int, "p_value": p_int},
        "equipment": {"df": df_equip, "ss": ss_equip, "ms": ms_equip},
        "total": {"df": df_total, "ss": ss_total},
    }

    pooled = p_int > _INTERACTION_POOL_ALPHA
    if pooled:
        ms_err = (ss_int + ss_equip) / (df_int + df_equip)
        var_repeat = ms_err
        var_interaction = 0.0
        var_oper = max((ms_oper - ms_err) / (p * r), 0.0)
        var_part = max((ms_parts - ms_err) / (o * r), 0.0)
    else:
        var_repeat = ms_equip
        var_interaction = max((ms_int - ms_equip) / r, 0.0)
        var_oper = max((ms_oper - ms_int) / (p * r), 0.0)
        var_part = max((ms_parts - ms_int) / (o * r), 0.0)

    ev = float(np.sqrt(var_repeat))
    av = float(np.sqrt(var_oper))
    interaction = float(np.sqrt(var_interaction))
    pv = float(np.sqrt(var_part))
    grr = float(np.sqrt(var_repeat + var_oper + var_interaction))

    # ANOVA-specific assumption checks
    residuals = (arr - cell_means[:, :, None])[~np.isnan(arr)]
    checks.append(_assume.check_normality(np.asarray(residuals)))
    oper_groups = [arr[:, j, :][~np.isnan(arr[:, j, :])] for j in range(o)]
    checks.append(_assume.check_homogeneity(oper_groups))

    return (var_repeat, var_oper, var_interaction, var_part,
            ev, av, interaction, grr, pv, anova_table, pooled)


def _xbar_r(arr, p, o, r, checks):
    # Ranges per (part, operator) cell, then each operator's average range.
    cell_range = np.nanmax(arr, axis=2) - np.nanmin(arr, axis=2)  # (p, o)
    oper_rbar = np.nanmean(cell_range, axis=0)                    # (o,)
    rbar = float(np.mean(oper_rbar))

    oper_means = np.nanmean(arr, axis=(0, 2))                     # (o,)
    xdiff = float(np.max(oper_means) - np.min(oper_means))

    part_means = np.nanmean(arr, axis=(1, 2))                     # (p,)
    rp = float(np.max(part_means) - np.min(part_means))

    k1, k2, k3 = gage_k1(r), gage_k2(o), gage_k3(p)
    ev = rbar * k1
    av_sq = (xdiff * k2) ** 2 - (ev ** 2) / (p * r)
    av = float(np.sqrt(max(av_sq, 0.0)))
    pv = rp * k3

    var_repeat = ev ** 2
    var_oper = av ** 2
    var_interaction = 0.0
    var_part = pv ** 2
    grr = float(np.sqrt(var_repeat + var_oper))

    # Homogeneity across operators (cheap, informative)
    oper_groups = [arr[:, j, :][~np.isnan(arr[:, j, :])] for j in range(o)]
    checks.append(_assume.check_homogeneity(oper_groups))

    return (var_repeat, var_oper, var_interaction, var_part,
            ev, av, 0.0, grr, pv, None, False)
