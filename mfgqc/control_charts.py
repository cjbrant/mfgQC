"""Control charts: variable (I-MR, Xbar-R, Xbar-S) and attribute (p, np, c, u),
plus a run-rules engine (Nelson / Western Electric).

Limits use the conventional constants table (``mfgqc.constants``). The chart kind
is inferred when not given, and the inferred choice is recorded in history. The
package never silently switches an explicitly requested kind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import assumptions as _assume
from ._result import QCResult
from .assumptions import AssumptionCheck
from .constants import control_constant
from .data import QCData, Step

_VARIABLE = ("i_mr", "xbar_r", "xbar_s")
_ATTRIBUTE = ("p", "np", "c", "u")
_VALID_KINDS = _VARIABLE + _ATTRIBUTE


@dataclass(frozen=True)
class Violation:
    """A single run-rule violation."""

    point: int          # 1-based index
    value: float
    chart: str          # "location" | "dispersion"
    rule: str           # e.g. "nelson_1"
    description: str


@dataclass(frozen=True, repr=False)
class ControlChartResult(QCResult):
    """Result of a control-chart analysis (immutable)."""

    kind: str
    inferred: bool
    rules: str
    n: int | None
    location_label: str
    location_points: np.ndarray
    location_cl: float
    location_ucl: np.ndarray
    location_lcl: np.ndarray
    disp_label: str | None
    disp_points: np.ndarray | None
    disp_cl: float | None
    disp_ucl: np.ndarray | None
    disp_lcl: np.ndarray | None
    labels: tuple = ()
    violations: list[Violation] = field(default_factory=list)
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        tag = "inferred" if self.inferred else "specified"
        return f"Control Chart: {self.kind} ({tag}); rules={self.rules}"

    def _summary_lines(self) -> list[str]:
        def lim(a):
            v = np.unique(np.round(np.asarray(a, dtype=float), 6))
            return f"{v[0]:.5g}" if v.size == 1 else "varies"

        lines = [
            f"{self.location_label}: CL={self.location_cl:.5g}  "
            f"UCL={lim(self.location_ucl)}  LCL={lim(self.location_lcl)}",
        ]
        if self.disp_label is not None:
            lines.append(
                f"{self.disp_label}: CL={self.disp_cl:.5g}  "
                f"UCL={lim(self.disp_ucl)}  LCL={lim(self.disp_lcl)}"
            )
        lines.append("")
        if self.violations:
            lines.append(f"Out-of-control signals: {len(self.violations)}")
            for v in self.violations:
                lines.append(f"  point {v.point} ({v.chart}): {v.rule} - {v.description}")
        else:
            lines.append("Out-of-control signals: none (process in control)")
        return lines

    def summary(self) -> dict:
        """Flat {label: value} dict of the headline numbers (dashboard-ready)."""
        def one(a):
            v = np.unique(np.round(np.asarray(a, dtype=float), 6))
            return float(v[0]) if v.size == 1 else None  # None when limits step

        return {
            "kind": self.kind,
            "rules": self.rules,
            "location_label": self.location_label,
            "location_CL": float(self.location_cl),
            "location_UCL": one(self.location_ucl),
            "location_LCL": one(self.location_lcl),
            "disp_label": self.disp_label,
            "disp_CL": None if self.disp_cl is None else float(self.disp_cl),
            "n_signals": len(self.violations),
            "in_control": not self.violations,
        }

    def _render_standalone(self, fig, kind, **kwargs):
        from . import plotting
        if self.disp_label is not None and kind in (None, "default"):
            ax_top = fig.add_subplot(211)
            ax_bot = fig.add_subplot(212, sharex=ax_top)
            plotting.control_panel(ax_top, self, panel="location")
            plotting.control_panel(ax_bot, self, panel="dispersion")
        else:
            ax = fig.add_subplot(111)
            self._render_axes(ax, kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        from . import plotting
        panel = "dispersion" if kind == "dispersion" else "location"
        plotting.control_panel(ax, self, panel=panel)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _assumption_step(a: AssumptionCheck) -> Step:
    return Step(
        operation=f"assumption:{a.name}",
        params={"test": a.test, "passed": a.passed, "magnitude": a.magnitude,
                "reliability": a.reliability, "p_value": a.p_value, "statistic": a.statistic},
        n_affected=None, timestamp=_now(),
    )


# --------------------------------------------------------------------------- #
# Run-rules engine
# --------------------------------------------------------------------------- #
def _runs_same_side(signs: np.ndarray, length: int) -> list[int]:
    """Indices completing a run of `length` consecutive same-sign points."""
    out = []
    for i in range(length - 1, len(signs)):
        window = signs[i - length + 1:i + 1]
        if np.all(window > 0) or np.all(window < 0):
            out.append(i)
    return out


def _trend(x: np.ndarray, length: int) -> list[int]:
    out = []
    for i in range(length - 1, len(x)):
        w = x[i - length + 1:i + 1]
        d = np.diff(w)
        if np.all(d > 0) or np.all(d < 0):
            out.append(i)
    return out


def _alternating(x: np.ndarray, length: int) -> list[int]:
    out = []
    for i in range(length - 1, len(x)):
        w = x[i - length + 1:i + 1]
        d = np.diff(w)
        if np.all(d != 0) and np.all(np.sign(d[1:]) == -np.sign(d[:-1])):
            out.append(i)
    return out


def _k_of_m_beyond(z: np.ndarray, k: int, m: int, thresh: float) -> list[int]:
    """k of m consecutive points beyond `thresh` sigma on the same side."""
    out = []
    for i in range(m - 1, len(z)):
        w = z[i - m + 1:i + 1]
        hi = np.sum(w > thresh)
        lo = np.sum(w < -thresh)
        if (hi >= k or lo >= k) and abs(z[i]) > thresh:
            out.append(i)
    return out


def _apply_rules(points: np.ndarray, cl: float, sigma: float, ruleset: str, chart: str) -> list[Violation]:
    if sigma <= 0 or not np.isfinite(sigma):
        return []
    z = (points - cl) / sigma
    signs = np.sign(points - cl)
    found: dict[int, tuple[str, str]] = {}

    def add(idxs, rule, desc):
        for i in idxs:
            found.setdefault(i, (rule, desc))

    # Rule 1 is common to both rulesets.
    add([i for i in range(len(z)) if abs(z[i]) > 3], f"{ruleset}_1",
        "one point beyond 3 sigma")

    if ruleset == "nelson":
        add(_runs_same_side(signs, 9), "nelson_2", "nine points in a row on one side of CL")
        add(_trend(points, 6), "nelson_3", "six points steadily increasing or decreasing")
        add(_alternating(points, 14), "nelson_4", "fourteen points alternating up/down")
        add(_k_of_m_beyond(z, 2, 3, 2.0), "nelson_5", "two of three points beyond 2 sigma (same side)")
        add(_k_of_m_beyond(z, 4, 5, 1.0), "nelson_6", "four of five points beyond 1 sigma (same side)")
        # Rule 7: 15 in a row within 1 sigma
        for i in range(14, len(z)):
            if np.all(np.abs(z[i - 14:i + 1]) < 1):
                found.setdefault(i, ("nelson_7", "fifteen points in a row within 1 sigma"))
        # Rule 8: 8 in a row beyond 1 sigma (both sides)
        for i in range(7, len(z)):
            if np.all(np.abs(z[i - 7:i + 1]) > 1):
                found.setdefault(i, ("nelson_8", "eight points in a row beyond 1 sigma"))
    elif ruleset == "western_electric":
        add(_k_of_m_beyond(z, 2, 3, 2.0), "western_electric_2", "two of three beyond 2 sigma (same side)")
        add(_k_of_m_beyond(z, 4, 5, 1.0), "western_electric_3", "four of five beyond 1 sigma (same side)")
        add(_runs_same_side(signs, 8), "western_electric_4", "eight points in a row on one side of CL")
    else:
        raise ValueError("rules must be 'nelson' or 'western_electric'.")

    return [
        Violation(point=i + 1, value=float(points[i]), chart=chart, rule=r, description=d)
        for i, (r, d) in sorted(found.items())
    ]


# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #
def _beyond_limits(points, ucl, lcl, ruleset: str, chart: str) -> list[Violation]:
    """Rule 1 (beyond control limits) on a chart's points - used for the dispersion
    (R/S/MR) panel so a subgroup whose RANGE blows up is flagged even when its mean
    stays inside the X-bar limits."""
    pts = np.asarray(points, dtype=float)
    ucl_a = np.broadcast_to(np.asarray(ucl, dtype=float), pts.shape)
    lcl_a = np.broadcast_to(np.asarray(lcl, dtype=float), pts.shape)
    out: list[Violation] = []
    for i in range(pts.size):
        if not np.isfinite(pts[i]):
            continue
        if pts[i] > ucl_a[i] or pts[i] < lcl_a[i]:
            out.append(Violation(point=i + 1, value=float(pts[i]), chart=chart,
                                 rule=f"{ruleset}_1", description="one point beyond control limits"))
    return out


def _infer_variable_kind(sizes: tuple[int, ...]) -> str:
    if all(s == 1 for s in sizes):
        return "i_mr"
    n = sizes[0]
    return "xbar_r" if n <= 10 else "xbar_s"


def compute(qc: QCData, *, kind: str | None = None, rules: str = "nelson",
            n: "str | int | None" = None) -> ControlChartResult:
    """Compute a control chart.

    Parameters
    ----------
    qc : QCData
    kind : str or None, optional
        ``i_mr``, ``xbar_r``, ``xbar_s``, ``p``, ``np``, ``c``, ``u``. If
        ``None``, a variable chart is inferred from subgroup size (the inferred
        choice is recorded). Attribute charts must be requested explicitly.
    rules : str, optional
        ``"nelson"`` (default) or ``"western_electric"``.
    n : str or int or None, optional
        Sample size for attribute charts (p/np/u): a column name (per-point sizes,
        producing stepped limits when they vary), an int (constant size), or None
        (falls back to a ``size`` role, else 1). Ignored by variable charts.

    Returns
    -------
    ControlChartResult
    """
    if kind == "i":
        kind = "i_mr"  # 'i' (individuals) is an alias for the individuals + moving-range chart
    if kind is not None and kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS} or None; got {kind!r}.")
    if rules not in ("nelson", "western_electric"):
        raise ValueError("rules must be 'nelson' or 'western_electric'.")

    checks: list[AssumptionCheck] = []

    if kind in _ATTRIBUTE:
        return _compute_attribute(qc, kind, rules, checks, n=n)

    # An explicit individuals chart on a plain measure column defaults to size-1
    # subgroups (each row is its own subgroup) rather than erroring - 'i_mr' IS the
    # individuals chart, so requiring subgroup_size=1 as well would be friction.
    if kind == "i_mr":
        meta = qc.meta
        if not ("subgroup" in meta.roles or "time" in meta.roles
                or meta.subgroup_size is not None):
            from dataclasses import replace as _replace
            qc = QCData(_frame=qc._frame, meta=_replace(meta, subgroup_size=1),
                        history=qc.history)

    sg = qc.subgroups()
    inferred = kind is None
    if inferred:
        kind = _infer_variable_kind(sg.sizes)

    if kind == "i_mr":
        return _compute_imr(qc, sg, inferred, rules, checks)
    if not sg.equal_n:
        raise ValueError(
            f"{kind} requires constant subgroup size; sizes vary ({set(sg.sizes)})."
        )
    n = sg.n
    means = np.array([g.mean() for g in sg.groups], dtype=float)
    grand = float(means.mean())

    if kind == "xbar_r":
        disp = np.array([g.max() - g.min() for g in sg.groups], dtype=float)
        dbar = float(disp.mean())
        A2 = control_constant("A2", n)
        D3 = control_constant("D3", n)
        D4 = control_constant("D4", n)
        loc_ucl, loc_lcl = grand + A2 * dbar, grand - A2 * dbar
        disp_ucl, disp_lcl = D4 * dbar, D3 * dbar
        disp_label = "R"
    else:  # xbar_s
        disp = np.array([g.std(ddof=1) for g in sg.groups], dtype=float)
        dbar = float(disp.mean())
        A3 = control_constant("A3", n)
        B3 = control_constant("B3", n)
        B4 = control_constant("B4", n)
        loc_ucl, loc_lcl = grand + A3 * dbar, grand - A3 * dbar
        disp_ucl, disp_lcl = B4 * dbar, B3 * dbar
        disp_label = "S"

    sigma_loc = (loc_ucl - grand) / 3.0
    checks.append(_assume.check_independence(means))
    violations = _apply_rules(means, grand, sigma_loc, rules, "location")
    # Beyond-limits (Rule 1) on the dispersion chart too - a single inflated subgroup
    # range/sd must surface even if the subgroup mean is in-limits.
    violations += _beyond_limits(disp, disp_ucl, disp_lcl, rules, "dispersion")

    step = Step(operation="control_chart",
                params={"kind": kind, "inferred": inferred, "rules": rules, "n": n},
                n_affected=len(means), timestamp=_now())
    history = qc.history + (step,) + tuple(_assumption_step(a) for a in checks)

    k = len(means)
    return ControlChartResult(
        kind=kind, inferred=inferred, rules=rules, n=n,
        location_label="Xbar", location_points=means, location_cl=grand,
        location_ucl=np.full(k, loc_ucl), location_lcl=np.full(k, loc_lcl),
        disp_label=disp_label, disp_points=disp, disp_cl=dbar,
        disp_ucl=np.full(k, disp_ucl), disp_lcl=np.full(k, disp_lcl),
        labels=sg.labels, violations=violations, assumptions=checks, history=history,
    )


def _compute_imr(qc, sg, inferred, rules, checks):
    x = np.array([g[0] for g in sg.groups], dtype=float)
    xbar = float(x.mean())
    mr = np.abs(np.diff(x))
    mrbar = float(mr.mean()) if mr.size else 0.0
    d2 = control_constant("d2", 2)
    sigma = mrbar / d2 if mrbar > 0 else 0.0
    loc_ucl, loc_lcl = xbar + 3 * sigma, xbar - 3 * sigma
    D4 = control_constant("D4", 2)
    mr_ucl = D4 * mrbar
    checks.append(_assume.check_independence(x))
    violations = _apply_rules(x, xbar, sigma, rules, "location")
    k = len(x)
    # MR series aligned to points 2..k (length k-1); pad leading NaN for plotting.
    mr_points = np.concatenate([[np.nan], mr]) if mr.size else np.array([np.nan])
    # Beyond-limits (Rule 1) on the MR chart - flags an inflated moving range.
    violations += _beyond_limits(mr_points, mr_ucl, 0.0, rules, "dispersion")
    step = Step(operation="control_chart",
                params={"kind": "i_mr", "inferred": inferred, "rules": rules, "n": 1},
                n_affected=k, timestamp=_now())
    history = qc.history + (step,) + tuple(_assumption_step(a) for a in checks)
    return ControlChartResult(
        kind="i_mr", inferred=inferred, rules=rules, n=1,
        location_label="Individual", location_points=x, location_cl=xbar,
        location_ucl=np.full(k, loc_ucl), location_lcl=np.full(k, loc_lcl),
        disp_label="MR", disp_points=mr_points, disp_cl=mrbar,
        disp_ucl=np.full(k, mr_ucl), disp_lcl=np.zeros(k),
        labels=sg.labels, violations=violations, assumptions=checks, history=history,
    )


def _compute_attribute(qc, kind, rules, checks, n=None):
    measure = qc.meta.measure
    counts = qc._frame[measure].to_numpy(dtype=float)
    if n is None:
        size_role = qc.meta.roles.get("size")
        sizes = (qc._frame[size_role].to_numpy(dtype=float)
                 if size_role is not None else np.ones_like(counts))
    elif isinstance(n, str):
        if n not in qc._frame.columns:
            raise ValueError(f"n={n!r} is not a column in this QCData.")
        sizes = qc._frame[n].to_numpy(dtype=float)
    elif np.isscalar(n):
        sizes = np.full_like(counts, float(n))
    else:
        sizes = np.asarray(n, dtype=float)
    k = counts.size

    if kind in ("p", "np"):
        if kind == "np" and len(np.unique(sizes)) != 1:
            raise ValueError("np-chart requires constant subgroup size.")
        pbar = counts.sum() / sizes.sum()
        if kind == "p":
            points = counts / sizes
            cl = pbar
            sd = np.sqrt(pbar * (1 - pbar) / sizes)
        else:
            points = counts
            n0 = sizes[0]
            cl = n0 * pbar
            sd = np.full(k, np.sqrt(n0 * pbar * (1 - pbar)))
        checks.append(_assume.check_dispersion(counts, sizes, family="binomial"))
        loc_label = "Proportion (p)" if kind == "p" else "Count (np)"
    else:  # c, u
        if kind == "c":
            points = counts
            cbar = counts.mean()
            cl = cbar
            sd = np.full(k, np.sqrt(cbar))
            disp_sizes = np.ones_like(counts)
        else:
            ubar = counts.sum() / sizes.sum()
            points = counts / sizes
            cl = ubar
            sd = np.sqrt(ubar / sizes)
            disp_sizes = sizes
        checks.append(_assume.check_dispersion(counts, disp_sizes, family="poisson"))
        loc_label = "Count (c)" if kind == "c" else "Rate (u)"

    ucl = cl + 3 * sd
    lcl = np.clip(cl - 3 * sd, 0, None)
    # Per-point sigma varies with n; flag only points beyond their own limits.
    violations = [
        Violation(point=i + 1, value=float(points[i]), chart="location",
                  rule=f"{rules}_1", description="one point beyond 3 sigma")
        for i in range(k) if points[i] > ucl[i] or points[i] < lcl[i]
    ]
    step = Step(operation="control_chart",
                params={"kind": kind, "inferred": False, "rules": rules},
                n_affected=k, timestamp=_now())
    history = qc.history + (step,) + tuple(_assumption_step(a) for a in checks)
    return ControlChartResult(
        kind=kind, inferred=False, rules=rules, n=None,
        location_label=loc_label, location_points=points, location_cl=float(cl),
        location_ucl=ucl, location_lcl=lcl,
        disp_label=None, disp_points=None, disp_cl=None, disp_ucl=None, disp_lcl=None,
        labels=tuple(range(1, k + 1)), violations=violations,
        assumptions=checks, history=history,
    )


def compute_short_run(qc: QCData, by: str, target=None, *, rules: str = "nelson"):
    """Standardized / short-run chart: chart the standardized deviation from each
    part's target so multiple part numbers share one chart.

    ``by`` names the part-number column; ``target`` is a scalar, a {part: target}
    map, or None (each part's own mean is the target). Each value is standardized
    z = (x - target_part) / sigma_part with sigma_part the part's within-spread.

    Surfacing: the chart assumes homogeneous within-part variance across the
    pooled parts; a part whose spread departs from the pool is flagged, because
    that breaks the shared standardized scale.
    """
    frame = qc.frame
    measure = qc.meta.measure
    if by not in frame.columns:
        raise ValueError(f"part column {by!r} not found in the frame.")
    sub = frame[[measure, by]].copy()
    sub[measure] = pd.to_numeric(sub[measure], errors="coerce")
    sub = sub.dropna()
    parts = list(pd.unique(sub[by]))

    z_all, labels, part_sds = [], [], {}
    for p in parts:
        v = sub[sub[by] == p][measure].to_numpy(dtype=float)
        if isinstance(target, dict):
            tgt = float(target[p])
        elif target is not None:
            tgt = float(target)
        else:
            tgt = float(np.mean(v))
        sd = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
        part_sds[p] = sd
        if sd > 0:
            z_all.extend(((v - tgt) / sd).tolist())
        else:
            z_all.extend(np.zeros_like(v).tolist())
        labels.extend([p] * v.size)
    z = np.array(z_all, dtype=float)
    k = z.size

    # homogeneity of within-part variance across the pooled parts (the assumption)
    usable = [sub[sub[by] == p][measure].to_numpy(dtype=float) for p in parts]
    usable = [g for g in usable if g.size > 1]
    checks = []
    if len(usable) >= 2:
        checks.append(_assume.check_homogeneity(usable))
    violations = _apply_rules(z, 0.0, 1.0, rules, "location")
    step = Step(operation="short_run_chart",
                params={"by": by, "target": "per-part-mean" if target is None else target,
                        "parts": len(parts)}, n_affected=k, timestamp=_now())
    history = qc.history + (step,) + tuple(_assumption_step(a) for a in checks)
    return ControlChartResult(
        kind="short_run", inferred=False, rules=rules, n=1,
        location_label="Standardized (z)", location_points=z, location_cl=0.0,
        location_ucl=np.full(k, 3.0), location_lcl=np.full(k, -3.0),
        disp_label=None, disp_points=None, disp_cl=None, disp_ucl=None, disp_lcl=None,
        labels=tuple(labels), violations=violations, assumptions=checks, history=history,
    )
