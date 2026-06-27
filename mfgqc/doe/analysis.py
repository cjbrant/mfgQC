"""DOE analysis (post-data): ``QCData.doe`` and the immutable ``DOEResult``.

The analysis sits on top of the existing regression engine: it codes the design
to -1/+1, builds the model matrix (main effects and interactions up to ``order``),
fits through :func:`mfgqc.regression.compute_regression`, and reports effects as
``2 * coefficient``. It then forks on whether a pure-error estimate exists:

- replicated / center points / pooled terms -> residual df > 0 -> standard
  t and F significance from the regression engine.
- saturated (zero residual df) -> no pure error; significance from Lenth's PSE
  and the half-normal plot, with the no-pure-error condition surfaced, never a
  fabricated error term.

Confounding (alias list for regular designs, correlation map in general) and a
set of adequacy flags are surfaced, never silently resolved. The one opt-in is
explicit model reduction via ``order=`` (or ``reduce=``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .. import assumptions as _assume
from .._result import QCResult
from ..assumptions import AssumptionCheck
from ..data import QCData, Step
from . import alias as _alias
from .significance import LenthResult, lenth

_ALPHA = 0.05


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _term_label(term: tuple[str, ...]) -> str:
    return ":".join(term)


def _model_terms(factors: list[str], order: int) -> list[tuple[str, ...]]:
    terms: list[tuple[str, ...]] = []
    for r in range(1, order + 1):
        terms.extend(combinations(factors, r))
    return terms


def _detect_fraction(factor_names: list[str], matrix: np.ndarray):
    """Detect regular fractional structure from a coded design matrix (v2 4.2a).

    A subset of factor columns whose elementwise product is constant (all +1 or
    all -1, i.e. equal to I up to sign) is a defining word. The set of all such
    words is the defining relation; from it come the resolution, a minimal
    generator set, and the alias list. Returns ``(group, resolution, generators,
    alias_lines)`` or ``None`` for an unconfounded (full) design.
    """
    k = len(factor_names)
    nonidentity: list[frozenset] = []
    for r in range(1, k + 1):
        for combo in combinations(range(k), r):
            col = np.ones(matrix.shape[0])
            for j in combo:
                col = col * matrix[:, j]
            if np.all(col > 0) or np.all(col < 0):
                nonidentity.append(frozenset(factor_names[j] for j in combo))
    if not nonidentity:
        return None

    group = [frozenset()] + sorted(nonidentity, key=lambda w: (len(w), sorted(w)))
    resolution = min(len(w) for w in nonidentity)
    generators = _minimal_generators(nonidentity, factor_names)
    alias_lines = tuple(_alias.alias_list(factor_names, group))
    return group, resolution, tuple(generators), alias_lines


def _minimal_generators(words, factor_names: list[str]) -> list[str]:
    """A minimal independent generating set of the defining words, rendered as
    generator equations (``"E=ABCD"``: the highest-index factor as a product of
    the rest). Independence is taken over GF(2) on the factor-presence vectors."""
    idx = {f: i for i, f in enumerate(factor_names)}
    basis: dict[int, int] = {}        # high bit -> reduced vector
    chosen: list[str] = []
    for w in sorted(words, key=lambda w: (len(w), sorted(w))):
        vec = 0
        for f in w:
            vec |= 1 << idx[f]
        v = vec
        while v:
            hb = v.bit_length() - 1
            if hb in basis:
                v ^= basis[hb]
            else:
                basis[hb] = v
                lhs = max(w, key=lambda f: idx[f])
                rhs = "".join(f for f in factor_names if f in w and f != lhs)
                chosen.append(f"{lhs}={rhs}")
                break
    return chosen


def _default_order(k: int, n: int, design) -> int:
    """full for a full factorial / explicit; otherwise the largest interaction
    order whose model still fits in n runs (caps a fractional design at the
    saturated model and never asks for more columns than runs)."""
    if design is not None:
        return k if design.kind == "full" else 2
    order = 1
    for cand in range(1, k + 1):
        params = 1 + sum(len(list(combinations(range(k), r))) for r in range(1, cand + 1))
        if params <= n:
            order = cand
        else:
            break
    return order


# --------------------------------------------------------------------------- #
# Adequacy flags (binary + context + recommendation; warn, never resolve)
# --------------------------------------------------------------------------- #
def _adequacy_flags(*, kind, resolution, order, n_center, df_resid, max_offdiag, n) -> list[AssumptionCheck]:
    flags: list[AssumptionCheck] = []
    fractional = kind == "fractional"

    # aliasing / resolution
    if fractional:
        res_label = {3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII"}.get(resolution, str(resolution))
        flags.append(AssumptionCheck(
            "aliasing", "design resolution", float(resolution or 0), None,
            False, float(resolution or 0), "resolution", "ok", n,
            f"Fractional design (resolution {res_label}); effects are aliased - read the "
            "alias list / correlation map before attributing an effect."))
    else:
        flags.append(AssumptionCheck(
            "aliasing", "design resolution", float("inf"), None,
            True, None, "resolution", "ok", n, None))

    # insufficient resolution for the intended model order
    if fractional and resolution is not None:
        clear = resolution > 2 * order      # terms up to `order` not aliased with each other
        flags.append(AssumptionCheck(
            "resolution_for_model", "resolution vs fit order", float(resolution), None,
            bool(clear), float(order), "fit order", "ok", n,
            None if clear else
            (f"Resolution {resolution} is insufficient for an order-{order} model: terms in the "
             "model are aliased with each other - some estimates are sums of effects.")))

    # center points / curvature checkable
    flags.append(AssumptionCheck(
        "center_points", "curvature checkable", float(n_center), None,
        bool(n_center > 0), float(n_center), "center points", "ok", n,
        None if n_center > 0 else
        "No center points: curvature cannot be checked - add center runs to test for a "
        "quadratic response (the trigger to consider a response-surface design)."))

    # pure error for t/F
    flags.append(AssumptionCheck(
        "pure_error", "residual df", float(df_resid), None,
        bool(df_resid > 0), float(df_resid), "residual df", "ok", n,
        None if df_resid > 0 else
        "No pure-error estimate (saturated model, zero residual df); significance from Lenth "
        "PSE / half-normal, not t or F. Replicate or add center points for a pure-error test."))

    # orthogonality / VIF of the fitted model matrix
    orthogonal = (max_offdiag is None) or (max_offdiag < 1e-9)
    flags.append(AssumptionCheck(
        "orthogonality", "max |off-diagonal corr|",
        0.0 if max_offdiag is None else float(max_offdiag), None,
        bool(orthogonal), None if max_offdiag is None else float(max_offdiag),
        "model-matrix corr", "ok", n,
        None if orthogonal else
        (f"Model matrix is non-orthogonal (max |corr|={max_offdiag:.2f}); coefficient estimates "
         "are correlated - interpret effects jointly.")))
    return flags


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class DOEResult(QCResult):
    """Result of a two-level DOE analysis (immutable)."""

    response: str
    factors: tuple[str, ...]
    terms: tuple[str, ...]              # model terms (no intercept), display labels
    coef: dict                         # term -> half effect (coefficient)
    effect: dict                       # term -> effect (2 * coefficient)
    se: dict
    t: dict
    p_value: dict
    n: int
    df_resid: int
    fit_kind: str                      # "replicated" | "saturated"
    significant: tuple[str, ...] = ()  # replicated: t-test p < alpha
    lenth: LenthResult | None = None
    active: tuple[str, ...] = ()        # saturated: Lenth SME-active
    possibly_active: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()       # alias list lines (fractional)
    alias_of: dict = field(default_factory=dict)   # term -> aliased-with note
    resolution: int | None = None
    generators: tuple[str, ...] = ()
    intercept: float = 0.0
    r_squared: float = float("nan")
    adequacy: list = field(default_factory=list)
    curvature: AssumptionCheck | None = None
    _design_matrix: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _y: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    _term_words: tuple = field(repr=False, default_factory=tuple)
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    # ---- reporting -------------------------------------------------------
    def _title(self) -> str:
        kind = "full" if not self.generators else "fractional"
        return f"DOE ({kind} 2-level): {self.response} ~ {'*'.join(self.factors)}"

    def _summary_lines(self) -> list[str]:
        saturated = self.fit_kind == "saturated"
        lines = [f"n = {self.n}   model terms = {len(self.terms)}   df(resid) = {self.df_resid}"]
        if self.resolution is not None:
            res = {3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII"}.get(self.resolution, str(self.resolution))
            lines.append(f"fractional, resolution {res}; generators: {', '.join(self.generators)}")
        if saturated:
            lines.append("no pure-error estimate; significance from Lenth PSE / half-normal")
        else:
            lines.append(f"R^2 = {self.r_squared:.4g}")
        lines.append("")
        header = f"{'term':<10}{'effect':>12}{'half effect':>14}"
        if not saturated:
            header += f"{'std err':>12}{'t':>10}{'p':>10}"
        else:
            header += f"{'|t_Lenth|':>12}{'verdict':>16}"
        lines.append(header)
        for term in self.terms:
            row = (f"{term:<10}{self.effect[term]:>12.5g}{self.coef[term]:>14.5g}")
            if not saturated:
                p = self.p_value[term]
                sig = " *" if (p is not None and np.isfinite(p) and p < _ALPHA) else ""
                row += (f"{self.se[term]:>12.4g}{self.t[term]:>10.3g}"
                        f"{p:>10.3g}{sig}")
            else:
                tl = abs(self.lenth.pseudo_t[term]) if self.lenth else float("nan")
                row += f"{tl:>12.3g}{self.lenth.labels[term]:>16}"
            lines.append(row)
        if saturated and self.lenth:
            lines.append("")
            lines.append(f"Lenth PSE = {self.lenth.pse:.4g}   ME = {self.lenth.me:.4g}   "
                         f"SME = {self.lenth.sme:.4g}   (ME/SME are Tier-2 secondary)")
            lines.append(f"active (|effect| > SME): {', '.join(self.active) or '(none)'}")
            if self.possibly_active:
                lines.append(f"possibly active (ME < |effect| <= SME): {', '.join(self.possibly_active)}")
        else:
            lines.append("")
            lines.append(f"significant (p < {_ALPHA}): {', '.join(self.significant) or '(none)'}")
        if self.aliases:
            lines.append("")
            lines.append("alias structure:")
            lines += [f"  {a}" for a in self.aliases]
        if self.adequacy:
            lines.append("")
            lines.append("Adequacy flags:")
            for fl in self.adequacy:
                tag = "PASS" if fl.passed else "WARN"
                lines.append(f"  [{tag}] {fl.name}")
                if not fl.passed and fl.recommendation:
                    lines.append(f"         {fl.recommendation}")
        return lines

    def summary(self) -> dict:
        out: dict = {"response": self.response, "n": self.n, "df_resid": self.df_resid,
                     "fit_kind": self.fit_kind, "intercept": self.intercept}
        for term in self.terms:
            out[f"effect[{term}]"] = self.effect[term]
            out[f"coef[{term}]"] = self.coef[term]
            if self.fit_kind == "replicated":
                out[f"p[{term}]"] = self.p_value[term]
        if self.fit_kind == "replicated":
            out["significant"] = list(self.significant)
        else:
            out["active"] = list(self.active)
            out["possibly_active"] = list(self.possibly_active)
            if self.lenth:
                out["lenth_pse"] = self.lenth.pse
                out["lenth_me"] = self.lenth.me
                out["lenth_sme"] = self.lenth.sme
        if self.resolution is not None:
            out["resolution"] = self.resolution
        return out

    # ---- interaction crossings -------------------------------------------
    def crossings(self, tol_frac: float = 0.02) -> dict:
        """Two-factor interactions whose surfaces cross over the +/-1 range
        (interaction present). See :func:`interaction_crossings`."""
        return interaction_crossings(self, tol_frac=tol_frac)

    # ---- views -----------------------------------------------------------
    def view(self, ax=None, kind: str | None = None, **kwargs):
        """Render a DOE view. DOE static views default to the LIGHT theme
        (white background, dark marks), scoped to this call."""
        from .. import palette as _pal
        with _pal.using(_pal.LIGHT):
            return super().view(ax=ax, kind=kind, **kwargs)

    def _render_standalone(self, fig, kind, **kwargs):
        from . import views
        views.doe_standalone(self, fig, kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        from . import views
        views.doe_axes(self, ax, kind, **kwargs)


# --------------------------------------------------------------------------- #
# The analysis entry point
# --------------------------------------------------------------------------- #
def doe(qc: QCData, design=None, factors=None, order=None, reduce: bool = False) -> DOEResult:
    """Analyze a two-level designed experiment on the QCData measure.

    Parameters
    ----------
    qc : QCData
        The response is the measure / quality role.
    design : Design or None
        A :class:`mfgqc.doe.Design`; supplies the factor names, alias structure,
        and the default model order.
    factors : list of str or None
        Factor column names in the frame (an external matrix). Coded to -1/+1
        from their two observed levels.
    order : int, "full", or None
        Cap on interaction order. Default: full for a full-factorial design, 2
        for a fractional design, otherwise the largest model that fits in n runs.
    """
    if (design is None) == (factors is None):
        raise ValueError("pass exactly one of design= or factors=.")

    frame = qc.frame
    response = qc.meta.measure
    factor_names = list(design.factors) if design is not None else list(factors)
    missing = [c for c in factor_names if c not in frame.columns]
    if missing:
        raise ValueError(f"factor column(s) {missing} not found in the frame.")

    cols = [response] + factor_names
    sub = frame[cols].apply(pd.to_numeric, errors="coerce").dropna()
    y_all = sub[response].to_numpy(dtype=float)

    # Factor-structure guard (rev 2, REQUIRED before coding): a two-level factor
    # must have exactly two observed levels, optionally plus a center point at
    # their exact midpoint. An out-of-range or 3+-level factor (e.g. {-1, 1, 2})
    # is REFUSED, never silently coded: min/max coding would map {-1, 1, 2} to
    # {-1, 0.333, 1} and corrupt the legitimate levels too. Surface, do not coerce.
    for f in factor_names:
        u = np.unique(sub[f].to_numpy(dtype=float))
        ok = (u.size == 2) or (u.size == 3 and np.isclose(u[1], (u[0] + u[2]) / 2.0))
        if not ok:
            raise ValueError(
                f"factor {f!r} is not a clean two-level factor: observed levels "
                f"{[float(x) for x in u]}. A two-level DOE factor needs exactly two "
                f"levels (optionally plus a center point at their midpoint). mfgQC "
                f"refuses to code an out-of-range or multi-level factor rather than "
                f"silently distort it; screen the data (overview()/clean()) and re-run.")

    # code each factor to -1/+1 from its two corner levels (the design min/max);
    # a center point sits at the midpoint and codes to 0.
    coded_all = {}
    for f in factor_names:
        v = sub[f].to_numpy(dtype=float)
        lo, hi = float(np.min(v)), float(np.max(v))
        if hi == lo:
            raise ValueError(f"factor {f!r} has a single level; cannot code it.")
        coded_all[f] = (v - (hi + lo) / 2.0) / ((hi - lo) / 2.0)
    matrix_all = np.column_stack([coded_all[f] for f in factor_names])

    # split off center points (all factors at coded 0): they inform curvature and
    # pure error, never the factorial effect estimates.
    is_center = np.all(np.abs(matrix_all) < 1e-9, axis=1)
    n_center = int(np.sum(is_center))
    fac = ~is_center
    y = y_all[fac]
    coded = {f: coded_all[f][fac] for f in factor_names}
    matrix = matrix_all[fac]
    n = y.size

    k = len(factor_names)
    if order == "full" or (order is None and design is not None and design.kind == "full"):
        order_n = k
    elif order is None:
        order_n = _default_order(k, n, design)
    else:
        order_n = int(order)

    term_words = _model_terms(factor_names, order_n)
    labels = [_term_label(t) for t in term_words]

    # Over-parameterization refusal (v2 4.3): a model is estimable only if its
    # parameter count does not exceed the number of DISTINCT design points. Check
    # distinct runs, not total: a replicated design has enough rows to slip past
    # the regression engine's n<p guard yet still be rank-deficient (it would then
    # return a silent min-norm fit). Refuse with a DOE-aware message, not a raw
    # linear-algebra error.
    n_distinct = int(np.unique(matrix, axis=0).shape[0])
    n_params = 1 + len(labels)
    if n_params > n_distinct:
        raise ValueError(
            f"model has {n_params} parameters but only {n_distinct} distinct runs "
            f"(order={order_n} over {k} factors); it is not estimable. Lower the "
            f"interaction order, replicate the design, or pass a Design so the "
            f"alias-aware reduced model can be fit.")

    # build the model-matrix frame and fit through the regression engine
    model_df = pd.DataFrame({response: y})
    for t, lbl in zip(term_words, labels):
        col = np.ones(n)
        for f in t:
            col = col * coded[f]
        model_df[lbl] = col

    from ..data import _from_dataframe
    from ..regression import compute_regression
    qc_model = _from_dataframe(model_df, measure=response, roles={})
    reg = compute_regression(qc_model, labels)

    coef = {lbl: reg.coef[lbl] for lbl in labels}
    effect = {lbl: 2.0 * reg.coef[lbl] for lbl in labels}
    se = {lbl: reg.se[lbl] for lbl in labels}
    t = {lbl: reg.t[lbl] for lbl in labels}
    p_value = {lbl: reg.p_value[lbl] for lbl in labels}
    df_resid = reg.df_resid

    # orthogonality of the fitted columns
    model_cols = np.column_stack([model_df[lbl].to_numpy() for lbl in labels])
    if model_cols.shape[1] > 1:
        cmat = np.corrcoef(model_cols, rowvar=False)
        off = cmat[~np.eye(cmat.shape[0], dtype=bool)]
        max_offdiag = float(np.nanmax(np.abs(off))) if off.size else 0.0
    else:
        max_offdiag = 0.0

    # significance fork
    fit_kind = "replicated" if df_resid > 0 else "saturated"
    significant: tuple = ()
    lenth_res = None
    active: tuple = ()
    possibly: tuple = ()
    diagnostics: list = []
    if fit_kind == "replicated":
        significant = tuple(lbl for lbl in labels
                            if np.isfinite(p_value[lbl]) and p_value[lbl] < _ALPHA)
        diagnostics = _doe_diagnostics(reg._resid, reg._fitted, matrix, factor_names, df_resid)
    else:
        lenth_res = lenth(effect)
        active = tuple(lbl for lbl in labels if lenth_res.labels[lbl] == "active")
        possibly = tuple(lbl for lbl in labels if lenth_res.labels[lbl] == "possibly_active")

    # alias attribution. From a Design (fractional) the structure is known; on the
    # external factors= path, detect a fraction from the data (v2 4.2a) and surface
    # the same generators / defining relation / resolution / alias list, computed
    # from the design structure and independent of the fit order.
    aliases: tuple = ()
    alias_of: dict = {}
    resolution = None
    generators: tuple = ()
    group = None
    if design is not None and design.kind == "fractional":
        aliases = design.aliases
        resolution = design.resolution
        generators = design.generators
        group = _alias.defining_group([_alias.parse_word(eq.replace(" ", "").replace("=", ""))
                                       for eq in design.generators])
    elif design is None:
        det = _detect_fraction(factor_names, matrix)
        if det is not None:
            group, resolution, generators, aliases = det
    if group is not None:
        for t, lbl in zip(term_words, labels):
            others = sorted((_alias.multiply(frozenset(t), g) for g in group if g),
                            key=lambda w: (len(w), sorted(w)))
            note = " = ".join(_alias.word_str(w, factor_names) for w in others if w)
            if note:
                alias_of[lbl] = note

    # curvature / lack-of-fit (center points present): compare the factorial-point
    # mean to the center-point mean against pure error from the center replicates.
    curvature = None
    if n_center >= 1:
        curvature = _curvature_check(y, y_all[is_center])

    adequacy = _adequacy_flags(
        kind=("fractional" if generators else "full"), resolution=resolution,
        order=order_n, n_center=n_center, df_resid=df_resid, max_offdiag=max_offdiag, n=n)
    if curvature is not None:
        adequacy.append(curvature)

    step = Step(operation="doe",
                params={"response": response, "factors": factor_names, "order": order_n,
                        "fit_kind": fit_kind},
                n_affected=n, timestamp=_now())
    history = qc.history + (step,)

    return DOEResult(
        response=response, factors=tuple(factor_names), terms=tuple(labels),
        coef=coef, effect=effect, se=se, t=t, p_value=p_value,
        n=int(n), df_resid=int(df_resid), fit_kind=fit_kind,
        significant=significant, lenth=lenth_res, active=active, possibly_active=possibly,
        aliases=aliases, alias_of=alias_of, resolution=resolution, generators=generators,
        intercept=float(reg.coef["intercept"]), r_squared=float(reg.r_squared),
        adequacy=adequacy, curvature=curvature,
        _design_matrix=matrix, _y=y, _term_words=tuple(term_words),
        assumptions=diagnostics, history=history,
    )


def _curvature_check(y_factorial: np.ndarray, y_center: np.ndarray) -> AssumptionCheck:
    """Pure-quadratic curvature / lack-of-fit: compare the factorial-point mean to
    the center-point mean. ``passed`` is the direct F test (no significant
    curvature) when center replicates give pure error; with a single center point
    the contrast is reported but cannot be tested."""
    from scipy import stats as _st
    nf, nc = y_factorial.size, y_center.size
    ybar_f, ybar_c = float(np.mean(y_factorial)), float(np.mean(y_center))
    ss_curv = (nf * nc) / (nf + nc) * (ybar_f - ybar_c) ** 2
    if nc >= 2:
        df_pe = nc - 1
        ms_pe = float(np.sum((y_center - ybar_c) ** 2)) / df_pe
        f_stat = (ss_curv / 1.0) / ms_pe if ms_pe > 0 else float("inf")
        p = float(_st.f.sf(f_stat, 1, df_pe)) if np.isfinite(f_stat) else 0.0
        passed = p >= _ALPHA
        rec = None if passed else (
            f"Significant curvature (factorial mean {ybar_f:.4g} vs center {ybar_c:.4g}, "
            f"F={f_stat:.3g}, p={p:.3g}); the response is not planar - consider a "
            "response-surface design (CCD / Box-Behnken) to fit the quadratic.")
        return AssumptionCheck("curvature", "center-point F", float(f_stat), p,
                               bool(passed), float(ss_curv), "curvature SS",
                               "low_power" if nc < 4 else "ok", nf + nc, rec)
    return AssumptionCheck("curvature", "center-point contrast", float("nan"), None,
                           True, float(ss_curv), "curvature SS", "low_power", nf + nc,
                           "Single center point: curvature is estimated but cannot be tested - "
                           "replicate the center to get a pure-error denominator.")


def interaction_crossings(result, tol_frac: float = 0.02) -> dict:
    """Flag two-factor interactions whose surfaces cross over the +/-1 range.

    For a two-level model the surface over two factors is the bilinear twist; the
    two conditioning lines cross when the interaction effect departs from zero.
    An interaction is a crossing when ``|effect|`` exceeds ``tol_frac`` of the
    response range (adaptive 2% tolerance). Parallel surfaces (no interaction)
    are not flagged.
    """
    twofi = [t for t in result.terms if t.count(":") == 1]
    fitted = _fitted_corner_range(result)
    z_range = fitted if fitted > 0 else 1.0
    tol = tol_frac * z_range
    return {t: bool(abs(result.effect[t]) > tol) for t in twofi}


def _fitted_corner_range(result) -> float:
    """Range of the fitted response over the design corners (for the crossing
    tolerance scale)."""
    mat = result._design_matrix
    pred = np.full(mat.shape[0], result.intercept)
    for t, lbl in zip(result._term_words, result.terms):
        col = np.ones(mat.shape[0])
        for f in t:
            col = col * mat[:, result.factors.index(f)]
        pred = pred + result.coef[lbl] * col
    return float(np.max(pred) - np.min(pred))


def _doe_reliability(df_resid: int) -> str:
    """Reliability of a residual check given the residual df (v2 4.6 tempering).
    Below ~8 residual df the normality/variance tests are not dependable."""
    if df_resid < 4:
        return "low_power"
    if df_resid < 8:
        return "low_power"
    return "ok"


def _doe_constant_variance(resid: np.ndarray, fitted: np.ndarray,
                           df_resid: int) -> AssumptionCheck:
    """Primary constant-variance verdict (v2 4.6): Breusch-Pagan regressing the
    squared residuals on the fitted values. This catches mean-linked
    heteroscedasticity (spread that trends with the response level, the kind a
    response transform addresses); `passed` follows it at conventional alpha. It
    replaces corr(|resid|, fitted), which over-rejects at small residual df."""
    n = resid.size
    rel = _doe_reliability(df_resid)
    e2 = resid ** 2
    if np.ptp(fitted) == 0 or np.ptp(e2) == 0:
        # nothing to regress (intercept-only fit, or a near-perfect fit): no
        # mean-variance trend can be estimated.
        return AssumptionCheck("constant_variance", "Breusch-Pagan", float("nan"), None,
                               True, None, "BP R^2", "low_power", n, None)
    ls = stats.linregress(fitted, e2)
    r2 = float(ls.rvalue) ** 2
    lm = float(n * r2)                          # LM statistic ~ chi-square(1)
    p = float(stats.chi2.sf(lm, 1))
    passed = p >= _assume.ALPHA
    rec = None
    if not passed:
        rec = (f"Residual variance trends with the fitted value (Breusch-Pagan LM={lm:.3g}, "
               f"p={p:.3g}); heteroscedastic - consider a response transform (log or sqrt)."
               + (" (low residual df: read as a caution.)" if rel != "ok" else ""))
    return AssumptionCheck("constant_variance", "Breusch-Pagan", lm, p,
                           bool(passed), float(r2), "BP R^2", rel, n, rec)


def _doe_dispersion_effect(resid: np.ndarray, matrix: np.ndarray,
                           factor_names: list[str], df_resid: int) -> AssumptionCheck:
    """Secondary, separate check (v2 4.6): Brown-Forsythe (Levene, median-centered)
    on residual spread across each factor's high/low groups - a *dispersion effect*
    (a factor driving the variance). Reported on its own, not as the constant-
    variance verdict: it is a legitimate DOE finding (model the variance or
    transform), not a failure of the mean model."""
    n = resid.size
    rel = _doe_reliability(df_resid)
    worst_p, worst_f, worst_ratio, worst_stat, n_tested = 1.0, None, 1.0, float("nan"), 0
    for j, f in enumerate(factor_names):
        hi = resid[matrix[:, j] > 0]
        lo = resid[matrix[:, j] < 0]
        if hi.size < 2 or lo.size < 2:
            continue
        # both groups perfectly constant (e.g. a near-exact fit) -> no dispersion
        # to test; skip rather than divide by zero in Levene.
        if np.ptp(hi) == 0 and np.ptp(lo) == 0:
            continue
        try:
            stat, p = stats.levene(hi, lo, center="median")
        except ValueError:
            continue
        if not np.isfinite(p):
            continue
        n_tested += 1
        vh, vl = float(np.var(hi, ddof=1)), float(np.var(lo, ddof=1))
        ratio = (max(vh, vl) / min(vh, vl)) if min(vh, vl) > 0 else float("inf")
        if p < worst_p:
            worst_p, worst_f, worst_ratio, worst_stat = float(p), f, ratio, float(stat)
    if worst_f is None:
        return AssumptionCheck("dispersion_effect", "Brown-Forsythe", float("nan"), None,
                               True, None, "variance ratio", "low_power", n, None)
    # Bonferroni-correct across the factors tested: scanning every factor and
    # reporting the most extreme would otherwise inflate the false-positive rate
    # (the over-rejection v2 4.6 warns against).
    worst_p = min(1.0, worst_p * max(1, n_tested))
    passed = worst_p >= _assume.ALPHA
    rec = None
    if not passed:
        rec = (f"Factor {worst_f!r} drives the residual spread (variance ratio "
               f"{worst_ratio:.3g}, Brown-Forsythe p={worst_p:.3g}); a dispersion effect, not a "
               "mean-model failure - model the variance or transform the response if it matters."
               + (" (low residual df: read as a caution.)" if rel != "ok" else ""))
    return AssumptionCheck("dispersion_effect", "Brown-Forsythe (across factor levels)",
                           worst_stat, worst_p, bool(passed), float(worst_ratio),
                           "variance ratio", rel, n, rec)


def _doe_diagnostics(resid: np.ndarray, fitted: np.ndarray, matrix: np.ndarray,
                     factor_names: list[str], df_resid: int) -> list:
    """Residual diagnostics on the reduced (df>0) model: normality (AD), a
    design-calibrated constant-variance test, and run-order independence (DW).
    `passed` follows the direct test; `reliability` is tempered by the residual df
    and recommendations are DOE-appropriate (transform, curvature/center points,
    missing term), never capability or hypothesis-test boilerplate."""
    from dataclasses import replace

    from ..regression import _independence_check

    rel = _doe_reliability(df_resid)

    norm = _assume.check_normality(resid, context="regression")
    norm = replace(norm, reliability=rel if rel != "ok" else norm.reliability)
    if not norm.passed:
        norm = replace(norm, recommendation=(
            f"Residuals are not normal (AD={norm.statistic:.3g}); consider a response "
            "transform or a missing term - not a capability or hypothesis-test remedy."
            + (" (low residual df: read as a caution.)" if rel != "ok" else "")))

    cvar = _doe_constant_variance(resid, fitted, df_resid)
    disp = _doe_dispersion_effect(resid, matrix, factor_names, df_resid)

    indep = _independence_check(resid)
    indep = replace(indep, reliability=rel if rel != "ok" else indep.reliability)
    if not indep.passed:
        indep = replace(indep, recommendation=(
            "Residuals are autocorrelated in run order; check the randomization and for "
            "a time/drift term."
            + (" (low residual df: read as a caution.)" if rel != "ok" else "")))

    return [norm, cvar, disp, indep]
