"""Ingestion layer: ``overview`` (polymorphic diagnostic) and ``clean`` (structural
df->df pipeline).

Design boundary (the QC-safety rule): ``clean`` does STRUCTURE and INTEGRITY only -
it never imputes, caps/removes outliers, or scales/normalizes. Outliers and
missingness are SURFACED (by ``overview``), never silently altered. ``overview``
presents facts and proposes role candidates; it never decides or recomputes a
result's checks.
"""

from __future__ import annotations

from . import palette as _pal

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from ._result import QCResult, _assumption_line
from .data import QCData, Step

_SENTINELS = (-99999, -9999, -999, 999, 9999, 99999)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# =========================================================================== #
# overview()
# =========================================================================== #
@dataclass(frozen=True)
class Overview:
    """A diagnostic snapshot whose text and charts are equal partners.

    In a notebook, evaluating ``overview(x)`` as the last cell expression shows
    BOTH the text table AND the charts inline (one expression) via ``_repr_html_``.
    ``print(overview(x))`` / ``repr(...)`` give text only; ``.view()`` gives the
    figure only (scripts, or charts without the text)."""

    kind: str
    _text: str = field(repr=False)
    _view_fn: Callable[[], Any] | None = field(repr=False, default=None)
    _auto_visual: bool = field(repr=False, default=False)

    def __repr__(self) -> str:
        # Pure-text path (print / REPL / logs). Charts aren't shown here; point to
        # the two ways to get them.
        if self._view_fn is None:
            return self._text
        hint = "[ .view() for charts, or overview(..., visual=True) to show them inline ]"
        return f"{self._text}\n\n{hint}"

    def _ipython_display_(self):
        """Notebook display. TEXT-ONLY by default (wide frames shouldn't dump a wall
        of charts); with ``visual=True`` it also emits the chart(s).

        When charts are shown they go out as a SEPARATE ``image/png`` output (not a
        single mimebundle) so both text and figure render inline AND survive
        nbconvert -> pandoc -> PDF (pandoc embeds the image/png output)."""
        print(self._text)
        if self._view_fn is None:
            return
        if not self._auto_visual:
            print("\n[ overview(..., visual=True) to show charts inline; or .view() for the figure ]")
            return
        import matplotlib.pyplot as plt
        from IPython.display import display
        with _pal.rc_context():
            fig = self._view_fn()
        if fig is not None:
            display(fig)        # emits image/png via the Figure's own repr
            plt.close(fig)      # close so the inline backend doesn't re-flush it

    def _repr_html_(self) -> str:
        # Fallback for pure-HTML renderers that don't call _ipython_display_.
        import html as _html
        body = "<pre style='line-height:1.25'>" + _html.escape(self._text) + "</pre>"
        if self._auto_visual:
            png = self._render_png()
            if png is not None:
                body += f'<img src="data:image/png;base64,{png}"/>'
        return body

    def _render_png(self) -> str | None:
        """Render the figure to a base64 PNG and CLOSE it (so the inline backend
        does not also auto-display it -> the double-render bug)."""
        if self._view_fn is None:
            return None
        import base64
        import io

        import matplotlib.pyplot as plt
        with _pal.rc_context():
            fig = self._view_fn()
        if fig is None:
            return None
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def view(self):
        """Render the diagnostic chart(s) explicitly. Returns a matplotlib Figure (or None)."""
        if self._view_fn is None:
            return None
        with _pal.rc_context():
            return self._view_fn()


def overview(obj: Any, *, visual: bool = False) -> Overview:
    """Polymorphic diagnostic. Accepts a DataFrame (raw hygiene), a QCData
    (role/spec-aware), or an analysis result (presents the checks it already ran).

    Text-only by default. Pass ``visual=True`` to also render the chart(s) inline in a
    notebook (text stays the default because a wide raw frame can have many measures).
    ``.view()`` returns/draws the figure explicitly regardless of ``visual``."""
    if isinstance(obj, pd.DataFrame):
        return _overview_df(obj, visual)
    if isinstance(obj, QCData):
        return _overview_qcdata(obj, visual)
    if isinstance(obj, QCResult):
        return _overview_result(obj, visual)
    raise TypeError(f"overview() expects a DataFrame, QCData, or analysis result; got {type(obj).__name__}.")


def _numeric_candidates(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _measure_candidates(df: pd.DataFrame) -> list[str]:
    """Numeric, high-cardinality columns proposed as measures (not counts/codes)."""
    n = len(df)
    return [c for c in _numeric_candidates(df)
            if int(df[c].nunique(dropna=True)) > max(20, 0.5 * n)]


def _role_category(s: pd.Series, n: int) -> str | None:
    """Single proposed role category for a column (proposed, never assigned)."""
    distinct = int(s.nunique(dropna=True))
    if pd.api.types.is_datetime64_any_dtype(s):
        return "time"
    if pd.api.types.is_numeric_dtype(s):
        if distinct > max(20, 0.5 * n):
            return "measures"
        if 2 <= distinct <= 30:
            return "subgroup?"
        return None
    if 2 <= distinct <= 30:
        return "factors"
    return None


def _aligned_table(headers: list[str], rows: list[list[str]], left: set[str]) -> list[str]:
    """Fixed-width table: pad each column to its widest cell; left/right justify."""
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
              for i in range(len(headers))]

    def _fmt(cells: list[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            out.append(cell.ljust(widths[i]) if headers[i] in left else cell.rjust(widths[i]))
        return "  ".join(out).rstrip()

    return [_fmt(headers)] + [_fmt(r) for r in rows]


def _overview_df(df: pd.DataFrame, visual: bool = False) -> Overview:
    n = len(df)
    lines = [f"Overview(DataFrame): {df.shape[0]} rows x {df.shape[1]} cols", ""]

    headers = ["column", "dtype", "missing", "distinct", "min", "max", "mean", "median", "std"]
    left = {"column", "dtype", "missing"}
    rows: list[list[str]] = []
    role_groups: dict[str, list[str]] = {}
    for col in df.columns:
        s = df[col]
        miss = int(s.isna().sum())
        miss_cell = "0 (0%)" if miss == 0 else f"{miss} ({miss / n * 100:.1f}%)"
        distinct = int(s.nunique(dropna=True))
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            num = s.dropna()
            mn, mx = f"{num.min():.2f}", f"{num.max():.2f}"
            mean, median, std = f"{num.mean():.2f}", f"{num.median():.2f}", f"{num.std(ddof=1):.2f}"
        else:
            mn = mx = mean = median = std = "-"
        rows.append([str(col), str(s.dtype), miss_cell, str(distinct), mn, mx, mean, median, std])
        cat = _role_category(s, n)
        if cat:
            role_groups.setdefault(cat, []).append(str(col))

    lines.extend(_aligned_table(headers, rows, left))

    # role candidates - grouped, one line per category
    lines.append("")
    lines.append("Role candidates (proposed, not assigned):")
    order = ["measures", "time", "factors", "subgroup?"]
    present = [c for c in order if role_groups.get(c)]
    if present:
        w = max(len(c) + 1 for c in present)  # +1 for the trailing colon
        for cat in present:
            lines.append(f"  {(cat + ':'):<{w}}  {', '.join(role_groups[cat])}")
    else:
        lines.append("  (none obvious)")

    # FLAGS
    flags: list[str] = []
    cols = list(df.columns)
    if len(cols) != len(set(map(str, cols))):
        flags.append("duplicate column names")
    ws = [c for c in cols if str(c) != str(c).strip() or " " in str(c) or str(c) != str(c).lower()]
    if ws:
        flags.append(f"non-tidy column names: {ws[:5]}")
    dup_rows = int(df.duplicated().sum())
    if dup_rows:
        flags.append(f"{dup_rows} duplicate rows")
    for col in cols:
        s = df[col]
        distinct = int(s.nunique(dropna=True))
        if distinct <= 1:
            flags.append(f"constant column: {col}")
        if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
            coerced = pd.to_numeric(s, errors="coerce")
            parse_rate = coerced.notna().mean() if len(s) else 0
            if parse_rate > 0.8 and s.notna().any():
                flags.append(f"numeric-as-text: {col} ({parse_rate*100:.0f}% parse numeric)")
        num = s if pd.api.types.is_numeric_dtype(s) else pd.to_numeric(s, errors="coerce")
        if num.notna().any():
            present = [v for v in _SENTINELS if (num == v).any()]
            if present:
                flags.append(f"possible sentinel value(s) in {col}: {present}")
        miss_frac = s.isna().mean() if len(s) else 0
        if miss_frac > 0.2:
            flags.append(f"high missingness: {col} ({miss_frac*100:.0f}%)")

    lines.append("")
    lines.append("Flags:")
    if flags:
        for f in flags:
            lines.append(f"  ! {f}")
    else:
        lines.append("  (none)")

    # de-dup measure candidates ONCE; chart only measures (never counts/codes like size)
    measures = list(dict.fromkeys(_measure_candidates(df)))

    def _view():
        import matplotlib.pyplot as plt
        cols_to_plot = measures
        if not cols_to_plot:
            fig, ax = plt.subplots(figsize=(6, 1.5))
            ax.text(0.5, 0.5, "no measure candidates to chart", ha="center", va="center")
            ax.axis("off")
            return fig
        assert len(cols_to_plot) == len(set(cols_to_plot)), "duplicate measure in chart grid"
        fig, axes = plt.subplots(len(cols_to_plot), 2, figsize=(9, 2.4 * len(cols_to_plot)))
        axes = np.atleast_2d(axes)
        for i, c in enumerate(cols_to_plot):
            vals = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy()
            axes[i, 0].hist(vals, bins="auto", color=_pal.active().data, edgecolor=_pal.active().bg)
            axes[i, 0].set_title(f"{c} - histogram", fontsize=9)
            axes[i, 1].boxplot(vals, vert=False)
            axes[i, 1].set_title(f"{c} - boxplot", fontsize=9)
        fig.tight_layout()
        return fig

    return Overview("dataframe", "\n".join(lines), _view, visual)


def _overview_qcdata(qc: QCData, visual: bool = False) -> Overview:
    m = qc.meta
    vals = qc.values()
    vals = vals[~np.isnan(vals)]
    lines = [f"Overview(QCData): measure={m.measure!r}" + (f" [{m.units}]" if m.units else ""), ""]
    lines.append(f"n={vals.size}  mean={np.mean(vals):.5g}  std={np.std(vals, ddof=1):.4g}  "
                 f"skew={float(stats_skew(vals)):.3g}  kurtosis={float(stats_kurtosis(vals)):.3g}")
    from .assumptions import _anderson_darling_statistic
    a2 = _anderson_darling_statistic(vals)
    lines.append(f"normality (Anderson-Darling) statistic = {a2:.3g}  (information only; not a decision)")
    lines.append("")
    lines.append(f"roles: {dict(m.roles) or '{}'}")

    # subgroup-size consistency
    if "subgroup" in m.roles or "time" in m.roles or m.subgroup_size:
        try:
            sg = qc.subgroups()
            consistent = "consistent" if sg.equal_n else f"VARYING ({sorted(set(sg.sizes))})"
            lines.append(f"subgroups: {len(sg.groups)} groups, size {consistent}")
        except ValueError as e:
            lines.append(f"subgroups: not available ({e})")

    # balance (crossed/gage)
    if {"part", "operator", "replicate"} <= set(m.roles):
        try:
            cr = qc.crossed()
            lines.append(f"crossed design: {len(cr.parts)} parts x {len(cr.operators)} operators, "
                         f"{'balanced' if cr.balanced else 'UNBALANCED'}")
        except ValueError as e:
            lines.append(f"crossed: not available ({e})")

    # spec orientation - describe where the MEAN sits in the band (not the limits)
    if m.limits.has_any():
        mean = float(np.mean(vals))
        lo, up, tg = m.limits.lower, m.limits.upper, m.limits.target
        head = ", ".join(f"{name}={val}" for name, val in
                         (("LSL", lo), ("USL", up), ("target", tg)) if val is not None)
        lines.append("spec: " + head)
        if lo is not None and up is not None:
            pos = (mean - lo) / (up - lo) * 100
            lines.append(f"  mean {mean:.2f} sits {pos:.0f}% up the band [LSL..USL]")
            lines.append(f"  margin to LSL = {mean - lo:.2f}, margin to USL = {up - mean:.2f}")
        elif lo is not None:
            lines.append(f"  mean {mean:.2f}, margin to LSL = {mean - lo:.2f}")
        elif up is not None:
            lines.append(f"  mean {mean:.2f}, margin to USL = {up - mean:.2f}")
        if tg is not None:
            lines.append(f"  off target by {mean - tg:+.2f}")
    else:
        lines.append("spec: none attached (use .spec(...))")

    def _view():
        import matplotlib.pyplot as plt
        fig, (a1, a2_) = plt.subplots(1, 2, figsize=(9, 3.2))
        a1.hist(vals, bins="auto", color=_pal.active().data, edgecolor=_pal.active().bg)
        for v, lab, col in ((m.limits.lower, "LSL", _pal.active().ooc), (m.limits.upper, "USL", _pal.active().ooc),
                            (m.limits.target, "T", _pal.active().target)):
            if v is not None:
                a1.axvline(v, color=col, ls="--", lw=1.3)
        a1.set_title(f"{m.measure} distribution", fontsize=9)
        a2_.boxplot(vals, vert=False)
        a2_.set_title(f"{m.measure} boxplot", fontsize=9)
        fig.tight_layout()
        return fig

    return Overview("qcdata", "\n".join(lines), _view, visual)


def _overview_result(result: QCResult, visual: bool = False) -> Overview:
    lines = [f"Overview({type(result).__name__}): presenting checks already run "
             "(no recomputation)", ""]
    lines.extend(result._summary_lines()[:6])
    lines.append("")
    lines.append("Assumption checks:")
    if not result.assumptions:
        lines.append("  (none)")
    for a in result.assumptions:
        lines.append(_assumption_line(a))

    def _view():
        return result.view()

    return Overview("result", "\n".join(lines), _view, visual)


# small skew/kurtosis without importing scipy at module top twice
def stats_skew(x):
    from scipy import stats
    return stats.skew(x)


def stats_kurtosis(x):
    from scipy import stats
    return stats.kurtosis(x)


# =========================================================================== #
# clean()  - structural/integrity only (NEVER impute/cap/scale)
# =========================================================================== #
Task = Callable[[pd.DataFrame], "tuple[pd.DataFrame, Step | list[Step]]"]


def _step(op: str, params: dict, n: int | None) -> Step:
    return Step(operation=f"clean.{op}", params=params, n_affected=n, timestamp=_now())


# --------------------------------------------------------------------------- #
# Conservative value classification (mechanical recovery; refuse-and-flag the
# ambiguous). Never fabricates a value: an interpretation it can't make safely
# becomes NA/NaT plus a flag the user reviews.
# --------------------------------------------------------------------------- #
_CENSORED = re.compile(r"^[<>]=?\s*[-+]?\d")          # "<1.20", ">5.0", ">=3"
_UNIT_SUFFIX = re.compile(r"^([-+]?\d*\.?\d+)\s*[a-zA-Z%°]+$")  # "1.48 mm", "1.55mm"


def _coerce_one(v: Any) -> tuple[float, tuple[str, str] | None]:
    """Return (numeric_or_nan, flag_or_None) for a single cell - mechanical only.

    Recovers: real numbers, plain numeric strings, and number+unit ("1.48 mm").
    Refuses+flags: censored ("<1.20") and decimal/thousands-ambiguous ("1,520").
    Anything else unparseable -> NA, no flag (e.g. a string sentinel like 'N/A').
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan, None
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return float(v), None                              # already numeric (recovered)
    s = str(v).strip()
    if s == "":
        return np.nan, None
    if _CENSORED.match(s):
        return np.nan, ("censored", s)                     # "<1.20" -> NA (don't pick a number)
    if "," in s:
        return np.nan, ("ambiguous numeric", s)            # "1,520": 1520 vs 1.520? refuse
    try:
        return float(s), None                              # plain numeric string
    except ValueError:
        pass
    m = _UNIT_SUFFIX.match(s)
    if m:
        try:
            return float(m.group(1)), None                 # mechanical unit strip
        except ValueError:
            pass
    return np.nan, None                                    # unparseable -> NA, unflagged


_ISO_DATE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_SEP_DATE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$")


def _parse_one_date(v: Any) -> tuple[Any, tuple[str, str] | None]:
    """Return (Timestamp_or_NaT, flag_or_None) - unambiguous parses only.

    ISO (YYYY-MM-DD) and dates where one field exceeds 12 (so the day is
    determined) parse; a value where both fields are <=12 is genuinely ambiguous
    (M/D vs D/M) -> NaT + flag. Never silently picks a format.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return pd.NaT, None
    s = str(v).strip()
    m = _ISO_DATE.match(s)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        try:
            return pd.Timestamp(year=y, month=mo, day=d), None
        except ValueError:
            return pd.NaT, None
    m = _SEP_DATE.match(s)
    if m:
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        if a <= 12 and b <= 12:
            return pd.NaT, ("ambiguous date", s)           # M/D vs D/M - refuse
        if a > 12 and b <= 12:
            day, mo = a, b
        elif b > 12 and a <= 12:
            mo, day = a, b
        else:
            return pd.NaT, None                            # both > 12: invalid
        try:
            return pd.Timestamp(year=y, month=mo, day=day), None
        except ValueError:
            return pd.NaT, None
    return pd.NaT, None                                     # unrecognized -> NaT (no guess)


def fix_names() -> Task:
    """Lower-case, strip, and underscore column names."""
    def _f(df):
        before = list(df.columns)
        after = [str(c).strip().lower().replace(" ", "_") for c in before]
        df = df.rename(columns=dict(zip(before, after)))
        n = sum(a != str(b) for a, b in zip(after, before))
        return df, _step("fix_names", {"renamed": n}, n)
    return _f


def coerce_numeric(cols: list[str]) -> Task:
    """Coerce columns to numeric, CONSERVATIVELY.

    Mechanical recovery only: real numbers, plain numeric strings, and
    number+unit (``"1.48 mm" -> 1.48``). Values that need INTERPRETATION are
    refused, set to NA, and FLAGGED - never guessed:
    ``"1,520"`` (decimal-comma vs thousands) -> NA 'ambiguous numeric';
    ``"<1.20"`` (censored) -> NA 'censored'. Never fills a resulting NA.
    """
    def _f(df):
        df = df.copy()
        flags: list[dict] = []
        for c in cols:
            if c not in df.columns:
                continue
            out_vals = []
            for v in df[c].tolist():
                val, flag = _coerce_one(v)
                out_vals.append(val)
                if flag is not None:
                    flags.append({"task": "coerce_numeric", "column": c,
                                  "reason": flag[0], "value": flag[1]})
            df[c] = np.asarray(out_vals, dtype=float)
        return df, _step("coerce_numeric", {"cols": list(cols), "flags": flags}, len(flags))
    return _f


def parse_dates(cols: list[str], fmt: str | None = None) -> Task:
    """Parse columns to datetime, CONSERVATIVELY.

    Parses ISO and unambiguous values directly; a value that is genuinely
    ambiguous (``01/02/2026`` could be M/D or D/M) becomes NaT and is FLAGGED -
    never silently assigned a format. If ``fmt`` is given it is applied to every
    value (the user has taken responsibility) and nothing is flagged.
    """
    def _f(df):
        df = df.copy()
        flags: list[dict] = []
        for c in cols:
            if c not in df.columns:
                continue
            if fmt is not None:
                df[c] = pd.to_datetime(df[c], format=fmt, errors="coerce")
                continue
            out_vals = []
            for v in df[c].tolist():
                ts, flag = _parse_one_date(v)
                out_vals.append(ts)
                if flag is not None:
                    flags.append({"task": "parse_dates", "column": c,
                                  "reason": flag[0], "value": flag[1]})
            df[c] = pd.to_datetime(pd.Series(out_vals, index=df.index))
        return df, _step("parse_dates", {"cols": list(cols), "fmt": fmt, "flags": flags}, len(flags))
    return _f


def normalize_case(cols: list[str]) -> Task:
    """OPT-IN: merge categorical case/whitespace variants (``'Day' -> 'day'``).

    This INTERPRETS case as meaningless, so it is never automatic - clean() only
    trims whitespace and FLAGS case variants; call this task to actually merge them.
    """
    def _f(df):
        df = df.copy()
        n = 0
        for c in cols:
            if c in df.columns:
                col = df[c].astype("string")
                new = col.str.strip().str.lower()
                n += int((new != col).fillna(False).sum())
                df[c] = new
        return df, _step("normalize_case", {"cols": list(cols)}, n)
    return _f


def recode_missing(cols: list[str], sentinels: list) -> Task:
    """Recode sentinel values to NA. RECODE ONLY - never fills the resulting NA."""
    def _f(df):
        df = df.copy()
        n = 0
        for c in cols:
            if c in df.columns:
                mask = df[c].isin(sentinels)
                n += int(mask.sum())
                df.loc[mask, c] = np.nan
        return df, _step("recode_missing", {"cols": list(cols), "sentinels": list(sentinels)}, n)
    return _f


def recode_empty() -> Task:
    """Recode empty strings to NA across object columns."""
    def _f(df):
        df = df.copy()
        obj = [c for c in df.columns
               if not pd.api.types.is_numeric_dtype(df[c])
               and not pd.api.types.is_datetime64_any_dtype(df[c])]
        n = 0
        for c in obj:
            mask = df[c].astype("string").str.strip().eq("")
            mask = mask.fillna(False)
            n += int(mask.sum())
            df.loc[mask, c] = np.nan
        return df, _step("recode_empty", {}, n)
    return _f


def drop_constant() -> Task:
    """Drop columns with a single unique value."""
    def _f(df):
        const = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        df = df.drop(columns=const)
        return df, _step("drop_constant", {"dropped": const}, len(const))
    return _f


def drop_duplicates() -> Task:
    """Drop duplicate rows."""
    def _f(df):
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        return df, _step("drop_duplicates", {}, before - len(df))
    return _f


def select(cols: list[str]) -> Task:
    """Keep only the named columns (those that exist)."""
    def _f(df):
        keep = [c for c in cols if c in df.columns]
        return df[keep], _step("select", {"cols": keep}, len(keep))
    return _f


def drop(cols: list[str]) -> Task:
    """Drop the named columns (those that exist)."""
    def _f(df):
        present = [c for c in cols if c in df.columns]
        return df.drop(columns=present), _step("drop", {"cols": present}, len(present))
    return _f


def rename(mapping: dict[str, str]) -> Task:
    """Rename columns by an explicit mapping."""
    def _f(df):
        return df.rename(columns=mapping), _step("rename", {"mapping": dict(mapping)}, len(mapping))
    return _f


def standard_tidy(date_cols: list[str] | None = None, numeric_cols: list[str] | None = None) -> Task:
    """Bundle: fix_names + coerce_numeric + parse_dates + drop_constant. Expands to
    atomic Steps in the cleaning report (auditable, not opaque)."""
    subtasks: list[Task] = [fix_names()]
    if numeric_cols:
        subtasks.append(coerce_numeric(numeric_cols))
    if date_cols:
        subtasks.append(parse_dates(date_cols))
    subtasks.append(drop_constant())

    def _f(df):
        steps: list[Step] = []
        for st in subtasks:
            df, s = st(df)
            steps.append(s)
        return df, steps
    return _f


def _trim_object_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Always-safe mechanical pass: strip surrounding whitespace on string cells."""
    df = df.copy()
    n = 0
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        new = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
        n += int(sum(1 for a, b in zip(df[c].tolist(), new.tolist())
                     if isinstance(a, str) and a != b))
        df[c] = new
    return df, n


def _case_variant_flags(df: pd.DataFrame) -> list[dict]:
    """Flag string columns whose values differ only by case/whitespace (NOT merged)."""
    flags: list[dict] = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        raw = {str(v) for v in df[c].dropna().tolist()}
        if len(raw) < 2:
            continue
        folded = {x.strip().lower() for x in raw}
        if len(folded) < len(raw):
            flags.append({"task": "values", "column": c, "reason": "case/space variants",
                          "value": f"{len(raw)} labels differ only by case/space -> possibly {len(folded)}"})
    return flags


def _format_clean_summary(flags: list[dict]) -> str:
    """Render the end-of-run summary: what clean() refused and left for the user."""
    groups: dict = {}
    for f in flags:
        key = (f["task"], f["column"], f["reason"])
        groups.setdefault(key, []).append(f.get("value"))
    n = len(groups)
    lines = [f"clean() flagged {n} thing{'s' if n != 1 else ''} for your review "
             "(left as missing, not interpreted):"]
    for (task, col, reason), vals in groups.items():
        count = len(vals)
        if reason == "case/space variants":
            lines.append(f"  [{task}] {col}: {vals[0]} (not merged; call normalize_case to merge)")
        else:
            disp = "NaT" if "date" in reason else "NA"
            unit = "value" if count == 1 else "values"
            lines.append(f"  [{task}] {col}: {reason} e.g. {vals[0]!r} -> {disp}  ({count} {unit})")
    return "\n".join(lines)


def clean(df: pd.DataFrame, tasks: list[Task] | None = None, *, verbose: bool = True) -> pd.DataFrame:
    """Apply a pipeline of CONSERVATIVE cleaning tasks; return a cleaned DataFrame.

    clean() makes only mechanically-safe recoveries (unit strip, whitespace trim)
    and REFUSES to interpret ambiguous values - it surfaces them, leaves them as
    NA/NaT, and emits an end-of-run summary of what it flagged for the user. It is
    fully deterministic and NEVER imputes, caps/removes outliers, scales, or
    fabricates a value for an ambiguous token.

    The cleaning report is attached to ``.attrs['mfgqc_clean_steps']`` (absorbed into
    a QCData's history by :func:`mfgqc.load`); flagged dispositions are attached to
    ``.attrs['mfgqc_clean_flags']`` and printed (unless ``verbose=False``).

    With no tasks, applies the safe default bundle: ``fix_names`` + empty-string->NA
    + ``drop_duplicates``.
    """
    if tasks is None:
        tasks = [fix_names(), recode_empty(), drop_duplicates()]
    out = df.copy()
    steps: list[Step] = []
    for task in tasks:
        out, produced = task(out)
        if isinstance(produced, list):
            steps.extend(produced)
        else:
            steps.append(produced)
    # always-safe mechanical pass: trim surrounding whitespace on string cells
    out, n_trim = _trim_object_columns(out)
    if n_trim:
        steps.append(_step("trim_whitespace", {}, n_trim))
    out = out.copy()
    # collect flagged dispositions: per-task refusals + case/space-variant surfacing
    flags: list[dict] = []
    for s in steps:
        if isinstance(s.params, dict):
            flags.extend(s.params.get("flags", []))
    flags.extend(_case_variant_flags(out))
    out.attrs["mfgqc_clean_steps"] = steps
    out.attrs["mfgqc_clean_flags"] = flags
    if flags and verbose:
        print(_format_clean_summary(flags))
    return out


def clean_report(df: pd.DataFrame) -> dict:
    """Return a structured, JSON-serializable summary of the last :func:`clean`.

    A frontend calls this on the frame returned by :func:`clean` to render the
    flagged-items UI ("5 values flagged for review") instead of parsing printed
    text. Returns ``{"n_flagged": int, "flags": [...], "steps": [...]}`` where
    each flag is ``{"task", "column", "reason", "value"}`` and each step is
    ``{"operation", "n_affected"}``. Empty/zero if ``df`` was not produced by
    :func:`clean`.
    """
    flags = list(df.attrs.get("mfgqc_clean_flags", []) or [])
    steps = list(df.attrs.get("mfgqc_clean_steps", []) or [])
    return {
        "n_flagged": len(flags),
        "flags": [dict(f) for f in flags],
        "steps": [{"operation": s.operation, "n_affected": s.n_affected} for s in steps],
    }
