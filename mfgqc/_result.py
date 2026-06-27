"""Shared base for analysis result objects.

Every analysis returns an immutable result that (1) carries its numbers,
(2) carries the assumption-check outcomes, (3) carries the propagated provenance
history, (4) renders a full text report, and (5) exposes a ``view`` hook that
works standalone or inside a provided Axes (for future composition).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .assumptions import AssumptionCheck
    from .data import Step

_TEST_ABBREV = {"Anderson-Darling": "AD"}


def _jsonable(x: Any) -> Any:
    """Recursively convert a value to a JSON-serializable form.

    numpy scalars -> python scalars; numpy arrays / tuples / sets -> lists;
    non-finite floats (NaN/inf) -> ``None``; datetimes -> ISO strings;
    dataclasses -> dicts (skipping private ``_``-prefixed fields). Used by
    :meth:`QCResult.summary` and :meth:`QCResult.to_dict` so every result
    serializes with ``json.dumps`` without special handling by the caller.
    """
    import numpy as np

    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (_dt.datetime, _dt.date)):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in x]
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _jsonable(getattr(x, f.name))
                for f in dataclasses.fields(x) if not f.name.startswith("_")}
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return str(x)


def _is_flat_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


# --------------------------------------------------------------------------- #
# Provenance: lineage reconstruction + a tamper-evident chained digest.
# --------------------------------------------------------------------------- #
def _canonical_step(step: "Step") -> dict[str, Any]:
    """The integrity-bearing content of one provenance step (timestamp excluded so
    the digest is reproducible run to run; it is the *computation* that is pinned)."""
    return {
        "operation": step.operation,
        "params": _jsonable(step.params),
        "n_affected": step.n_affected,
    }


def history_lineage(history: "tuple[Step, ...]") -> list[dict[str, Any]]:
    """The full provenance chain as a list of JSON-serializable step dicts, each
    carrying the running digest up to and including that step."""
    out: list[dict[str, Any]] = []
    running = ""
    for step in history:
        canon = _canonical_step(step)
        running = _chain(running, canon)
        out.append({**canon, "digest": running})
    return out


def _chain(prev_digest: str, canon: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(prev_digest.encode())
    h.update(json.dumps(canon, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def history_digest(history: "tuple[Step, ...]") -> str:
    """A single SHA-256 digest over the whole provenance chain. Editing the
    operation, params, or n_affected of ANY recorded step changes this digest, so a
    digest captured at analysis time detects later edits to the recorded history.

    Guarantee and its limit: the history is append-only by construction (an
    immutable tuple of frozen steps), and the digest makes the recorded content
    verifiable. It is a content hash, not a cryptographic signature: an actor who
    can run code in this process can edit a step's params dict in place AND
    recompute the digest. The digest defends against accidental corruption and
    post-hoc edits to a stored result, not against an adversary controlling the
    interpreter."""
    running = ""
    for step in history:
        running = _chain(running, _canonical_step(step))
    return running


def _abbrev(test: str) -> str:
    return _TEST_ABBREV.get(test, test.split()[0] if test else "stat")


def _assumption_line(a: "AssumptionCheck") -> str:
    """Render one assumption check: binary PASS/FAIL bracket + magnitude/reliability context."""
    tag = "PASS" if a.passed else "FAIL"
    p = f", p={a.p_value:.3g}" if (a.p_value is not None and math.isfinite(a.p_value)) else ""
    lbl = a.magnitude_label
    stat_ok = math.isfinite(a.statistic)
    if lbl == "est. Cpk impact" and a.magnitude is not None:
        head = f"AD={a.statistic:.3g}{p}; est. Cpk impact {a.magnitude * 100:.1f}%"
    elif lbl == "skew" and a.magnitude is not None:
        head = f"AD={a.statistic:.3g}{p}; skew {a.magnitude:.3g}"
    elif lbl == "variance ratio" and a.magnitude is not None:
        head = f"variance ratio {a.magnitude:.3g}{p}"
    elif lbl == "lag-1 autocorr":
        head = (f"r={a.statistic:.3g}{p}" if stat_ok else "r=n/a")
    elif lbl == "dispersion ratio" and a.magnitude is not None:
        head = f"dispersion ratio {a.magnitude:.3g}{p}"
    elif lbl in ("ndc", "subgroup count", "min expected count") and a.magnitude is not None:
        head = f"{lbl} {a.magnitude:.4g}"
    elif stat_ok:
        head = f"{_abbrev(a.test)}={a.statistic:.3g}{p}"
    else:
        head = a.test
    caveat = ""
    if a.reliability != "ok":
        caveat = f" [{'low power' if a.reliability == 'low_power' else 'oversensitive'}]"
    return f"  [{tag}] {a.name} ({a.test}): {head}; n={a.n}{caveat}"


class QCResult:
    """Mixin providing the report and view behaviour shared by all results.

    Subclasses are expected to be frozen dataclasses that define at least
    ``assumptions`` and ``history`` fields, plus implement ``_title``,
    ``_summary_lines``, ``_render_standalone`` and ``_render_axes``.
    """

    assumptions: list["AssumptionCheck"]
    history: tuple["Step", ...]

    # ---- reporting -------------------------------------------------------
    def _title(self) -> str:  # pragma: no cover - overridden
        return type(self).__name__

    def _summary_lines(self) -> list[str]:  # pragma: no cover - overridden
        return []

    def report(self) -> str:
        """Return the full text report: numbers, assumptions, recommendations."""
        title = self._title()
        lines: list[str] = [title, "=" * len(title)]
        lines.extend(self._summary_lines())

        lines.append("")
        lines.append("Assumption checks:")
        if not self.assumptions:
            lines.append("  (none)")
        for a in self.assumptions:
            lines.append(_assumption_line(a))

        recs = [a.recommendation for a in self.assumptions if a.recommendation]
        own = getattr(self, "recommendation", None)  # result-level (e.g. test routing)
        if own:
            recs.append(own)
        if recs:
            lines.append("")
            lines.append("Recommendations:")
            for r in recs:
                lines.append(f"  - {r}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.report()

    def _repr_html_(self) -> str:
        return "<pre>" + self.report() + "</pre>"

    # ---- serialization ---------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Return a FLAT, JSON-serializable dict of the result's scalar fields.

        This base implementation auto-flattens the public scalar fields of the
        result dataclass (skipping arrays, nested structures, assumptions and
        history). Many result types override it with a curated key set; both the
        override and this fallback are guaranteed ``json.dumps``-able. For the
        full structured form (arrays, CI tuples, assumption flags, provenance)
        use :meth:`to_dict`.
        """
        out: dict[str, Any] = {}
        if dataclasses.is_dataclass(self):
            for f in dataclasses.fields(self):
                if f.name.startswith("_") or f.name in ("assumptions", "history"):
                    continue
                v = _jsonable(getattr(self, f.name))
                if _is_flat_scalar(v):
                    out[f.name] = v
        return out

    def to_dict(self) -> dict[str, Any]:
        """Return the full structured, JSON-serializable form of the result.

        Includes every public field (arrays and CI tuples as lists), the flat
        :meth:`summary`, the assumption checks, the provenance ``history``, and the
        ``provenance_digest`` that pins that history. This is the canonical payload
        for a report builder or a wire transfer; callers must not parse
        :meth:`report` text.
        """
        fields: dict[str, Any] = {}
        if dataclasses.is_dataclass(self):
            for f in dataclasses.fields(self):
                if f.name.startswith("_") or f.name in ("assumptions", "history"):
                    continue
                fields[f.name] = _jsonable(getattr(self, f.name))
        return {
            "result_type": type(self).__name__,
            "title": self._title(),
            "summary": self.summary(),
            "fields": fields,
            "assumptions": [_jsonable(a) for a in (getattr(self, "assumptions", None) or [])],
            "history": self.lineage(),
            "provenance_digest": self.provenance_digest(),
        }

    # ---- provenance ------------------------------------------------------
    def lineage(self) -> list[dict[str, Any]]:
        """The provenance chain that produced this result, as a list of step dicts
        (each with its running ``digest``), reconstructable end to end."""
        return history_lineage(getattr(self, "history", ()) or ())

    def provenance_digest(self) -> str:
        """SHA-256 digest pinning the full provenance chain (see
        :func:`history_digest` for the guarantee and its limit)."""
        return history_digest(getattr(self, "history", ()) or ())

    def verify_provenance(self, expected_digest: str) -> bool:
        """True iff the recorded history still matches a digest captured earlier."""
        return self.provenance_digest() == expected_digest

    # ---- visualization ---------------------------------------------------
    def _render_standalone(self, fig: "Figure", kind: str | None, **kwargs: Any) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _render_axes(self, ax: "Axes", kind: str | None, **kwargs: Any) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def view(self, ax: "Axes | None" = None, kind: str | None = None, *,
             save: str | None = None, dpi: int = 150, **kwargs: Any):
        """Render the analysis.

        Parameters
        ----------
        ax : matplotlib Axes or None, optional
            If ``None``, a new Figure is created and returned (the canonical,
            possibly multi-panel chart). If an Axes is given, the analysis draws
            its primary panel into it and returns that Axes (enables composition).
        kind : str or None, optional
            ``None`` selects the canonical chart for the analysis; a string
            selects an explicit alternative view.
        save : str or None, optional
            If given (only when ``ax is None``), write the figure to this path
            before returning it. The format is taken from the extension, so
            ``"chart.png"`` and ``"chart.svg"`` both work. The Figure is still
            returned so the caller can also embed it as bytes.
        dpi : int, optional
            Resolution for raster output when ``save`` is a raster format.

        Returns
        -------
        matplotlib.figure.Figure or matplotlib.axes.Axes

        Notes
        -----
        Rendering is headless-safe: it never calls ``plt.show`` and detaches the
        Figure from pyplot's registry, so it works under the Agg backend on a
        server with no display.
        """
        import matplotlib.pyplot as plt

        from . import palette as _pal
        with _pal.rc_context():  # apply the brand chart theme to mfgQC's own figures
            if ax is None:
                fig = plt.figure(figsize=kwargs.pop("figsize", (9, 6)))
                self._render_standalone(fig, kind, **kwargs)
                fig.tight_layout()
                if save is not None:
                    fig.savefig(save, dpi=dpi, bbox_inches="tight")
                # Detach from pyplot's registry so an inline/notebook backend does
                # not auto-display it in addition to the caller's display(fig);
                # the returned Figure still renders once. (One figure, one render.)
                plt.close(fig)
                return fig
            if save is not None:
                raise ValueError("save= is only supported when ax is None "
                                 "(the standalone figure path).")
            self._render_axes(ax, kind, **kwargs)
            return ax
