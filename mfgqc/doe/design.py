"""The immutable :class:`Design` and its generators (pre-data).

Generation is pre-data: it returns a coded design matrix, its run order, and -
for fractional designs - its full confounding structure, before a single run is
spent. Setters (:meth:`Design.center_points`, :meth:`Design.randomize`,
:meth:`Design.replicate`) each return a new ``Design`` with a provenance Step
appended. Analysis is separate (a method on ``QCData``); see
:mod:`mfgqc.doe.analysis`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..data import Step
from . import alias as _alias
from . import generate as _gen

# Curated minimum-aberration generators for the common regular 2^(k-p) designs
# (Box, Hunter and Hunter; Montgomery, Table of fractional factorial designs).
# Keyed by (k, p); values are the generator equations for the added factors.
_MIN_ABERRATION = {
    (3, 1): ["C=AB"],
    (4, 1): ["D=ABC"],
    (5, 1): ["E=ABCD"],
    (5, 2): ["D=AB", "E=AC"],
    (6, 1): ["F=ABCDE"],
    (6, 2): ["E=ABC", "F=BCD"],
    (7, 1): ["G=ABCDEF"],
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Design:
    """An immutable two-level experimental design (coded -1/+1)."""

    kind: str                                  # "full" | "fractional"
    factors: tuple[str, ...]
    matrix: np.ndarray                         # (n_base, k) coded, standard order
    levels: dict = field(default_factory=dict)        # name -> (low, high) actual, or {} coded
    run_order: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    replicates: int = 1
    n_center: int = 0
    generators: tuple[str, ...] = ()
    defining_relation: tuple[str, ...] = ()
    resolution: int | None = None
    aliases: tuple[str, ...] = ()
    history: tuple[Step, ...] = field(default_factory=tuple)

    # ---- derived ---------------------------------------------------------
    @property
    def k(self) -> int:
        return len(self.factors)

    @property
    def n_runs(self) -> int:
        """Total runs including replication and center points."""
        return self.matrix.shape[0] * self.replicates + self.n_center

    def _append(self, op: str, params: dict, **changes) -> "Design":
        step = Step(operation=op, params=params, n_affected=None, timestamp=_now())
        return replace(self, history=self.history + (step,), **changes)

    # ---- setters (return new Design) -------------------------------------
    def center_points(self, n: int) -> "Design":
        """Add ``n`` center points (all factors at coded 0). Enables a curvature
        check and a pure-error estimate."""
        if n < 0:
            raise ValueError("center-point count must be non-negative.")
        return self._append("center_points", {"n": n}, n_center=n)

    def randomize(self, seed: int) -> "Design":
        """Randomize the run order with a fixed seed (reproducible)."""
        order = _gen.randomized_order(self.matrix.shape[0] * self.replicates + self.n_center, seed)
        return self._append("randomize", {"seed": seed}, run_order=order)

    def replicate(self, r: int) -> "Design":
        """Replicate the whole design ``r`` times (gives pure error for t/F)."""
        if r < 1:
            raise ValueError("replicate count must be >= 1.")
        return self._append("replicate", {"r": r}, replicates=r)

    # ---- output ----------------------------------------------------------
    def _full_coded(self) -> np.ndarray:
        base = np.repeat(self.matrix, 1, axis=0)
        stacked = np.vstack([self.matrix] * self.replicates) if self.replicates > 1 else self.matrix
        if self.n_center:
            center = np.zeros((self.n_center, self.k))
            stacked = np.vstack([stacked, center])
        return stacked

    def run_sheet(self) -> pd.DataFrame:
        """The run sheet: one row per run, coded levels (and actual, if levels
        were given), in run order. Standard order is retained as a column."""
        coded = self._full_coded()
        n = coded.shape[0]
        order = self.run_order if self.run_order.size == n else np.arange(n)
        data = {"std_order": np.arange(1, n + 1)}
        for j, f in enumerate(self.factors):
            data[f] = coded[:, j]
        if self.levels:
            for j, f in enumerate(self.factors):
                if f in self.levels:
                    lo, hi = self.levels[f]
                    data[f"{f}_actual"] = _gen.decode(coded[:, j], lo, hi)
        df = pd.DataFrame(data)
        df = df.iloc[order].reset_index(drop=True)
        df.insert(0, "run", np.arange(1, n + 1))
        return df

    def summary(self) -> dict:
        out = {
            "kind": self.kind,
            "factors": list(self.factors),
            "k": self.k,
            "base_runs": int(self.matrix.shape[0]),
            "replicates": self.replicates,
            "center_points": self.n_center,
            "n_runs": self.n_runs,
        }
        if self.kind == "fractional":
            out["generators"] = list(self.generators)
            out["defining_relation"] = list(self.defining_relation)
            out["resolution"] = self.resolution
        return out

    def report(self) -> str:
        title = f"Design ({self.kind} 2-level): {', '.join(self.factors)}"
        lines = [title, "=" * len(title),
                 f"factors = {self.k}   base runs = {self.matrix.shape[0]}   "
                 f"replicates = {self.replicates}   center points = {self.n_center}   "
                 f"total runs = {self.n_runs}"]
        if self.kind == "fractional":
            res = {3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII"}.get(self.resolution, str(self.resolution))
            lines += ["",
                      f"generators: {', '.join(self.generators)}",
                      f"defining relation: I = {' = '.join(w for w in self.defining_relation if w != 'I')}",
                      f"resolution: {res}",
                      "",
                      "alias structure:"]
            lines += [f"  {a}" for a in self.aliases]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.report()

    def view(self, kind: str = "layout", **kwargs):
        from . import views
        return views.design_view(self, kind=kind, **kwargs)


# --------------------------------------------------------------------------- #
# Generators (module-level, pre-data)
# --------------------------------------------------------------------------- #
def _split_factors(factors):
    """Accept a list of names or a dict {name: (low, high)}; return (names, levels)."""
    if isinstance(factors, dict):
        names = list(factors)
        levels = {n: tuple(factors[n]) for n in names}
        return names, levels
    names = list(factors)
    return names, {}


def full_factorial(factors, replicates: int = 1, seed: int | None = None) -> Design:
    """A full 2^k factorial design coded to -1/+1 in standard order.

    Parameters
    ----------
    factors : list of str or dict
        Factor names, or ``{name: (low, high)}`` to also carry actual levels.
    replicates : int
        Replicate the whole design this many times (default 1).
    seed : int or None
        Randomize the run order with this seed; ``None`` keeps standard order.
    """
    names, levels = _split_factors(factors)
    k = len(names)
    if k < 1:
        raise ValueError("full_factorial needs at least one factor.")
    matrix = _gen.coded_full_matrix(k)
    n_total = matrix.shape[0] * replicates
    order = _gen.randomized_order(n_total, seed)
    step = Step(operation="full_factorial",
                params={"factors": names, "replicates": replicates, "seed": seed},
                n_affected=n_total, timestamp=_now())
    return Design(kind="full", factors=tuple(names), matrix=matrix, levels=levels,
                  run_order=order, replicates=replicates, history=(step,))


def _parse_fraction(fraction, k: int) -> int:
    """Return the number of generators p from a fraction spec like ``"1/2"`` or
    ``0.5`` or an int p."""
    if isinstance(fraction, str) and "/" in fraction:
        num, den = fraction.split("/")
        ratio = float(num) / float(den)
    elif isinstance(fraction, (int, float)) and 0 < fraction < 1:
        ratio = float(fraction)
    elif isinstance(fraction, int):
        return fraction                       # already the generator count p
    else:
        raise ValueError(f"could not parse fraction={fraction!r}; use '1/2', 0.5, or p.")
    p = round(-np.log2(ratio))
    if p < 1:
        raise ValueError(f"fraction {fraction!r} implies p={p}; use full_factorial for a full design.")
    return int(p)


def fractional_factorial(factors, generators=None, fraction=None, replicates: int = 1,
                         seed: int | None = None) -> Design:
    """A regular 2^(k-p) fractional factorial, with full alias structure.

    Supply EITHER ``generators`` OR ``fraction`` (not necessarily both). When
    ``generators`` are given the fraction is inferred (``p`` = number of
    generators) and ``fraction`` need not be passed; a ``fraction`` that
    contradicts the generators raises. When only ``fraction`` is given a
    minimum-aberration generator set is chosen and surfaced (never silently).

    Parameters
    ----------
    factors : list of str or dict
        The k factor names (or ``{name: (low, high)}``).
    generators : list of str or None
        Generator equations for the added factors, e.g. ``["E=ABCD"]``.
    fraction : str, float, int, or None
        The fraction (``"1/2"``, ``0.5``) or the generator count ``p``. Optional
        when ``generators`` is supplied.
    """
    names, levels = _split_factors(factors)
    k = len(names)

    if generators is None and fraction is None:
        raise ValueError("fractional_factorial needs either generators= or fraction=.")
    if generators is not None:
        p = len(generators)
        if fraction is not None and _parse_fraction(fraction, k) != p:
            raise ValueError(
                f"fraction={fraction!r} implies p={_parse_fraction(fraction, k)} but "
                f"{len(generators)} generator(s) were given (p={p}); they contradict.")
    else:
        p = _parse_fraction(fraction, k)
        chosen = _MIN_ABERRATION.get((k, p))
        if chosen is None:
            raise NotImplementedError(
                f"no curated minimum-aberration generator set for 2^({k}-{p}); "
                "pass generators= explicitly (e.g. ['E=ABCD']).")
        generators = chosen

    base_k = k - p
    if base_k < 1:
        raise ValueError(f"too few base factors: k={k}, p={p}.")

    # base full factorial in the first base_k factors
    base = _gen.coded_full_matrix(base_k)
    base_names = names[:base_k]
    added_names = names[base_k:]
    cols = {f: base[:, i] for i, f in enumerate(base_names)}

    gen_words = []
    for eq in generators:
        lhs, rhs = eq.replace(" ", "").split("=")
        rhs_factors = list(rhs)
        cols[lhs] = _gen.product_column(np.column_stack([cols[f] for f in rhs_factors]),
                                        range(len(rhs_factors)))
        # defining word: lhs * rhs = I  ->  word = letters of lhs and rhs together
        gen_words.append(_alias.parse_word(lhs + rhs))

    matrix = np.column_stack([cols[f] for f in names])

    group = _alias.defining_group(gen_words)
    res = _alias.resolution(group)
    alias_lines = _alias.alias_list(names, group)
    defining = tuple(_alias.word_str(w, names) for w in group)

    n_total = matrix.shape[0] * replicates
    order = _gen.randomized_order(n_total, seed)
    step = Step(operation="fractional_factorial",
                params={"factors": names, "fraction": fraction, "p": p,
                        "generators": list(generators), "resolution": res},
                n_affected=n_total, timestamp=_now())
    return Design(kind="fractional", factors=tuple(names), matrix=matrix, levels=levels,
                  run_order=order, replicates=replicates,
                  generators=tuple(generators), defining_relation=defining,
                  resolution=res, aliases=tuple(alias_lines), history=(step,))
