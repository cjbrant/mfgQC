"""The data contract: :class:`QCData`, its typed metadata, and its views.

``QCData`` wraps a tidy DataFrame plus a typed schema plus an immutable
provenance log. It is the single input every analysis consumes and the single
output every ingestion function produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .capability import CapabilityResult
    from .control_charts import ControlChartResult
    from .gage_rr import GageRRResult


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Limits:
    """Internal value carrier for spec limits (NOT public API).

    Spec limits are QCData metadata (set via :meth:`QCData.spec`); this private,
    immutable triple just hands ``lower``/``upper``/``target`` to the analysis
    math without re-plumbing every call. There is no public ``Spec`` class.
    """

    lower: float | None = None
    upper: float | None = None
    target: float | None = None

    def has_any(self) -> bool:
        return self.lower is not None or self.upper is not None


# Roles the model understands. Arbitrary role names are accepted, but these are the
# documented ones; ``quality``/``time`` are RESERVED forward-compat hooks (accepted
# now without error so adding behavior later is non-breaking).
KNOWN_ROLES = frozenset({"subgroup", "time", "part", "operator", "replicate", "size", "quality", "event"})


@dataclass(frozen=True)
class QCMeta:
    """Typed, validated metadata describing a :class:`QCData`.

    Spec limits (``lower``/``upper``/``target``) and ``roles`` are metadata fields
    here, exactly like ``measure``/``units`` - there is no separate spec object.
    """

    measure: str
    units: str | None
    lower: float | None
    upper: float | None
    target: float | None
    roles: dict[str, str]
    subgroup_size: int | None

    @property
    def limits(self) -> _Limits:
        """The spec limits as an internal triple (for the analysis math)."""
        return _Limits(self.lower, self.upper, self.target)

    @property
    def has_spec(self) -> bool:
        return self.lower is not None or self.upper is not None


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Step:
    """One structured entry in a :class:`QCData` provenance history."""

    operation: str
    params: dict[str, Any]
    n_affected: int | None
    timestamp: datetime


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# View return types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Subgroups:
    """Measure grouped by subgroup/time, in sequence order."""

    groups: tuple[np.ndarray, ...]
    labels: tuple[Any, ...]
    sizes: tuple[int, ...]

    @property
    def equal_n(self) -> bool:
        return len(set(self.sizes)) == 1

    @property
    def n(self) -> int | None:
        return self.sizes[0] if self.equal_n and self.sizes else None


@dataclass(frozen=True)
class Crossed:
    """Part x operator x replicate structure for gage R&R."""

    frame: pd.DataFrame  # canonical columns: part, operator, replicate, value
    parts: tuple[Any, ...]
    operators: tuple[Any, ...]
    n_replicates: int
    balanced: bool


# --------------------------------------------------------------------------- #
# QCData
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class QCData:
    """Immutable wrapper over a tidy measurement frame.

    Construct via :func:`load`. Every transform returns a NEW ``QCData`` with a
    :class:`Step` appended to a copy of the history; the original is never mutated.
    """

    _frame: pd.DataFrame
    meta: QCMeta
    history: tuple[Step, ...] = field(default_factory=tuple)

    def __repr__(self) -> str:
        m = self.meta
        units = f" [{m.units}]" if m.units else ""
        roles = "{" + ", ".join(m.roles) + "}" if m.roles else "{}"
        spec_bits = [f"{k}={v}" for k, v in
                     (("lower", m.lower), ("upper", m.upper), ("target", m.target))
                     if v is not None]
        spec = "spec(" + ", ".join(spec_bits) + ")" if spec_bits else "spec(none)"
        n_steps = len(self.history)
        return (f"QCData(measure={m.measure!r}{units}, n={len(self._frame)}, "
                f"roles={roles}, {spec}, history={n_steps} "
                f"step{'s' if n_steps != 1 else ''})")

    # ---- basic access ----------------------------------------------------
    @property
    def frame(self) -> pd.DataFrame:
        """A defensive copy of the underlying tidy frame."""
        return self._frame.copy()

    def __len__(self) -> int:
        return len(self._frame)

    # ---- provenance ------------------------------------------------------
    def lineage(self) -> list[dict]:
        """The provenance chain so far, as a list of step dicts (each with its
        running ``digest``), reconstructable end to end."""
        from ._result import history_lineage
        return history_lineage(self.history)

    def provenance_digest(self) -> str:
        """SHA-256 digest pinning the full provenance chain. The history is
        append-only by construction (an immutable tuple of frozen steps); this
        digest additionally makes the recorded content verifiable. It is a content
        hash, not a signature (see :func:`mfgqc._result.history_digest`)."""
        from ._result import history_digest
        return history_digest(self.history)

    def verify_provenance(self, expected_digest: str) -> bool:
        """True iff the recorded history still matches a digest captured earlier."""
        return self.provenance_digest() == expected_digest

    def _with_step(
        self, step: Step, frame: pd.DataFrame | None = None, meta: QCMeta | None = None
    ) -> "QCData":
        return QCData(
            _frame=(self._frame if frame is None else frame).copy(),
            meta=self.meta if meta is None else meta,
            history=self.history + (step,),
        )

    # ---- role helpers ----------------------------------------------------
    def _require_roles(self, required: list[str], analysis: str) -> dict[str, str]:
        missing = [r for r in required if r not in self.meta.roles]
        if missing:
            have = sorted(self.meta.roles)
            from .errors import MissingPrerequisiteError
            raise MissingPrerequisiteError(
                f"{analysis} requires roles {set(required)}; "
                f"{missing[0]!r} not defined in this QCData "
                f"(defined roles: {have}). Set them with .roles(...).",
                analysis=analysis, missing=[f"role:{r}" for r in missing])
        return {r: self.meta.roles[r] for r in required}

    # ---- views (read-only; do NOT alter history) -------------------------
    def values(self) -> np.ndarray:
        """Flat measure vector (capability, normality).

        Returns an isolated, read-only array: ``to_numpy`` can hand back a
        writable view that aliases the frame on some numpy/pandas versions
        (seen on Python 3.10), which would let a caller corrupt the original.
        Copy and freeze so the immutability guarantee holds everywhere.
        """
        arr = self._frame[self.meta.measure].to_numpy(dtype=float, copy=True)
        arr.setflags(write=False)
        return arr

    def subgroups(self) -> Subgroups:
        """Measure grouped by subgroup/time role, in sequence order.

        If no ``subgroup``/``time`` role is defined, falls back to
        ``subgroup_size``: size 1 yields singleton subgroups (for I-MR), size
        ``k`` chunks consecutive rows into groups of ``k``. Raises if neither a
        role nor a usable ``subgroup_size`` is available.
        """
        measure = self.meta.measure
        role = None
        for candidate in ("subgroup", "time"):
            if candidate in self.meta.roles:
                role = self.meta.roles[candidate]
                break

        if role is not None:
            groups: list[np.ndarray] = []
            labels: list[Any] = []
            # Preserve first-appearance order (sequence order).
            for label, idx in self._frame.groupby(role, sort=False).groups.items():
                vals = self._frame.loc[idx, measure].to_numpy(dtype=float)
                groups.append(vals)
                labels.append(label)
            sizes = [g.size for g in groups]
            return Subgroups(tuple(groups), tuple(labels), tuple(sizes))

        size = self.meta.subgroup_size
        if size is None:
            from .errors import MissingPrerequisiteError
            raise MissingPrerequisiteError(
                "control charts require a 'subgroup' or 'time' role, or a "
                "subgroup_size; none is defined in this QCData.",
                analysis="control_chart", missing=["subgroup"])
        flat = self.values()
        if size == 1:
            groups = [np.array([v]) for v in flat]
            labels = list(range(1, len(groups) + 1))
        else:
            n_full = len(flat) // size
            groups = [flat[i * size:(i + 1) * size] for i in range(n_full)]
            rem = flat[n_full * size:]
            if rem.size:
                groups.append(rem)
            labels = list(range(1, len(groups) + 1))
        sizes = [g.size for g in groups]
        return Subgroups(tuple(groups), tuple(labels), tuple(sizes))

    def crossed(self) -> Crossed:
        """Part x operator x replicate structure for gage R&R."""
        roles = self._require_roles(["part", "operator", "replicate"], "gage R&R")
        measure = self.meta.measure
        sub = self._frame[[roles["part"], roles["operator"], roles["replicate"], measure]].copy()
        sub.columns = ["part", "operator", "replicate", "value"]
        sub = sub.dropna(subset=["value"])
        parts = tuple(pd.unique(sub["part"]))
        operators = tuple(pd.unique(sub["operator"]))
        # Balanced if every (part, operator) cell has the same replicate count.
        cell_counts = sub.groupby(["part", "operator"], sort=False).size()
        balanced = bool(cell_counts.nunique() == 1)
        n_rep = int(cell_counts.iloc[0]) if len(cell_counts) else 0
        return Crossed(sub, parts, operators, n_rep, balanced)

    # ---- optional cleaning (immutable + logged) --------------------------
    def clean(self, *, drop_na: bool = True, sigma_clip: float | None = None) -> "QCData":
        """Return a new ``QCData`` with light cleaning applied and logged.

        Parameters
        ----------
        drop_na : bool, optional
            Drop rows with a missing measure value.
        sigma_clip : float or None, optional
            If given, drop rows whose measure is more than ``sigma_clip`` overall
            standard deviations from the overall mean (basic outlier removal).
        """
        df = self._frame
        before = len(df)
        measure = self.meta.measure
        if drop_na:
            df = df.dropna(subset=[measure])
        if sigma_clip is not None:
            vals = df[measure].to_numpy(dtype=float)
            mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
            if sd > 0:
                keep = np.abs(vals - mu) <= sigma_clip * sd
                df = df.loc[df.index[keep]]
        df = df.reset_index(drop=True)
        n_affected = before - len(df)
        step = Step(
            operation="clean",
            params={"drop_na": drop_na, "sigma_clip": sigma_clip},
            n_affected=n_affected,
            timestamp=_now(),
        )
        return self._with_step(step, frame=df)

    # ---- metadata setters (immutable; each returns a NEW QCData) ----------
    def spec(self, lower: float | None = None, upper: float | None = None,
             target: float | None = None) -> "QCData":
        """Set specification limits as QCData metadata; returns a NEW QCData.

        Limits live on the QCData alongside ``measure``/``units`` - there is no
        separate spec object. Omit a bound for a one-sided spec. Each call sets the
        limits from its arguments (omitted = absent), and never mutates the original.

        Examples
        --------
        >>> data = data.spec(lower=1.0, upper=2.0, target=1.5)
        >>> data = data.spec(lower=200)          # one-sided
        >>> sweep = [data.spec(lower=1.0, upper=u).capability() for u in candidates]
        """
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("lower must be < upper")
        new_meta = replace(self.meta, lower=lower, upper=upper, target=target)
        step = Step(operation="spec",
                    params={"lower": lower, "upper": upper, "target": target},
                    n_affected=None, timestamp=_now())
        return self._with_step(step, meta=new_meta)

    def spec_from(self, other: "QCData") -> "QCData":
        """Copy spec limits (lower/upper/target) from another QCData; returns a NEW QCData.

        Plain-Python reuse for a 'named standard' without a bespoke class:
        ``wall = dict(lower=1.0, upper=2.0, target=1.5); data.spec(**wall)`` also works.
        """
        m = other.meta
        return self.spec(lower=m.lower, upper=m.upper, target=m.target)

    def roles(self, **mapping: str) -> "QCData":
        """Bind role names to columns as metadata; returns a NEW QCData.

        Flattened lowercase kwargs (no nested dict), parallel to :meth:`spec`.
        Known roles: part, operator, replicate, subgroup, time, quality, size.
        Bindings merge with any already set (e.g. a ``subgroup=`` from ``load``).

        Examples
        --------
        >>> data = data.roles(part='part', operator='operator', replicate='trial')
        """
        for role, col in mapping.items():
            if col not in self._frame.columns:
                raise ValueError(f"role {role!r} names column {col!r}, which is not in the frame.")
        new_roles = {**self.meta.roles, **mapping}
        new_meta = replace(self.meta, roles=new_roles)
        step = Step(operation="roles", params={"roles": dict(mapping)},
                    n_affected=None, timestamp=_now())
        return self._with_step(step, meta=new_meta)

    # ---- analyses (delegate to modules) ----------------------------------
    def capability(self, method: str = "normal", *, alpha: float = 0.05) -> "CapabilityResult":
        """Process-capability analysis. See :func:`mfgqc.capability.compute`."""
        from .capability import compute as _compute
        return _compute(self, method=method, alpha=alpha)

    def bayes_capability(self, *, prior=None, seed: int, draws: int = 100_000,
                         cred_level: float = 0.95):
        """Bayesian process-capability analysis (posterior indices with credible
        intervals). See :func:`mfgqc.bayes.capability.compute`."""
        from .bayes.capability import compute as _compute
        return _compute(self, prior=prior, seed=seed, draws=draws, cred_level=cred_level)

    def control_chart(self, kind: str | None = None, rules: str = "nelson",
                      n: "str | int | None" = None) -> "ControlChartResult":
        """Control-chart analysis. See :func:`mfgqc.control_charts.compute`.

        ``n`` (attribute charts) is a sample-size column name, a constant int, or
        None (falls back to a ``size`` role, else 1)."""
        from .control_charts import compute as _compute
        return _compute(self, kind=kind, rules=rules, n=n)

    def gage_rr(self, method: str = "anova", *, alpha: float = 0.10) -> "GageRRResult":
        """Gage R&R (MSA) analysis. See :func:`mfgqc.gage_rr.compute`."""
        from .gage_rr import compute as _compute
        return _compute(self, method=method, alpha=alpha)

    # ---- MSA studies beyond gage R&R -------------------------------------
    def bias_study(self, reference: float, *, alpha: float = 0.05):
        """MSA bias study vs a known ``reference``. See :func:`mfgqc.msa.bias`."""
        from . import msa
        return msa.bias(self, reference, alpha=alpha)

    def linearity_study(self, reference, *, alpha: float = 0.05):
        """MSA linearity study. ``reference`` is a column name or a {group: ref}
        mapping. See :func:`mfgqc.msa.linearity`."""
        from . import msa
        return msa.linearity(self, reference, alpha=alpha)

    def stability_study(self, *, kind: str | None = None, rules: str = "nelson"):
        """MSA stability study (control chart over time). See :func:`mfgqc.msa.stability`."""
        from . import msa
        return msa.stability(self, kind=kind, rules=rules)

    # ---- regression / ANOVA (expose the internal machinery) --------------
    def regress(self, on, *, select=None, criterion: str = "aic", model=None, start=None):
        """OLS regression of the measure on ``on``.

        ``select`` ('forward'/'backward'/'stepwise') runs automated model selection
        by ``criterion`` ('aic'/'bic'/'p'); ``model`` (a callable) with ``start``
        runs non-linear least squares instead. See :mod:`mfgqc.regression_ext`."""
        if model is not None:
            from .regression_ext import nls
            return nls(self, on, model, start)
        if select is not None:
            from .regression_ext import select as _select
            cands = [on] if isinstance(on, str) else list(on)
            return _select(self, cands, direction=select, criterion=criterion)
        from .regression import compute_regression
        return compute_regression(self, on)

    def logistic(self, on):
        """Binary logistic regression of the measure on ``on``. See
        :func:`mfgqc.regression_ext.logistic`."""
        from .regression_ext import logistic
        return logistic(self, on)

    def transform(self, method: str = "boxcox"):
        """Return a NEW QCData with the measure Box-Cox transformed; the fitted
        lambda and its CI are logged in the provenance. The transform is explicit
        and opt-in - mfgQC never transforms silently inside another analysis."""
        if method != "boxcox":
            raise ValueError("only method='boxcox' is supported.")
        from scipy import stats
        measure = self.meta.measure
        col = pd.to_numeric(self._frame[measure], errors="coerce").to_numpy(dtype=float)
        finite = col[np.isfinite(col)]
        if finite.size < 3 or np.min(finite) <= 0:
            raise ValueError("Box-Cox needs at least 3 strictly positive values; shift the data "
                             "(add a constant) or use a different transform.")
        _, lmbda, ci = stats.boxcox(finite, alpha=0.05)
        new = self._frame.copy()
        out = np.full(col.shape, np.nan)
        mask = np.isfinite(col)
        out[mask] = stats.boxcox(col[mask], lmbda=lmbda)
        new[measure] = out
        step = Step(operation="transform",
                    params={"method": "boxcox", "lambda": float(lmbda),
                            "lambda_ci": (float(ci[0]), float(ci[1]))},
                    n_affected=int(mask.sum()), timestamp=_now())
        return self._with_step(step, frame=new)

    def anova(self, factors, interaction: bool = True):
        """One-/two-way ANOVA of the measure across ``factors`` (list of 1-2 column names).

        ``interaction`` (two-way only): fit the interaction term (default) or pool
        it into error for an additive model."""
        from .regression import compute_anova
        return compute_anova(self, factors, interaction=interaction)

    # ---- reliability ------------------------------------------------------
    def life_fit(self, dist: str = "weibull", method: str = "mle", *, conf: float = 0.95):
        """Fit a life distribution with censoring (event role: 1=failure, 0=suspension).
        See :func:`mfgqc.reliability.life_fit`."""
        from .reliability.life import life_fit
        return life_fit(self, dist=dist, method=method, conf=conf)

    def life_table(self, *, conf: float = 0.95):
        """Kaplan-Meier nonparametric R(t). See :func:`mfgqc.reliability.kaplan_meier`."""
        from .reliability.nonparametric import kaplan_meier
        return kaplan_meier(self, conf=conf)

    def mtbf(self, kind: str = "time_terminated", conf: float = 0.90):
        """Constant-rate MTBF with chi-square bounds. See :func:`mfgqc.reliability.mtbf`."""
        from .reliability.demonstrate import mtbf
        return mtbf(self, kind=kind, conf=conf)

    # ---- time-series & multi-vari ----------------------------------------
    def timeseries(self, *, lags: int = 20):
        """Exploratory time-series screen: trend (linear + Mann-Kendall) and
        autocorrelation (ACF), surfaced as flags. See
        :func:`mfgqc.timeseries.compute_timeseries`."""
        from .timeseries import compute_timeseries
        return compute_timeseries(self, lags=lags)

    def multivari(self, factors):
        """Multi-vari study decomposing variation into positional / cyclical /
        temporal families across the nested ``factors``. See
        :func:`mfgqc.multivari.compute`."""
        from .multivari import compute
        return compute(self, factors)

    # ---- SPC additions ---------------------------------------------------
    def precontrol(self):
        """Pre-control (stoplight) against the attached spec limits. See
        :func:`mfgqc.precontrol.compute`."""
        from .precontrol import compute
        return compute(self)

    def short_run_chart(self, by: str, target=None, *, rules: str = "nelson"):
        """Standardized short-run control chart pooling part numbers via ``by``.
        See :func:`mfgqc.control_charts.compute_short_run`."""
        from .control_charts import compute_short_run
        return compute_short_run(self, by, target=target, rules=rules)

    # ---- attributes capability -------------------------------------------
    def attribute_capability(self, defect: str | None = None,
                             opportunities: int | None = None, kind: str | None = None):
        """Attributes capability (DPMO, yields, process sigma) from a defect or
        pass/fail column. ``defect`` defaults to the measure; ``kind`` is inferred
        (binary column -> defectives, counts -> defects). See
        :func:`mfgqc.process_sigma`."""
        from .process_sigma import compute
        col = defect or self.meta.measure
        if col not in self._frame.columns:
            raise ValueError(f"defect column {col!r} not found in the frame.")
        vals = pd.to_numeric(self._frame[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        units = vals.size
        total = float(np.sum(vals))
        if kind is None:
            uniq = set(np.unique(vals).tolist())
            kind = "defectives" if uniq <= {0.0, 1.0} else "defects"
        return compute(total, units, opportunities=opportunities or 1, kind=kind)

    def attribute_agreement(self, rating: str, part: str, appraiser: str,
                            reference=None, *, trial: str | None = None,
                            ordinal: bool = False):
        """Attribute MSA / kappa: within-, between-, and vs-reference agreement.
        See :func:`mfgqc.attribute_agreement.compute`."""
        from .attribute_agreement import compute
        return compute(self, rating, part, appraiser, reference=reference,
                       trial=trial, ordinal=ordinal)

    # ---- design of experiments -------------------------------------------
    def doe(self, design=None, factors=None, order=None, reduce: bool = False):
        """Analyze a two-level designed experiment on the measure.

        Pass either ``design=`` (a :class:`mfgqc.doe.Design`, which also supplies
        the alias structure) or ``factors=`` (an external coded matrix). See
        :func:`mfgqc.doe.doe`."""
        from .doe import doe as _doe
        return _doe(self, design=design, factors=factors, order=order, reduce=reduce)

    # ---- time-series control charts --------------------------------------
    def ewma_chart(self, lam: float = 0.1, L: float = 2.7, *,
                   mu0: float | None = None, sigma: float | None = None):
        """EWMA control chart. See :func:`mfgqc.timeseries_charts.compute_ewma`.

        ``mu0``/``sigma`` default to the sample mean (or spec target) and MR-bar/d2;
        pass known phase-I values to use them directly."""
        from .timeseries_charts import compute_ewma
        return compute_ewma(self, lam=lam, L=L, mu0=mu0, sigma=sigma)

    def cusum_chart(self, k: float = 0.5, h: float = 5, *,
                    mu0: float | None = None, sigma: float | None = None):
        """Tabular two-sided CUSUM chart. See :func:`mfgqc.timeseries_charts.compute_cusum`.

        ``mu0``/``sigma`` default as in :meth:`ewma_chart`; pass known phase-I values
        to use them directly."""
        from .timeseries_charts import compute_cusum
        return compute_cusum(self, k=k, h=h, mu0=mu0, sigma=sigma)

    # ---- hypothesis tests (delegate to the hypothesis module) ------------
    def _split_by(self, by: str) -> tuple[list, list[np.ndarray]]:
        """Split the measure into groups by a role name or column. Sequence order."""
        col = self.meta.roles.get(by, by)
        if col not in self._frame.columns:
            raise ValueError(f"grouping {by!r} is not a defined role or column.")
        measure = self.meta.measure
        labels: list = []
        groups: list[np.ndarray] = []
        for label, idx in self._frame.groupby(col, sort=False).groups.items():
            labels.append(label)
            groups.append(self._frame.loc[idx, measure].to_numpy(dtype=float))
        return labels, groups

    def test_mean(self, target: float, **kwargs):
        """One-sample test of the measure mean vs ``target``."""
        from . import hypothesis
        return hypothesis.test_mean(self.values(), target, base_history=self.history, **kwargs)

    def _grouping(self, by: str | None) -> str:
        """Resolve the grouping for a split-by analysis. When ``by`` is omitted,
        fall back to a ``group`` role (the consolidated idiom: bind it via .roles())."""
        if by is not None:
            return by
        if "group" in self.meta.roles:
            return "group"
        raise ValueError(
            "no grouping given: pass by='<column>' or bind a 'group' role via "
            ".roles(group='<column>').")

    def test_means(self, by: str | None = None, **kwargs):
        """Two-sample test of the measure split by a dimension (must yield 2 groups).

        ``by`` defaults to the ``group`` role when omitted."""
        from . import hypothesis
        by = self._grouping(by)
        labels, groups = self._split_by(by)
        if len(groups) != 2:
            raise ValueError(f"test_means needs exactly 2 groups in {by!r}; found {len(groups)}.")
        return hypothesis.test_means(groups[0], groups[1],
                                     labels=tuple(str(label) for label in labels),
                                     base_history=self.history, **kwargs)

    def test_anova(self, by: str | None = None, **kwargs):
        """One-way ANOVA of the measure split by a dimension (defaults to the ``group`` role)."""
        from . import hypothesis
        by = self._grouping(by)
        labels, groups = self._split_by(by)
        return hypothesis.test_anova(*groups, labels=tuple(str(label) for label in labels),
                                     base_history=self.history, **kwargs)

    def test_variance(self, by: str | None = None, **kwargs):
        """Variance comparison of the measure split by a dimension (defaults to the ``group`` role)."""
        from . import hypothesis
        by = self._grouping(by)
        labels, groups = self._split_by(by)
        return hypothesis.test_variance(*groups, labels=tuple(str(label) for label in labels),
                                        base_history=self.history, **kwargs)


# --------------------------------------------------------------------------- #
# Ingestion (top-level, pandas idiom)
# --------------------------------------------------------------------------- #
def _from_dataframe(
    df: pd.DataFrame,
    *,
    measure: str,
    lower: float | None = None,
    upper: float | None = None,
    target: float | None = None,
    roles: dict[str, str] | None = None,
    units: str | None = None,
    subgroup_size: int | None = None,
    operation: str = "load",
) -> QCData:
    """Internal QCData builder. The public entry point is :func:`load`.

    Validates the measure column and declared roles, attaches spec metadata, and
    seeds the provenance history. Kept private so the package presents a single
    public ingestion idiom (``load`` + :meth:`QCData.spec` / :meth:`QCData.roles`).
    """
    if measure not in df.columns:
        raise ValueError(f"measure column {measure!r} not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[measure]):
        raise ValueError(f"measure column {measure!r} must be numeric.")
    roles = dict(roles or {})
    for role, col in roles.items():
        if col not in df.columns:
            raise ValueError(f"role {role!r} names column {col!r}, which is not in the DataFrame.")
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("lower must be < upper")

    meta = QCMeta(
        measure=measure, units=units, lower=lower, upper=upper, target=target,
        roles=roles, subgroup_size=subgroup_size,
    )
    frame = df.reset_index(drop=True).copy()
    step = Step(
        operation=operation,
        params={
            "measure": measure,
            "roles": roles,
            "units": units,
            "subgroup_size": subgroup_size,
            "spec": {"lower": lower, "upper": upper, "target": target},
        },
        n_affected=len(frame),
        timestamp=_now(),
    )
    return QCData(_frame=frame, meta=meta, history=(step,))


# --------------------------------------------------------------------------- #
# load() - the fluent entry point
# --------------------------------------------------------------------------- #
def load(
    source: pd.DataFrame,
    *,
    measure: str,
    subgroup: str | None = None,
    roles: dict[str, str] | None = None,
    units: str | None = None,
    subgroup_size: int | None = None,
) -> QCData:
    """Load a source into an immutable :class:`QCData` (the canonical entry point).

    This is THE public idiom: ``load(df, measure=...)`` then attach metadata with
    :meth:`QCData.spec` / :meth:`QCData.roles`, then call an analysis. A frontend
    should target this path. To load a CSV, read it with pandas first:
    ``load(pandas.read_csv(path), measure=...)``.

    Parameters
    ----------
    source : pandas.DataFrame
        The data. A DataFrame today; the signature is kept open so a file path or
        connection can be added later without changing the call shape.
    measure : str
        Name of the numeric measure column.
    subgroup : str or None, optional
        Sugar for the common single ``subgroup`` binding.
    roles : dict, optional
        Backward-compatible role binding; the primary path for multi-role binding is
        the :meth:`QCData.roles` setter.
    units, subgroup_size : optional

    Returns
    -------
    QCData

    Notes
    -----
    Spec limits attach via :meth:`QCData.spec` and roles via :meth:`QCData.roles` -
    metadata setters that each return a new immutable QCData - keeping ``load``'s
    signature small.
    """
    if not isinstance(source, pd.DataFrame):
        raise TypeError(
            "load() currently accepts a pandas DataFrame; file/connection sources "
            "are a planned future addition.")
    roles = dict(roles or {})
    if subgroup is not None:
        roles.setdefault("subgroup", subgroup)
    qc = _from_dataframe(source, measure=measure, roles=roles,
                         units=units, subgroup_size=subgroup_size, operation="load")
    # Absorb a cleaning report carried on the DataFrame (clean() attaches it via attrs).
    clean_steps = tuple(getattr(source, "attrs", {}).get("mfgqc_clean_steps", ()))
    if not clean_steps:
        return qc
    load_step = replace(qc.history[0], operation="load")
    return QCData(_frame=qc._frame, meta=qc.meta, history=clean_steps + (load_step,))


def from_wide(
    df: pd.DataFrame,
    *,
    id_vars: list[str] | str,
    value_vars: list[str] | None = None,
    measure: str = "value",
    var_name: str = "variable",
    subgroup: str | None = None,
    roles: dict[str, str] | None = None,
    units: str | None = None,
    subgroup_size: int | None = None,
) -> QCData:
    """Load a WIDE table (id columns + one column per tag/replicate) by reshaping to
    long, then loading. The internal tidy form is identical to the tall path.

    Parameters
    ----------
    id_vars : list or str
        Identifier columns to keep (e.g. a timestamp or a batch id).
    value_vars : list or None
        Columns to melt into rows; default all non-id columns.
    measure : str
        Name for the melted value column (becomes the measure).
    var_name : str
        Name for the melted tag/replicate column.
    """
    long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=measure)
    qc = load(long, measure=measure, subgroup=subgroup, roles=roles,
              units=units, subgroup_size=subgroup_size)
    reshape = replace(qc.history[-1], operation="from_wide",
                      params={**qc.history[-1].params, "id_vars": id_vars, "var_name": var_name})
    return QCData(_frame=qc._frame, meta=qc.meta, history=qc.history[:-1] + (reshape,))
