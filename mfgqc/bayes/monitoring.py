"""Phase-1 reference and Bayesian monitoring (spec Algorithm G; BDA3 sec 6.3;
Hoff sec 4.4).

phase1() freezes a reference posterior from an in-control dataset into a
:class:`FrozenReference` (immutable, digest-verified, serializable). monitor()
then screens new subgroups against that reference by posterior-predictive
replication: for each candidate subgroup size m it draws R replicate subgroups
from the reference's posterior predictive, forms each test statistic, and reports
a two-sided Bayesian p-value per subgroup and test. The replicate parameter draws
follow the normative call order (chisquare then normal), so the worked-example
monitor p-values (spec T2.5(d)) reproduce bit for bit.

A monitor verdict can only be built from a FrozenReference, never from raw data,
and it carries the reference digest so a verdict is always traceable to the
reference that produced it (T5.4). Custom test callables are hashed by source into
the provenance chain (T5.5).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass, field, fields

import numpy as np

from mfgqc._result import QCResult
from mfgqc.data import Step

from ._results import _now, data_digest
from .conjugate import update


# --------------------------------------------------------------------------- #
# Built-in test quantities. Each maps a subgroup (1-D) to a scalar and a
# replicate array (..., m) to leading-axis values, via axis=-1.
# --------------------------------------------------------------------------- #
def _t_mean(a):
    return a.mean(axis=-1)


def _t_sd(a):
    return a.std(axis=-1, ddof=1)


def _t_min(a):
    return a.min(axis=-1)


def _t_max(a):
    return a.max(axis=-1)


def _t_lag1(a):
    """Lag-1 autocorrelation along the last axis (0 when the row is constant)."""
    x = a - a.mean(axis=-1, keepdims=True)
    num = (x[..., :-1] * x[..., 1:]).sum(axis=-1)
    den = (x * x).sum(axis=-1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


_BUILTIN_TESTS = {
    "mean": _t_mean, "sd": _t_sd, "min": _t_min, "max": _t_max,
    "lag1_autocorr": _t_lag1,
}


def _resolve_test(t) -> tuple:
    """Return (name, fn, spec) for a test given as a built-in name or a callable.

    ``spec`` is the provenance record: a built-in records its name; a callable
    records its name and the sha256 of its source (T5.5)."""
    if isinstance(t, str):
        if t not in _BUILTIN_TESTS:
            raise ValueError(f"unknown test {t!r}; choose from {sorted(_BUILTIN_TESTS)} "
                             f"or pass a callable.")
        return t, _BUILTIN_TESTS[t], {"name": t}
    if callable(t):
        src = inspect.getsource(t).encode()
        digest = hashlib.sha256(src).hexdigest()
        fn = _wrap_callable(t)
        return t.__name__, fn, {"name": t.__name__, "source_sha256": digest}
    raise TypeError(f"test must be a built-in name or a callable, got {type(t).__name__}.")


def _wrap_callable(f):
    """Apply a user statistic to a replicate array row-wise, keeping the scalar
    contract for a single subgroup."""
    def fn(a):
        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            return f(a)
        return np.apply_along_axis(f, 1, a)
    return fn


# --------------------------------------------------------------------------- #
# Frozen reference
# --------------------------------------------------------------------------- #
_REFERENCE_CONTENT = ("mun", "kn", "nun", "sn2", "nu_w", "sw2",
                      "n", "ybar", "s", "prior", "data_sha256")


def _reference_digest(content: dict) -> str:
    """SHA-256 over the integrity-bearing reference content (created_at excluded,
    matching the Step digest convention that pins the computation, not the clock)."""
    payload = {k: content[k] for k in _REFERENCE_CONTENT}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class FrozenReference:
    """An immutable phase-1 reference posterior, digest-verified and serializable.

    Holds the overall Normal-Inverse-chi2 posterior (mun, kn, nun, sn2) and, when
    phase-1 data arrived as subgroups, the pooled within-subgroup posterior
    (nu_w, sw2). ``digest`` pins the content so a tampered reload is rejected."""

    mun: float
    kn: float
    nun: float
    sn2: float
    nu_w: float | None
    sw2: float | None
    n: int
    ybar: float
    s: float
    prior: dict | None
    data_sha256: str
    created_at: str
    digest: str

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "FrozenReference":
        """Rebuild a reference, recomputing and verifying its content digest."""
        recomputed = _reference_digest(d)
        if recomputed != d.get("digest"):
            raise ValueError(
                "FrozenReference digest mismatch: the stored content has been "
                "altered since it was frozen.")
        return cls(**{f.name: d[f.name] for f in fields(cls)})


def phase1(y, *, prior=None) -> FrozenReference:
    """Freeze an in-control dataset into a monitoring reference.

    ``y`` may be a flat measurement vector or a 2-D array of subgroups (rows). The
    overall posterior is the noninformative BDA3 sec 3.2 fit when ``prior`` is
    None, else the Normal-Inverse-chi2 conjugate update. When ``y`` is 2-D the
    pooled within-subgroup variance posterior is also stored (disclosure only; the
    monitor screens on the overall posterior predictive).
    """
    arr = np.asarray(y, dtype=float)
    flat = arr.ravel()
    flat = flat[~np.isnan(flat)]
    n = int(flat.size)
    if n < 2:
        raise ValueError(f"phase1 needs n>=2 to form the reference variance; got n={n}.")

    ybar = float(flat.mean())
    s2 = float(flat.var(ddof=1))

    if prior is None:
        mun, kn, nun, sn2 = ybar, float(n), float(n - 1), s2
        prior_disc = None
    else:
        mun, kn, nun, sn2 = update(prior.mu0, prior.k0, prior.nu0, prior.s20, n, ybar, s2)
        prior_disc = prior.to_params()

    nu_w = sw2 = None
    if arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] >= 2:
        ss_w = float(np.nansum((arr - np.nanmean(arr, axis=1, keepdims=True)) ** 2))
        counts = np.sum(~np.isnan(arr), axis=1)
        dof_w = float(np.sum(counts - 1))
        if dof_w > 0:
            if prior is None:
                nu_w, sw2 = dof_w, ss_w / dof_w
            else:
                nu_w = prior.nu0 + dof_w
                sw2 = (prior.nu0 * prior.s20 + ss_w) / nu_w

    content = {
        "mun": mun, "kn": kn, "nun": nun, "sn2": sn2,
        "nu_w": nu_w, "sw2": sw2,
        "n": n, "ybar": ybar, "s": math.sqrt(s2),
        "prior": prior_disc, "data_sha256": data_digest(flat),
    }
    return FrozenReference(
        **content, created_at=_now().isoformat(), digest=_reference_digest(content))


# --------------------------------------------------------------------------- #
# Monitor result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class MonitorResult(QCResult):
    """Per-subgroup Bayesian monitoring verdicts against a frozen reference."""

    reference_digest: str
    tests: tuple
    alpha: float
    R: int
    seed: int
    labels: tuple
    flags: list
    _p: list = field(default_factory=list, repr=False)
    _specs: list = field(default_factory=list, repr=False)
    assumptions: list = field(default_factory=list)
    history: tuple = field(default_factory=tuple)

    def p_values_matrix(self) -> list:
        """Rows = subgroups, columns = tests (in ``self.tests`` order)."""
        return [list(row) for row in self._p]

    def test_specs(self) -> list:
        """Provenance records for the tests (built-in name, or name + source hash)."""
        return [dict(s) for s in self._specs]

    def _title(self) -> str:
        return "Bayesian Monitoring"

    def _summary_lines(self) -> list[str]:
        k = len(self.tests)
        fw = 1.0 - (1.0 - self.alpha) ** k
        lines = [
            f"reference digest = {self.reference_digest[:12]}...",
            f"tests = {', '.join(self.tests)}   alpha = {self.alpha:.3g}   R = {self.R}",
            f"subgroups = {len(self.labels)}   flagged = {sum(self.flags)}",
        ]
        for label, row, flag in zip(self.labels, self._p, self.flags):
            cells = "  ".join(f"{name}={p:.3g}" for name, p in zip(self.tests, row))
            mark = "  [FLAG]" if flag else ""
            lines.append(f"  {label}: {cells}{mark}")
        lines.append(f"family-wise alpha per subgroup (k={k}): 1-(1-alpha)^k = {fw:.3g}")
        return lines

    def _render_standalone(self, fig, kind, **kwargs) -> None:
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs) -> None:
        if kind not in (None, "pvalues"):
            raise ValueError(f"unknown monitor view kind={kind!r}; use None.")
        from . import plotting
        plotting.monitor_axes(ax, self)


def monitor(reference, subgroups, *, tests=("mean", "sd", "min", "max", "lag1_autocorr"),
            alpha: float = 0.005, R: int = 10_000, seed: int,
            labels=None) -> MonitorResult:
    """Screen new ``subgroups`` against a phase-1 ``reference``.

    ``reference`` must be a :class:`FrozenReference` (a verdict cannot be built
    from raw data, T5.4). For each distinct subgroup size the reference posterior
    predictive is sampled R times (parameter draws reused across sizes); each test
    statistic's two-sided Bayesian p-value is p = min(2*min(phi, 1-phi), 1) with
    phi = mean(T_replicate >= T_observed). A subgroup flags when any test p < alpha.
    """
    if not isinstance(reference, FrozenReference):
        raise TypeError(
            "monitor requires a FrozenReference from phase1(), not raw data; "
            "call phase1(in_control_data) first.")

    subs = [np.asarray(s, dtype=float) for s in subgroups]
    if labels is None:
        labels = tuple(f"sub-{i:03d}" for i in range(len(subs)))
    else:
        labels = tuple(labels)

    resolved = [_resolve_test(t) for t in tests]
    names = tuple(name for name, _, _ in resolved)
    specs = [spec for _, _, spec in resolved]

    rng = np.random.default_rng(seed)
    sig2r = reference.nun * reference.sn2 / rng.chisquare(reference.nun, R)
    mur = rng.normal(reference.mun, np.sqrt(sig2r / reference.kn))
    sigr = np.sqrt(sig2r)

    rep_cache: dict = {}
    stat_cache: dict = {}

    def rep_stats(m: int) -> dict:
        if m not in rep_cache:
            rep_cache[m] = rng.normal(mur[:, None], sigr[:, None], size=(R, m))
        reps = rep_cache[m]
        if m not in stat_cache:
            stat_cache[m] = {}
        cache = stat_cache[m]
        for name, fn, _ in resolved:
            if name not in cache:
                cache[name] = fn(reps)
        return cache

    p_matrix: list = []
    flags: list = []
    for sub in subs:
        m = sub.size
        cache = rep_stats(m)
        row: list = []
        for name, fn, _ in resolved:
            t_obs = float(fn(sub))
            phi = float((cache[name] >= t_obs).mean())
            row.append(min(2.0 * min(phi, 1.0 - phi), 1.0))
        p_matrix.append(row)
        flags.append(any(p < alpha for p in row))

    step = Step(
        operation="bayes.monitor",
        params={
            "reference_digest": reference.digest,
            "tests": specs,
            "alpha": float(alpha), "R": int(R), "seed": int(seed),
            "n_subgroups": len(subs),
        },
        n_affected=len(subs),
        timestamp=_now(),
    )
    return MonitorResult(
        reference_digest=reference.digest, tests=names, alpha=float(alpha),
        R=int(R), seed=int(seed), labels=labels, flags=flags,
        _p=p_matrix, _specs=specs, history=(step,),
    )


# --------------------------------------------------------------------------- #
# Posterior predictive check (BDA3 sec 6.3)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, repr=False)
class PredictiveCheckResult(QCResult):
    """A posterior predictive check: how extreme a test statistic of the observed
    data is under the model's own posterior predictive distribution."""

    statistic: str
    t_observed: float
    p_bayes: float
    p_two_sided: float
    n: int
    R: int
    seed: int
    reference_digest: str
    assumptions: list = field(default_factory=list)
    history: tuple = field(default_factory=tuple)

    def _title(self) -> str:
        return "Posterior Predictive Check"

    def _summary_lines(self) -> list[str]:
        return [
            f"statistic T = {self.statistic}   T(y_obs) = {self.t_observed:.4g}",
            f"Bayesian p-value P(T_rep >= T_obs) = {self.p_bayes:.3g}",
            f"two-sided p = {self.p_two_sided:.3g}   (n = {self.n}, R = {self.R})",
        ]

    def _render_standalone(self, fig, kind, **kwargs) -> None:
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs) -> None:
        if kind not in (None, "pvalue"):
            raise ValueError(f"unknown predictive-check view kind={kind!r}; use None.")
        from . import plotting
        plotting.predictive_check_axes(ax, self)


def predictive_check(y, statistic="min", *, prior=None, R: int = 10_000, seed: int,
                     base_history: tuple = ()) -> PredictiveCheckResult:
    """Posterior predictive check of a test ``statistic`` on data ``y`` (BDA3 sec 6.3).

    Fits the reference posterior to ``y`` (noninformative unless a ``prior`` is
    given), draws R replicate datasets of the same size from the posterior
    predictive, and reports the Bayesian p-value P(T(y_rep) >= T(y_obs)) with its
    two-sided form min(2*min(p, 1-p), 1). A p near 0 or 1 (two-sided near 0) marks
    the observed statistic as inconsistent with the model, as in BDA3's
    speed-of-light min check.
    """
    name, fn, _ = _resolve_test(statistic)
    ref = phase1(y, prior=prior)
    flat = np.asarray(y, dtype=float).ravel()
    flat = flat[~np.isnan(flat)]
    n = flat.size

    rng = np.random.default_rng(seed)
    sig2r = ref.nun * ref.sn2 / rng.chisquare(ref.nun, R)
    mur = rng.normal(ref.mun, np.sqrt(sig2r / ref.kn))
    reps = rng.normal(mur[:, None], np.sqrt(sig2r)[:, None], size=(R, n))

    t_obs = float(fn(flat))
    p_upper = float((fn(reps) >= t_obs).mean())
    p_two = min(2.0 * min(p_upper, 1.0 - p_upper), 1.0)

    step = Step(
        operation="bayes.predictive_check",
        params={
            "reference_digest": ref.digest, "statistic": name,
            "R": int(R), "seed": int(seed), "n": int(n),
        },
        n_affected=int(n),
        timestamp=_now(),
    )
    return PredictiveCheckResult(
        statistic=name, t_observed=t_obs, p_bayes=p_upper, p_two_sided=p_two,
        n=int(n), R=int(R), seed=int(seed), reference_digest=ref.digest,
        history=tuple(base_history) + (step,),
    )
