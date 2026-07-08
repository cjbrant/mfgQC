"""Short-run Bayesian control chart (spec Algorithm L; lineage Colosimo & del
Castillo ch. 3 and ch. 6, simplified).

v1 design (a deliberate simplification, stated in the report): sequential
conjugate Normal-Inverse-chi2 updating where each subgroup's posterior becomes the
next subgroup's prior (engine A chaining, which is the T1.3 sequential=batch
coherence). The chart statistic at each stage is P(|mu - target| > d | data),
evaluated from the closed-form t marginal for mu, and the stage flags when that
probability exceeds ``p_star``. There is no jump-mixture / random-walk-with-jumps
model (C&dC ch. 3's version is a documented future enhancement).

Because the reference is the accumulated posterior, a slow sustained drift is
gradually absorbed into it and may produce a delayed or absent signal - the
drift-absorption caveat, carried verbatim in the report. Short runs need a proper
prior to have a scale from the first part; a noninformative start is allowed only
with allow_vague=True, which warns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from mfgqc._result import QCResult, history_digest
from mfgqc.assumptions import AssumptionCheck
from mfgqc.data import Step

from ._results import _assumption_step, _now, data_digest
from .conjugate import suffstats, update

_DRIFT_CAVEAT = (
    "Drift-absorption caveat: each subgroup's posterior becomes the next subgroup's "
    "prior, so a slow sustained drift is gradually absorbed into the reference and "
    "may produce a delayed or absent signal. This chart detects change relative to "
    "the accumulated history, not an external fixed reference; pair it with a "
    "fixed-target rule when detecting slow drift matters."
)


def _chart_statistic(mun: float, kn: float, nun: float, sn2: float,
                     target: float, d: float) -> float:
    """P(|mu - target| > d | data) from the posterior t marginal for mu."""
    scale = np.sqrt(sn2 / kn)
    if scale <= 0.0:
        # degenerate point mass at mun: the probability is 0 or 1, not NaN.
        return 1.0 if abs(mun - target) > d else 0.0
    dist = stats.t(df=nun, loc=mun, scale=scale)
    return float(dist.sf(target + d) + dist.cdf(target - d))


@dataclass(frozen=True, repr=False)
class ShortRunResult(QCResult):
    """Sequential short-run chart (immutable). One stage per subgroup."""

    n_stages: int
    target: float
    d: float
    p_star: float
    cred_level: float
    prior_family: str
    allow_vague: bool
    stage_n: tuple
    stage_mean: tuple
    stage_posterior: tuple      # (mun, kn, nun, sn2) per stage
    chart_stat: tuple
    flags: tuple
    assumptions: list = field(default_factory=list)
    history: tuple = field(default_factory=tuple)

    def chart_statistic(self, stage: int) -> float:
        return self.chart_stat[stage]

    def first_flag(self) -> int | None:
        """Index of the first flagged stage, or None if the run stays in control."""
        return next((i for i, f in enumerate(self.flags) if f), None)

    def _title(self) -> str:
        return "Bayesian Short-Run Chart"

    def _summary_lines(self) -> list[str]:
        first = self.first_flag()
        lines = [
            f"stages = {self.n_stages}   target = {self.target:.5g}   d = {self.d:.4g}   "
            f"p* = {self.p_star:.2f}",
            f"prior = {self.prior_family}" + ("  (vague start)" if self.allow_vague else ""),
            f"first flag = {'stage ' + str(first) if first is not None else 'none (in control)'}",
        ]
        for i, (m, stat, flag) in enumerate(zip(self.stage_mean, self.chart_stat, self.flags)):
            mark = "  [FLAG]" if flag else ""
            lines.append(f"  stage {i}: mean={m:.5g}  P(|mu-target|>d)={stat:.3g}{mark}")
        lines.append(_DRIFT_CAVEAT)
        return lines


def shortrun(subgroups, *, target: float, d: float | None = None,
             lower: float | None = None, upper: float | None = None, prior=None,
             p_star: float = 0.9, allow_vague: bool = False, cred_level: float = 0.95,
             base_history: tuple = ()) -> ShortRunResult:
    """Sequential short-run capability/monitoring chart.

    ``subgroups`` is a list of per-stage measurement arrays (or a 2-D array, one
    row per stage). The posterior chains stage to stage. ``d`` defaults to
    (upper - lower) / 12 and requires both spec limits if not given. A proper
    ``prior`` (NormalPrior) is required unless ``allow_vague=True``, which starts
    noninformative from the first subgroup (needs n>=2 there) and warns.
    """
    groups = [np.asarray(g, dtype=float) for g in subgroups]
    groups = [g[~np.isnan(g)] for g in groups]
    if len(groups) == 0:
        raise ValueError("shortrun needs at least one subgroup.")

    if d is None:
        if lower is None or upper is None:
            raise ValueError("shortrun needs an explicit d, or both spec limits to "
                             "default d = (upper - lower) / 12.")
        d = (upper - lower) / 12.0
    if d <= 0:
        raise ValueError(f"d must be positive; got {d}.")

    checks: list = []
    if prior is None:
        if not allow_vague:
            raise ValueError(
                "shortrun requires a proper prior so the first short run has a scale. "
                "Pass a NormalPrior, or set allow_vague=True to start noninformative "
                "(a weaker, prior-sensitive chart).")
        n0, ybar0, s20 = suffstats(groups[0])
        if n0 < 2:
            raise ValueError("a vague start needs n>=2 in the first subgroup to form "
                             "an initial variance.")
        if s20 <= 0.0:
            raise ValueError(
                "a vague start needs positive variance in the first subgroup; its "
                "measurements are identical, so the noninformative chart has no scale. "
                "Supply a proper NormalPrior.")
        cur = (ybar0, float(n0), float(n0 - 1), s20)
        first_stage = [(n0, ybar0, cur)]
        rest = groups[1:]
        prior_family = "vague"
        checks.append(AssumptionCheck(
            name="vague_start", test="proper prior supplied", statistic=0.0,
            p_value=None, passed=False, magnitude=None, magnitude_label=None,
            reliability="low_power", n=int(n0),
            recommendation=("started from a noninformative prior; early stages are wide "
                            "and prior sensitive. Supply a proper NormalPrior for a "
                            "stable short-run chart.")))
    else:
        cur = (prior.mu0, prior.k0, prior.nu0, prior.s20)
        first_stage = []
        rest = groups
        prior_family = prior.to_params()["family"]

    stages = list(first_stage)
    for g in rest:
        n, ybar, s2 = suffstats(g)
        s2 = 0.0 if n < 2 else s2      # (n-1)*s2 term vanishes for n=1 short runs
        cur = update(cur[0], cur[1], cur[2], cur[3], n, ybar, s2)
        stages.append((n, ybar, cur))

    stage_n, stage_mean, stage_post, stat_list, flags = [], [], [], [], []
    history = list(base_history)
    for i, (n, ybar, post) in enumerate(stages):
        mun, kn, nun, sn2 = post
        stat = _chart_statistic(mun, kn, nun, sn2, target, d)
        flag = stat > p_star
        prev_digest = history_digest(tuple(history))
        step = Step(
            operation="bayes.shortrun_stage",
            params={
                "stage": i, "n": int(n), "ybar": float(ybar),
                "posterior": {"mun": float(mun), "kn": float(kn),
                              "nun": float(nun), "sn2": float(sn2)},
                "chart_stat": float(stat), "flag": bool(flag),
                "target": float(target), "d": float(d), "p_star": float(p_star),
                "prev_digest": prev_digest,
                "data_sha256": data_digest(groups[i]),
            },
            n_affected=int(n), timestamp=_now(),
        )
        history.append(step)
        stage_n.append(int(n))
        stage_mean.append(float(ybar))
        stage_post.append((float(mun), float(kn), float(nun), float(sn2)))
        stat_list.append(float(stat))
        flags.append(bool(flag))

    history.extend(_assumption_step(a) for a in checks)

    return ShortRunResult(
        n_stages=len(stages), target=float(target), d=float(d), p_star=float(p_star),
        cred_level=float(cred_level), prior_family=prior_family, allow_vague=bool(allow_vague),
        stage_n=tuple(stage_n), stage_mean=tuple(stage_mean),
        stage_posterior=tuple(stage_post), chart_stat=tuple(stat_list), flags=tuple(flags),
        assumptions=checks, history=tuple(history),
    )
