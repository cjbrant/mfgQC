"""Attribute agreement analysis (attribute MSA / kappa).

Appraisers rate items (binary or ordinal) over one or more trials, optionally
against a known reference standard. Parallels gage R&R but for categorical
ratings; reuses the item x appraiser x trial (crossed) structure.

Reports within-appraiser agreement (repeatability), between-appraiser agreement
(reproducibility, Fleiss for more than two appraisers, Cohen for two), and each
appraiser versus the standard (accuracy) when a reference is given. Weighted
kappa is used for ordinal ratings.

Surfacing: kappa carries the standard interpretation bands as labeled context,
not baked into ``passed``; the binary adequacy flag is percent agreement versus a
threshold. The known kappa pathology (high agreement plus skewed marginals
deflates kappa) is flagged rather than allowed to silently fail an agreeing
system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from ._result import QCResult
from .assumptions import AssumptionCheck
from .data import QCData, Step

# Landis-Koch interpretation bands (labeled context, never a baked-in verdict).
_BANDS = [(0.0, "poor"), (0.20, "slight"), (0.40, "fair"), (0.60, "moderate"),
          (0.80, "substantial"), (1.01, "almost perfect")]
_PCT_THRESHOLD = 0.90          # AIAG-style acceptability for percent agreement


def _now() -> datetime:
    return datetime.now(timezone.utc)


def band(k: float) -> str:
    if not np.isfinite(k):
        return "undefined"
    for hi, label in _BANDS:
        if k < hi:
            return label
    return "almost perfect"


def _wilson(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = x / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def cohen_kappa(a: np.ndarray, b: np.ndarray, categories=None, weights=None) -> float:
    """Cohen's kappa for two raters; ``weights`` in {None, 'linear', 'quadratic'}."""
    a = np.asarray(a); b = np.asarray(b)
    cats = list(categories) if categories is not None else sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    n = a.size
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    O /= n
    r = O.sum(axis=1); c = O.sum(axis=0)
    E = np.outer(r, c)
    if weights is None:
        w = 1.0 - np.eye(k)
    else:
        ii = np.arange(k)
        diff = np.abs(ii[:, None] - ii[None, :]).astype(float)
        w = diff if weights == "linear" else diff ** 2
    num = float((w * O).sum())
    den = float((w * E).sum())
    return 1.0 - num / den if den > 0 else float("nan")


def fleiss_kappa(counts: np.ndarray) -> float:
    """Fleiss' kappa. ``counts[i, j]`` = number of raters assigning category j to
    item i (each row sums to the number of raters)."""
    counts = np.asarray(counts, dtype=float)
    N, k = counts.shape
    n_rater = counts.sum(axis=1)
    if not np.allclose(n_rater, n_rater[0]):
        # unequal raters per item: fall back to the mean per-item rater count
        pass
    nr = n_rater.mean()
    p_j = counts.sum(axis=0) / (N * nr)
    P_i = (np.sum(counts ** 2, axis=1) - nr) / (nr * (nr - 1))
    P_bar = P_i.mean()
    P_e = float(np.sum(p_j ** 2))
    return (P_bar - P_e) / (1 - P_e) if (1 - P_e) > 0 else float("nan")


def _marginal_skew(a: np.ndarray, b: np.ndarray) -> float:
    """Prevalence of the most common category across both raters (0.5 balanced,
    near 1.0 skewed). High skew + high agreement deflates kappa (the paradox)."""
    vals = np.concatenate([a, b])
    _, counts = np.unique(vals, return_counts=True)
    return float(counts.max() / vals.size)


@dataclass(frozen=True, repr=False)
class AttributeAgreementResult(QCResult):
    """Attribute MSA result (immutable): within, between, and vs-reference tables."""

    within: dict                       # appraiser -> {pct, pct_ci, kappa, band, n}
    between: dict                      # {pct, kappa, band, method, n}
    vs_reference: dict                 # appraiser -> {pct, pct_ci, kappa, band, n}
    n_parts: int
    n_appraisers: int
    n_trials: int
    ordinal: bool
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        kind = "ordinal" if self.ordinal else "binary"
        return (f"Attribute agreement ({kind}): {self.n_parts} parts x "
                f"{self.n_appraisers} appraisers x {self.n_trials} trials")

    def _summary_lines(self) -> list[str]:
        lines = [f"{'appraiser':<12}{'within %':>10}{'within k':>10}{'band':>14}"]
        for a, d in self.within.items():
            lines.append(f"{str(a):<12}{d['pct']:>10.1%}{d['kappa']:>10.3f}{d['band']:>14}")
        b = self.between
        lines += ["",
                  f"between appraisers ({b['method']}): {b['pct']:.1%} all-agree, "
                  f"kappa = {b['kappa']:.3f} ({b['band']})"]
        if self.vs_reference:
            lines += ["", f"{'appraiser':<12}{'vs ref %':>10}{'vs ref k':>10}{'band':>14}"]
            for a, d in self.vs_reference.items():
                lines.append(f"{str(a):<12}{d['pct']:>10.1%}{d['kappa']:>10.3f}{d['band']:>14}")
        lines += ["",
                  "kappa bands are Landis-Koch context, not the verdict; the adequacy flag is "
                  f"percent-agreement vs {_PCT_THRESHOLD:.0%}."]
        return lines

    def summary(self) -> dict:
        out: dict = {"n_parts": self.n_parts, "n_appraisers": self.n_appraisers,
                     "n_trials": self.n_trials, "ordinal": self.ordinal,
                     "between_pct": self.between["pct"], "between_kappa": self.between["kappa"]}
        for a, d in self.within.items():
            out[f"within_pct[{a}]"] = d["pct"]
            out[f"within_kappa[{a}]"] = d["kappa"]
        for a, d in self.vs_reference.items():
            out[f"ref_pct[{a}]"] = d["pct"]
            out[f"ref_kappa[{a}]"] = d["kappa"]
        return out

    def _render_standalone(self, fig, kind, **kwargs):
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        from . import palette as _pal
        pal = _pal.active()
        appr = list(self.within)
        within = [self.within[a]["pct"] for a in appr]
        x = np.arange(len(appr))
        ax.bar(x - 0.2, within, 0.4, label="within (repeatability)", color=pal.data)
        if self.vs_reference:
            ref = [self.vs_reference[a]["pct"] for a in appr]
            ax.bar(x + 0.2, ref, 0.4, label="vs reference (accuracy)", color=pal.center)
        ax.axhline(_PCT_THRESHOLD, color=pal.ooc, ls="--", lw=1, label=f"{_PCT_THRESHOLD:.0%}")
        ax.set_xticks(x); ax.set_xticklabels([str(a) for a in appr])
        ax.set_ylabel("percent agreement"); ax.set_ylim(0, 1.02)
        ax.set_title(self._title()); ax.legend(fontsize=8, loc="lower right")
        return ax


def _agreement(ratings_by_part: list[np.ndarray]) -> float:
    """Fraction of parts where every rating in the group is identical."""
    return float(np.mean([len(set(r.tolist())) == 1 for r in ratings_by_part]))


def compute(qc: QCData, rating: str, part: str, appraiser: str,
            reference=None, *, trial: str | None = None, alpha: float = 0.05,
            ordinal: bool = False) -> AttributeAgreementResult:
    """Attribute agreement analysis. See module docstring."""
    frame = qc.frame
    for col in (rating, part, appraiser):
        if col not in frame.columns:
            raise ValueError(f"column {col!r} not found in the frame.")
    weights = "linear" if ordinal else None
    cats = sorted(pd.unique(frame[rating].dropna()))

    appraisers = list(pd.unique(frame[appraiser]))
    parts = list(pd.unique(frame[part]))

    # within-appraiser: across an appraiser's own trials per part (Fleiss across trials)
    within = {}
    n_trials = 0
    for a in appraisers:
        sub = frame[frame[appraiser] == a]
        groups = [sub[sub[part] == p][rating].to_numpy() for p in parts]
        groups = [g for g in groups if g.size >= 2]
        n_trials = max(n_trials, max((g.size for g in groups), default=1))
        pct = _agreement(groups) if groups else float("nan")
        x = int(round(pct * len(groups)))
        # Fleiss across the trial replicates within each part
        counts = np.array([[np.sum(g == c) for c in cats] for g in groups], dtype=float)
        kap = fleiss_kappa(counts) if counts.shape[0] > 1 else float("nan")
        within[a] = {"pct": pct, "pct_ci": _wilson(x, len(groups), alpha),
                     "kappa": kap, "band": band(kap), "n": len(groups)}

    # each appraiser's per-part consensus rating (mode of their trials)
    def consensus(a):
        sub = frame[frame[appraiser] == a]
        return {p: stats.mode(sub[sub[part] == p][rating].to_numpy(), keepdims=False).mode
                for p in parts}
    cons = {a: consensus(a) for a in appraisers}

    # between-appraiser: all appraisers agree on a part; Fleiss/Cohen on consensus
    between_groups = [np.array([cons[a][p] for a in appraisers]) for p in parts]
    between_pct = _agreement(between_groups)
    if len(appraisers) == 2:
        r1 = np.array([cons[appraisers[0]][p] for p in parts])
        r2 = np.array([cons[appraisers[1]][p] for p in parts])
        between_k = cohen_kappa(r1, r2, categories=cats, weights=weights)
        method = "Cohen"
    else:
        counts = np.array([[np.sum(g == c) for c in cats] for g in between_groups], dtype=float)
        between_k = fleiss_kappa(counts)
        method = "Fleiss"
    between = {"pct": between_pct, "kappa": between_k, "band": band(between_k),
               "method": method, "n": len(parts)}

    # vs reference (accuracy)
    vs_reference = {}
    ref_map = None
    if reference is not None:
        if isinstance(reference, dict):
            ref_map = reference
        elif reference in frame.columns:
            ref_map = {p: frame[frame[part] == p][reference].iloc[0] for p in parts}
        else:
            raise ValueError(f"reference {reference!r} is not a column or a {{part: ref}} map.")
        ref_vec = np.array([ref_map[p] for p in parts])
        for a in appraisers:
            av = np.array([cons[a][p] for p in parts])
            pct = float(np.mean(av == ref_vec))
            x = int(round(pct * len(parts)))
            kap = cohen_kappa(av, ref_vec, categories=cats, weights=weights)
            vs_reference[a] = {"pct": pct, "pct_ci": _wilson(x, len(parts), alpha),
                               "kappa": kap, "band": band(kap), "n": len(parts)}

    checks = _adequacy(within, between, vs_reference, frame, rating, appraiser)
    step = Step(operation="attribute_agreement",
                params={"rating": rating, "part": part, "appraiser": appraiser,
                        "reference": reference if isinstance(reference, str) else bool(reference),
                        "between_kappa": between_k}, n_affected=len(frame), timestamp=_now())
    return AttributeAgreementResult(
        within=within, between=between, vs_reference=vs_reference,
        n_parts=len(parts), n_appraisers=len(appraisers), n_trials=n_trials,
        ordinal=ordinal, assumptions=checks, history=qc.history + (step,))


def _adequacy(within, between, vs_reference, frame, rating, appraiser) -> list:
    checks = []
    # binary adequacy: percent agreement vs threshold (kappa is context, not the verdict)
    passed = bool(between["pct"] >= _PCT_THRESHOLD)
    rec = None if passed else (
        f"Between-appraiser agreement {between['pct']:.1%} is below {_PCT_THRESHOLD:.0%}; "
        "the rating system is not reproducible enough - retrain or refine the operational "
        "definition before trusting the attribute gauge.")
    checks.append(AssumptionCheck("agreement", "percent agreement vs threshold",
                                  float(between["pct"]), None, passed, float(between["pct"]),
                                  "percent agreement", "ok", between["n"], rec))
    # marginal-skew / kappa-paradox flag: high agreement but a deflated kappa
    skew = _marginal_skew(frame[rating].to_numpy(), frame[rating].to_numpy())
    paradox = bool(between["pct"] >= _PCT_THRESHOLD and between["kappa"] < 0.6 and skew > 0.85)
    krec = None if not paradox else (
        f"Kappa is deflated by skewed marginals (one category is {skew:.0%} of ratings): "
        f"agreement is high ({between['pct']:.1%}) but kappa reads {between['kappa']:.2f}. "
        "This is the kappa paradox - judge the system on agreement here, not kappa alone.")
    checks.append(AssumptionCheck("kappa_marginal_skew", "category prevalence", float(skew), None,
                                  not paradox, float(skew), "max prevalence", "ok",
                                  between["n"], krec))
    return checks
