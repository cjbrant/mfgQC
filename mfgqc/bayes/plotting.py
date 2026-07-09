"""Canonical charts for the Bayesian results (posterior densities, credible
intervals, assurance and control panels).

Each function takes a matplotlib Axes (or Figure for multi-panel) and reads only
fields the frozen result already stores, so the same logic works standalone and
inside a composition layer. Result objects expose ``.view()``; the user never
needs matplotlib knowledge. Style matches mfgqc.plotting (phosphor palette, dashed
spec lines in the danger color, a rounded annotation box).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .. import palette as _pal
from .conjugate import mu_marginal, predictive, sigma2_marginal

_SQRT_2PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Shared primitives
# --------------------------------------------------------------------------- #
def _annotate(ax, text: str) -> None:
    """Top-left rounded annotation box, matching the classical capability chart."""
    p = _pal.active()
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color=p.text,
            bbox=dict(boxstyle="round", fc=p.panel, ec=p.center, alpha=0.92))


def _spec_lines(ax, spec) -> None:
    """Dashed LSL/USL/Target verticals, labelled vertically at the top of the axes."""
    p = _pal.active()
    top = ax.get_ylim()[1]
    for val, name, color in ((spec.lower, "LSL", p.ooc), (spec.upper, "USL", p.ooc),
                             (getattr(spec, "target", None), "Target", p.target)):
        if val is not None:
            ax.axvline(val, color=color, ls="--", lw=1.5)
            ax.text(val, top * 0.98, f" {name}", color=color, va="top", ha="left",
                    fontsize=8, rotation=90)


def _density(ax, dist, *, color: str, label: str | None = None,
             lo_p: float = 0.001, hi_p: float = 0.999) -> tuple:
    """Plot a frozen scipy distribution's pdf over its central mass; return (xs, ys)."""
    xlo, xhi = float(dist.ppf(lo_p)), float(dist.ppf(hi_p))
    xs = np.linspace(xlo, xhi, 300)
    ys = dist.pdf(xs)
    ax.plot(xs, ys, color=color, lw=2, label=label)
    return xs, ys


def _shade_ci(ax, dist, level: float, color: str) -> tuple:
    """Shade the equal-tailed credible interval under an analytic density."""
    lo = float(dist.ppf((1.0 - level) / 2.0))
    hi = float(dist.ppf((1.0 + level) / 2.0))
    xs = np.linspace(lo, hi, 200)
    ax.fill_between(xs, dist.pdf(xs), color=color, alpha=0.25)
    return lo, hi


def _draws_posterior(ax, arr, *, title: str, xlabel: str, level: float,
                     ref: float | None = None, ref_label: str | None = None,
                     annotate: str | None = None) -> None:
    """Histogram of Monte Carlo posterior draws with a shaded credible interval,
    a median line, and an optional reference threshold (amber dashed)."""
    p = _pal.active()
    arr = np.asarray(arr, dtype=float)
    ax.hist(arr, bins="auto", density=True, color=p.data, edgecolor=p.bg, alpha=0.9)
    lo, hi = np.quantile(arr, [(1.0 - level) / 2.0, (1.0 + level) / 2.0])
    ax.axvspan(lo, hi, color=p.center, alpha=0.18)
    ax.axvline(float(np.median(arr)), color=p.center, lw=2, label="median")
    if ref is not None:
        ax.axvline(ref, color=p.limit, ls="--", lw=1.5, label=ref_label)
    if annotate is not None:
        _annotate(ax, annotate)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def _predictive_density(ax, mu, sigma, spec) -> None:
    """Posterior-predictive density as the mixture mean_i N(x; mu_i, sigma_i) over
    the posterior draws, with spec lines. Draws are thinned to <=2000 components so
    the mixture stays cheap at production draw counts (deterministic thinning)."""
    p = _pal.active()
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    step = max(1, mu.size // 2000)
    mu, sigma = mu[::step], sigma[::step]
    xlo = float(np.min(mu - 4.0 * sigma))
    xhi = float(np.max(mu + 4.0 * sigma))
    if spec.lower is not None:
        xlo = min(xlo, spec.lower)
    if spec.upper is not None:
        xhi = max(xhi, spec.upper)
    xs = np.linspace(xlo, xhi, 300)
    g = np.exp(-0.5 * ((xs[:, None] - mu[None, :]) / sigma[None, :]) ** 2)
    dens = (g / (sigma[None, :] * _SQRT_2PI)).mean(axis=1)
    ax.plot(xs, dens, color=p.center, lw=2, label="posterior predictive")
    ax.set_ylim(0.0, float(dens.max()) * 1.15)
    _spec_lines(ax, spec)
    ax.set_xlabel("measure")
    ax.set_ylabel("density")
    ax.set_title("Posterior Predictive")
    ax.legend(loc="upper right", fontsize=8)


# --------------------------------------------------------------------------- #
# Normal fit
# --------------------------------------------------------------------------- #
def normal_mu_axes(ax, result) -> None:
    """Posterior of the mean: t marginal with the credible interval shaded."""
    p = _pal.active()
    dist = mu_marginal(result.mun, result.kn, result.nun, result.sn2)
    _density(ax, dist, color=p.center, label="posterior")
    lo, hi = _shade_ci(ax, dist, result.cred_level, p.center)
    ax.axvline(result.mun, color=p.target, lw=1.5, label="posterior mean")
    conf = round(result.cred_level * 100)
    _annotate(ax, f"mu = {result.mun:.5g}\n{conf}% CI ({lo:.5g}, {hi:.5g})")
    ax.set_xlabel("mu")
    ax.set_ylabel("density")
    ax.set_title("Posterior of the Mean")
    ax.legend(loc="upper right", fontsize=8)


def normal_sigma_axes(ax, result) -> None:
    """Posterior of the standard deviation, transformed from the Inv-chi2 variance."""
    p = _pal.active()
    var = sigma2_marginal(result.nun, result.sn2)
    slo = math.sqrt(float(var.ppf(0.001)))
    shi = math.sqrt(float(var.ppf(0.999)))
    xs = np.linspace(slo, shi, 300)
    ys = var.pdf(xs ** 2) * 2.0 * xs           # density of sigma = sqrt(variance)
    ax.plot(xs, ys, color=p.center, lw=2, label="posterior")
    lo = math.sqrt(float(var.ppf((1.0 - result.cred_level) / 2.0)))
    hi = math.sqrt(float(var.ppf((1.0 + result.cred_level) / 2.0)))
    mask = (xs >= lo) & (xs <= hi)
    ax.fill_between(xs[mask], ys[mask], color=p.center, alpha=0.25)
    conf = round(result.cred_level * 100)
    _annotate(ax, f"sigma {conf}% CI\n({lo:.4g}, {hi:.4g})")
    ax.set_xlabel("sigma")
    ax.set_ylabel("density")
    ax.set_title("Posterior of the SD")
    ax.legend(loc="upper right", fontsize=8)


def normal_panels(fig, result) -> None:
    ax_mu, ax_sd = fig.subplots(1, 2)
    normal_mu_axes(ax_mu, result)
    normal_sigma_axes(ax_sd, result)


# --------------------------------------------------------------------------- #
# Capability (draws-based: continuous and censored)
# --------------------------------------------------------------------------- #
def capability_ppk_axes(ax, result) -> None:
    """Posterior of Ppk with the 1.33 capability line and P(Ppk >= 1.33)."""
    p133, mcse = result.prob("ppk", 1.33)
    conf = round(result.cred_level * 100)
    lo, hi = result.interval("ppk")
    _draws_posterior(
        ax, result._draws["ppk"], title="Posterior of Ppk", xlabel="Ppk",
        level=result.cred_level, ref=1.33, ref_label="Ppk = 1.33",
        annotate=(f"P(Ppk >= 1.33) = {p133:.3g}\n{conf}% CI ({lo:.3g}, {hi:.3g})"))


def capability_mu_axes(ax, result) -> None:
    """Posterior draws of the mean."""
    _draws_posterior(ax, result._draws["mu"], title="Posterior of mu",
                     xlabel="mu", level=result.cred_level)


def capability_ppm_axes(ax, result) -> None:
    """Posterior draws of parts-per-million nonconforming."""
    _draws_posterior(ax, result._draws["ppm"], title="Posterior of ppm",
                     xlabel="ppm nonconforming", level=result.cred_level)


def capability_predictive_axes(ax, result) -> None:
    """Posterior-predictive density with spec lines."""
    _predictive_density(ax, result._draws["mu"], result._draws["sigma"], result.spec)


def capability_panels(fig, result) -> None:
    """Predictive with spec lines, plus posterior draws of Ppk, mu and ppm."""
    axes = fig.subplots(2, 2)
    capability_predictive_axes(axes[0, 0], result)
    capability_ppk_axes(axes[0, 1], result)
    capability_mu_axes(axes[1, 0], result)
    capability_ppm_axes(axes[1, 1], result)


# --------------------------------------------------------------------------- #
# Attributes (analytic Beta / Gamma)
# --------------------------------------------------------------------------- #
def _attribute_axes(ax, result, *, dist, spec_val, spec_label, xlabel, title) -> None:
    p = _pal.active()
    _density(ax, dist, color=p.center, label="posterior")
    lo, hi = _shade_ci(ax, dist, result.cred_level, p.center)
    ax.axvline(result.mean, color=p.target, lw=1.5, label="posterior mean")
    conf = round(result.cred_level * 100)
    note = f"mean = {result.mean:.4g}\n{conf}% CI ({lo:.4g}, {hi:.4g})"
    if spec_val is not None:
        ax.axvline(spec_val, color=p.ooc, ls="--", lw=1.5, label=spec_label)
        within = result.prob_within_spec()
        note += f"\nP(<= {spec_val:.3g}) = {within:.3g}"
    _annotate(ax, note)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def proportion_axes(ax, result) -> None:
    _attribute_axes(ax, result, dist=stats.beta(result.a_post, result.b_post),
                    spec_val=result.max_proportion, spec_label="max proportion",
                    xlabel="defect proportion", title="Posterior of the Proportion")


def rate_axes(ax, result) -> None:
    _attribute_axes(ax, result,
                    dist=stats.gamma(result.a_post, scale=1.0 / result.b_post),
                    spec_val=result.max_rate, spec_label="max rate",
                    xlabel="count rate", title="Posterior of the Rate")


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def comparison_prob_axes(ax, result) -> None:
    """Horizontal bars of P(B better than A) for mean, sd and Ppk."""
    p = _pal.active()
    labels = ["mean larger", "sd smaller", "Ppk larger"]
    vals = [result.prob_mean_gt, result.prob_sd_lt, result.prob_ppk_gt]
    y = np.arange(len(vals))
    colors = [p.target if v >= 0.5 else p.amber for v in vals]
    ax.barh(y, vals, color=colors)
    ax.axvline(0.5, color=p.muted, ls="--", lw=1)
    for yi, v in zip(y, vals):
        ax.text(min(v + 0.02, 0.98), yi, f"{v:.3g}", va="center", ha="left",
                fontsize=8, color=p.text)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(f"P({result.label_b} better than {result.label_a})")
    ax.set_title("Comparison Probabilities")


def _delta_axes(ax, delta: tuple, *, title: str, xlabel: str) -> None:
    """A single delta with its credible interval as a horizontal whisker, zeroed."""
    p = _pal.active()
    med, lo, hi = delta
    ax.errorbar([med], [0], xerr=[[med - lo], [hi - med]], fmt="o", color=p.center,
                ecolor=p.center, capsize=5, ms=8)
    ax.axvline(0.0, color=p.ooc, ls="--", lw=1.5, label="no difference")
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def comparison_panels(fig, result) -> None:
    ax_prob, ax_dm, ax_dp = fig.subplots(3, 1)
    comparison_prob_axes(ax_prob, result)
    _delta_axes(ax_dm, result.delta_mean, title="Delta mean (B - A)", xlabel="delta mean")
    _delta_axes(ax_dp, result.delta_ppk, title="Delta Ppk (B - A)", xlabel="delta Ppk")


# --------------------------------------------------------------------------- #
# Assurance
# --------------------------------------------------------------------------- #
def assurance_axes(ax, result) -> None:
    """Assurance curve: probability of a capable verdict vs candidate sample size."""
    p = _pal.active()
    n = np.asarray(result.n_grid, dtype=float)
    a = np.asarray(result.assurance, dtype=float)
    ax.plot(n, a, marker="o", color=p.center, lw=1.5)
    ax.axhline(result.decide_hi, color=p.limit, ls="--", lw=1.5,
               label=f"decision {result.decide_hi:.2f}")
    if result.recommended_n is not None:
        ax.axvline(result.recommended_n, color=p.target, lw=1.5,
                   label=f"recommended n = {result.recommended_n}")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("sample size n")
    ax.set_ylabel(f"assurance: P(P({result.quantity} >= {result.threshold:.3g}) exceeds "
                  f"{result.decide_hi:.2f})")
    ax.set_title("Sample-Size Assurance")
    ax.legend(loc="lower right", fontsize=8)


# --------------------------------------------------------------------------- #
# Guardband
# --------------------------------------------------------------------------- #
def guardband_limits_axes(ax, result) -> None:
    """Spec limits vs cost-optimal acceptance limits on a single measure axis."""
    p = _pal.active()
    spec = result.spec
    for val, name in ((spec.lower, "LSL"), (spec.upper, "USL")):
        if val is not None:
            ax.axvline(val, color=p.ooc, ls="--", lw=1.5)
            ax.text(val, 0.9, f" {name}", color=p.ooc, rotation=90, fontsize=8, va="top")
    for val, name in ((result.a_lo, "accept lo"), (result.a_hi, "accept hi")):
        if val is not None:
            ax.axvline(val, color=p.center, lw=2)
            ax.text(val, 0.1, f" {name}", color=p.center, rotation=90, fontsize=8, va="bottom")
    ax.set_yticks([])
    ax.set_xlabel("measure")
    ax.set_title(f"Acceptance Limits (gauge sd = {result.sigma_gauge:.4g})")


def guardband_cost_axes(ax, result) -> None:
    """Expected cost of the optimal guardband vs the naive (spec = accept) policy."""
    p = _pal.active()
    vals = [result.expected_cost, result.naive_expected_cost]
    x = np.arange(2)
    ax.bar(x, vals, color=[p.target, p.amber])
    ax.set_xticks(x)
    ax.set_xticklabels(["optimal", "naive (= spec)"])
    ax.set_ylabel("expected cost")
    ax.set_title("Expected Cost")
    _annotate(ax,
              f"optimal: scrap {result.scrap_pct:.3g}%  escape {result.escape_ppm:.0f} ppm\n"
              f"naive:   scrap {result.naive_scrap_pct:.3g}%  escape {result.naive_escape_ppm:.0f} ppm")


def guardband_panels(fig, result) -> None:
    ax_lim, ax_cost = fig.subplots(2, 1)
    guardband_limits_axes(ax_lim, result)
    guardband_cost_axes(ax_cost, result)


# --------------------------------------------------------------------------- #
# Pooled (hierarchical)
# --------------------------------------------------------------------------- #
def pooled_min_cpk_axes(ax, result) -> None:
    """Posterior of the worst-position Cpk with the target line and P(all capable)."""
    p_cap, mcse = result.prob_all_capable()
    _draws_posterior(
        ax, result._min_cpk, title="Posterior of min-position Cpk",
        xlabel="min_j Cpk_j", level=result.cred_level,
        ref=result.target, ref_label=f"target = {result.target:.3g}",
        annotate=f"P(min Cpk >= {result.target:.3g}) = {p_cap:.3g} +/- {mcse:.2g}")


def pooled_positions_axes(ax, result) -> None:
    """Per-position mean posterior intervals (whiskers) with the observed means."""
    p = _pal.active()
    j = np.arange(result.n_positions)
    meds, los, his = [], [], []
    for i in range(result.n_positions):
        lo, hi = result.theta_interval(i)
        los.append(lo)
        his.append(hi)
        meds.append(float(np.median(result._theta[:, i])))
    meds = np.asarray(meds)
    ax.errorbar(meds, j, xerr=[meds - np.asarray(los), np.asarray(his) - meds],
                fmt="o", color=p.center, ecolor=p.center, capsize=4, label="posterior mean")
    ax.scatter(result.position_mean, j, color=p.amber, marker="s", s=30, zorder=3,
               label="observed mean")
    ax.set_yticks(j)
    ax.set_yticklabels([f"pos {i}" for i in j])
    ax.invert_yaxis()
    ax.set_xlabel("position mean")
    ax.set_title("Per-Position Means")
    ax.legend(loc="best", fontsize=8)


def pooled_panels(fig, result) -> None:
    ax_pos, ax_cpk = fig.subplots(1, 2)
    pooled_positions_axes(ax_pos, result)
    pooled_min_cpk_axes(ax_cpk, result)


# --------------------------------------------------------------------------- #
# Short-run chart
# --------------------------------------------------------------------------- #
def shortrun_stat_axes(ax, result) -> None:
    """Chart statistic P(|mu - target| > d) per stage with the p* limit; flags red."""
    p = _pal.active()
    x = np.arange(result.n_stages)
    stat = np.asarray(result.chart_stat, dtype=float)
    ax.plot(x, stat, marker="o", color=p.center, lw=1, ms=4, zorder=2)
    ax.axhline(result.p_star, color=p.limit, ls="--", lw=1.2, label=f"p* = {result.p_star:.2f}")
    flagged = [i for i, f in enumerate(result.flags) if f]
    if flagged:
        ax.scatter(flagged, stat[flagged], color=p.ooc, s=45, zorder=3, label="flag")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("stage")
    ax.set_ylabel("P(|mu - target| > d)")
    ax.set_title("Short-Run Chart Statistic")
    ax.legend(loc="best", fontsize=8)


def shortrun_mean_axes(ax, result) -> None:
    """Per-stage subgroup mean with the target and the +/- d indifference band."""
    p = _pal.active()
    x = np.arange(result.n_stages)
    mean = np.asarray(result.stage_mean, dtype=float)
    ax.plot(x, mean, marker="o", color=p.center, lw=1, ms=4)
    ax.axhline(result.target, color=p.target, lw=1.2, label="target")
    ax.axhline(result.target + result.d, color=p.muted, ls="--", lw=1, label="target +/- d")
    ax.axhline(result.target - result.d, color=p.muted, ls="--", lw=1)
    ax.set_xlabel("stage")
    ax.set_ylabel("subgroup mean")
    ax.set_title("Stage Means")
    ax.legend(loc="best", fontsize=8)


def shortrun_panels(fig, result) -> None:
    ax_stat, ax_mean = fig.subplots(2, 1)
    shortrun_stat_axes(ax_stat, result)
    shortrun_mean_axes(ax_mean, result)


# --------------------------------------------------------------------------- #
# Monitoring
# --------------------------------------------------------------------------- #
def monitor_axes(ax, result) -> None:
    """Bayesian p-value grid (subgroups x tests); flagged subgroups marked red."""
    p = _pal.active()
    mat = np.asarray(result.p_values_matrix(), dtype=float)
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(result.tests)))
    ax.set_xticklabels(result.tests, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(result.labels)))
    ax.set_yticklabels(result.labels, fontsize=8)
    for yi, flag in enumerate(result.flags):
        if flag:
            ax.get_yticklabels()[yi].set_color(p.ooc)
    for yi in range(mat.shape[0]):
        for xi in range(mat.shape[1]):
            ax.text(xi, yi, f"{mat[yi, xi]:.2g}", va="center", ha="center",
                    fontsize=7, color=p.bg if mat[yi, xi] > 0.5 else p.text)
    ax.set_title(f"Monitoring p-values (alpha = {result.alpha:.3g})")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Bayesian p-value")


def predictive_check_axes(ax, result) -> None:
    """The Bayesian p-value on a [0, 1] scale with the extreme tails shaded."""
    p = _pal.active()
    ax.axvspan(0.0, 0.025, color=p.ooc, alpha=0.25)
    ax.axvspan(0.975, 1.0, color=p.ooc, alpha=0.25)
    ax.axvline(result.p_bayes, color=p.center, lw=3, label="P(T_rep >= T_obs)")
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Bayesian p-value")
    ax.set_title(f"Predictive Check: T = {result.statistic}")
    _annotate(ax, f"T(y_obs) = {result.t_observed:.4g}\n"
                  f"p = {result.p_bayes:.3g}   two-sided {result.p_two_sided:.3g}")
    ax.legend(loc="lower right", fontsize=8)
