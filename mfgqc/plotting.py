"""Canonical, audit-ready QC charts.

Each drawing function takes a matplotlib Axes (or Figure for multi-panel) so the
same logic works standalone and inside a composition layer. The user never needs
matplotlib knowledge: result objects expose ``.view()``.
"""

from __future__ import annotations

from . import palette as _pal

import numpy as np
from scipy import stats



# --------------------------------------------------------------------------- #
# Capability
# --------------------------------------------------------------------------- #
def capability_histogram(ax, result) -> None:
    """Histogram of the measure with spec lines, fitted curve and index annotations."""
    values = np.asarray(result._values, dtype=float)
    ax.hist(values, bins="auto", density=True, color=_pal.active().data,
            edgecolor=_pal.active().bg, alpha=0.9)

    xs = np.linspace(values.min(), values.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, result.mean, result.sigma_overall),
            color=_pal.active().center, lw=2, label="fitted normal")

    spec = result.spec
    for val, name, color in ((spec.lower, "LSL", _pal.active().ooc), (spec.upper, "USL", _pal.active().ooc),
                             (spec.target, "Target", _pal.active().target)):
        if val is not None:
            ax.axvline(val, color=color, ls="--", lw=1.5)
            ax.text(val, ax.get_ylim()[1] * 0.98, f" {name}", color=color,
                    va="top", ha="left", fontsize=8, rotation=90)

    def fmt(v):
        return "n/a" if v is None else f"{v:.3f}"
    txt = (f"Cp={fmt(result.cp)}  Cpk={fmt(result.cpk)}\n"
           f"Pp={fmt(result.pp)}  Ppk={fmt(result.ppk)}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color=_pal.active().text,
            bbox=dict(boxstyle="round", fc=_pal.active().panel,
                      ec=_pal.active().center, alpha=0.92))

    crit = [a for a in result.assumptions if not a.passed]
    if crit:
        ax.text(0.98, 0.02, "assumption FAIL:\n" + crit[0].name,
                transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
                color=_pal.active().ooc)
    ax.set_xlabel("measure")
    ax.set_ylabel("density")
    ax.set_title("Process Capability")
    ax.legend(loc="upper right", fontsize=8)


def capability_probplot(ax, result) -> None:
    """Normal probability plot of the measure."""
    stats.probplot(np.asarray(result._values, dtype=float), dist="norm", plot=ax)
    ax.set_title("Normal Probability Plot")


# --------------------------------------------------------------------------- #
# Control charts
# --------------------------------------------------------------------------- #
def control_panel(ax, result, *, panel: str = "location") -> None:
    """Draw one control-chart panel (location or dispersion)."""
    if panel == "dispersion" and result.disp_points is not None:
        points = np.asarray(result.disp_points, dtype=float)
        cl, ucl, lcl = result.disp_cl, result.disp_ucl, result.disp_lcl
        label = result.disp_label
    else:
        panel = "location"
        points = np.asarray(result.location_points, dtype=float)
        cl, ucl, lcl = result.location_cl, result.location_ucl, result.location_lcl
        label = result.location_label

    x = np.arange(1, len(points) + 1)
    ax.plot(x, points, marker="o", color=_pal.active().center, lw=1, ms=4, zorder=2)
    ax.axhline(cl, color=_pal.active().target, lw=1.2, label="CL")
    ax.plot(x, np.asarray(ucl, dtype=float), color=_pal.active().limit, ls="--", lw=1, label="UCL")
    ax.plot(x, np.asarray(lcl, dtype=float), color=_pal.active().limit, ls="--", lw=1, label="LCL")

    viol_pts = {v.point for v in result.violations if v.chart == panel}
    if viol_pts:
        idx = [p - 1 for p in viol_pts if 1 <= p <= len(points)]
        ax.scatter(np.array(idx) + 1, points[idx], color=_pal.active().ooc, zorder=3, s=45,
                   label="out of control")
    ax.set_ylabel(label)
    ax.set_xlabel("subgroup")
    ax.set_title(f"{label} chart")
    ax.legend(loc="best", fontsize=8)


# --------------------------------------------------------------------------- #
# Gage R&R
# --------------------------------------------------------------------------- #
def gage_components_bar(ax, result) -> None:
    """Components-of-variation bar chart (%study var and %contribution)."""
    comps = ["EV", "AV", "GRR", "PV"]
    study = [result.pct_study.get(c, 0.0) for c in comps]
    contrib = [result.pct_contrib.get(c, 0.0) for c in comps]
    x = np.arange(len(comps))
    w = 0.38
    ax.bar(x - w / 2, study, w, label="% study var", color=_pal.active().data)
    ax.bar(x + w / 2, contrib, w, label="% contribution", color=_pal.active().amber)
    ax.set_xticks(x)
    ax.set_xticklabels(comps)
    ax.set_ylabel("percent")
    ax.set_title(f"Components of Variation  (verdict: {result.verdict})")
    ax.legend(fontsize=8)


def hypothesis_plot(ax, result) -> None:
    """Grouped boxplots (or bars for proportions) with the test result annotated."""
    groups = [np.asarray(g, dtype=float) for g in result._groups]
    labels = list(result._labels)
    if all(g.size >= 2 for g in groups):
        ax.boxplot(groups, tick_labels=labels, showmeans=True)
        ax.set_ylabel("value")
    else:  # proportions / degenerate single values
        vals = [float(g.mean()) for g in groups]
        ax.bar(range(len(vals)), vals, color=_pal.active().data)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("value")
    if result._target is not None:
        ax.axhline(result._target, color=_pal.active().ooc, ls="--", label=f"target={result._target:g}")
        ax.legend(fontsize=8)
    ax.set_title(f"{result.test_used}: p={result.p_value:.4g}")


def gage_panels(fig, result) -> None:
    """Standard MSA multi-panel: components, by-operator (Xbar & R), by-part."""
    df = result._crossed
    axes = fig.subplots(2, 2)
    gage_components_bar(axes[0, 0], result)

    # Measurement by part
    ax_part = axes[0, 1]
    if len(df):
        part_means = df.groupby("part", sort=False)["value"].mean()
        ax_part.plot(range(len(part_means)), part_means.to_numpy(), marker="o", color=_pal.active().center)
        ax_part.set_xticks(range(len(part_means)))
        ax_part.set_xticklabels([str(p) for p in part_means.index], fontsize=7, rotation=45)
    ax_part.set_title("Measurement by Part")
    ax_part.set_ylabel("value")

    # Xbar by operator
    ax_xbar = axes[1, 0]
    # R by operator
    ax_r = axes[1, 1]
    if len(df):
        for op, grp in df.groupby("operator", sort=False):
            pm = grp.groupby("part", sort=False)["value"].mean()
            ax_xbar.plot(range(len(pm)), pm.to_numpy(), marker="o", label=str(op))
            rng = grp.groupby("part", sort=False)["value"].agg(lambda s: s.max() - s.min())
            ax_r.plot(range(len(rng)), rng.to_numpy(), marker="s", label=str(op))
        ax_xbar.legend(title="operator", fontsize=7)
        ax_r.legend(title="operator", fontsize=7)
    ax_xbar.set_title("Xbar by Operator")
    ax_xbar.set_ylabel("mean")
    ax_r.set_title("R by Operator")
    ax_r.set_ylabel("range")
