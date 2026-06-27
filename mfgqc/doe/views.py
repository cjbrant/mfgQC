"""DOE plots: effect Pareto, normal / half-normal / Lenth effect plots, main
effects, interaction plots, residual panel, and the design layout / alias map.

All static plots are matplotlib and themed through the package palette. The
interaction-surface view is an optional plotly figure (ported from the
ixsurface concept); it falls back to a matplotlib 2D interaction plot when
plotly is unavailable.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .. import palette as _pal


def _abs_sorted(result):
    labels = list(result.terms)
    eff = np.array([result.effect[l] for l in labels])
    order = np.argsort(np.abs(eff))
    return [labels[i] for i in order], eff[order]


# --------------------------------------------------------------------------- #
# Effect plots
# --------------------------------------------------------------------------- #
def pareto(result, ax):
    labels = list(result.terms)
    eff = np.array([abs(result.effect[l]) for l in labels])
    order = np.argsort(eff)
    pal = _pal.active()
    ax.barh([labels[i] for i in order], eff[order], color=pal.data)
    if result.lenth is not None:
        ax.axvline(result.lenth.me, color=pal.amber, ls="--", lw=1, label="ME")
        ax.axvline(result.lenth.sme, color=pal.ooc, ls="--", lw=1, label="SME")
        ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("|effect|")
    ax.set_title("Effect Pareto", fontsize=10)
    return ax


def normal_plot(result, ax):
    labels = list(result.terms)
    eff = np.array([result.effect[l] for l in labels])
    order = np.argsort(eff)
    se = eff[order]
    m = se.size
    q = stats.norm.ppf((np.arange(1, m + 1) - 0.5) / m)
    pal = _pal.active()
    ax.scatter(se, q, s=20, color=pal.data)
    for i, idx in enumerate(order):
        lab = labels[idx]
        if result.lenth is None or result.lenth.labels[lab] != "inactive":
            ax.annotate(lab, (se[i], q[i]), fontsize=7, xytext=(3, 0),
                        textcoords="offset points", color=pal.center)
    ax.axhline(0, color=pal.limit, lw=0.6)
    ax.axvline(0, color=pal.limit, lw=0.6)
    ax.set_xlabel("effect")
    ax.set_ylabel("normal quantile")
    ax.set_title("Normal plot of effects", fontsize=10)
    return ax


def halfnormal_plot(result, ax):
    labels, eff = _abs_sorted(result)
    abs_e = np.abs(eff)
    m = abs_e.size
    q = stats.norm.ppf(0.5 + 0.5 * (np.arange(1, m + 1) - 0.5) / m)
    pal = _pal.active()
    ax.scatter(q, abs_e, s=20, color=pal.data)
    for i, lab in enumerate(labels):
        active = (lab in result.active) or (lab in result.possibly_active) or (lab in result.significant)
        if active:
            ax.annotate(lab, (q[i], abs_e[i]), fontsize=7, xytext=(3, 0),
                        textcoords="offset points", color=pal.center)
    if result.lenth is not None:
        ax.axhline(result.lenth.me, color=pal.amber, ls="--", lw=1, label="ME")
        ax.axhline(result.lenth.sme, color=pal.ooc, ls="--", lw=1, label="SME")
        ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("half-normal quantile")
    ax.set_ylabel("|effect|")
    ax.set_title("Half-normal plot of effects", fontsize=10)
    return ax


def lenth_plot(result, ax):
    labels = list(result.terms)
    eff = np.array([result.effect[l] for l in labels])
    pal = _pal.active()
    ax.vlines(range(len(labels)), 0, eff, color=pal.data)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    if result.lenth is not None:
        for lvl, c, lab in ((result.lenth.me, pal.amber, "ME"), (result.lenth.sme, pal.ooc, "SME")):
            ax.axhline(lvl, color=c, ls="--", lw=1)
            ax.axhline(-lvl, color=c, ls="--", lw=1)
    ax.axhline(0, color=pal.limit, lw=0.6)
    ax.set_ylabel("effect")
    ax.set_title("Lenth plot", fontsize=10)
    return ax


def main_effects(result, ax):
    pal = _pal.active()
    mains = [t for t in result.terms if ":" not in t]
    grand = result.intercept
    for f in mains:
        e = result.effect[f]
        ax.plot([-1, 1], [grand - e / 2, grand + e / 2], marker="o", label=f)
    ax.axhline(grand, color=pal.limit, lw=0.6, ls=":")
    ax.set_xlabel("coded factor level")
    ax.set_ylabel(f"mean {result.response}")
    ax.set_title("Main effects", fontsize=10)
    ax.legend(fontsize=8)
    return ax


def interaction_plot(result, ax, pair=None):
    pal = _pal.active()
    twofi = [t for t in result.terms if t.count(":") == 1]
    if pair is None:
        if not twofi:
            return main_effects(result, ax)
        pair = max(twofi, key=lambda t: abs(result.effect[t]))
    a, b = pair.split(":")
    grand = result.intercept
    ea, eb, eab = result.effect[a], result.effect[b], result.effect.get(pair, 0.0)
    for lvl_b, style in ((-1, "-o"), (1, "--s")):
        ys = [grand + ea / 2 * la + eb / 2 * lvl_b + eab / 2 * la * lvl_b for la in (-1, 1)]
        ax.plot([-1, 1], ys, style, label=f"{b}={lvl_b:+d}",
                color=(pal.data if lvl_b == -1 else pal.center))
    ax.set_xlabel(f"{a} (coded)")
    ax.set_ylabel(f"mean {result.response}")
    ax.set_title(f"Interaction {a}:{b}", fontsize=10)
    ax.legend(fontsize=8, title=b)
    return ax


def _fitted_resid(result):
    """Fitted values and residuals on the reduced model, from the stored coded
    matrix, coefficients and response."""
    mat = result._design_matrix
    fitted = np.full(mat.shape[0], result.intercept)
    for t, lbl in zip(result._term_words, result.terms):
        col = np.ones(mat.shape[0])
        for f in t:
            col = col * mat[:, result.factors.index(f)]
        fitted = fitted + result.coef[lbl] * col
    resid = result._y - fitted
    return fitted, resid


def residuals_panel(result, fig):
    pal = _pal.active()
    # Saturated model (zero residual df): residuals are identically ~0, so a
    # residual plot is meaningless. State why rather than draw an empty panel.
    if result.df_resid <= 0:
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5,
                "No residual degrees of freedom (saturated model).\n"
                "Reduce the model to active effects to plot residuals.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color=pal.text)
        return ax

    fitted, resid = _fitted_resid(result)
    axes = fig.subplots(1, 2)
    # 1. residuals vs fitted
    axes[0].scatter(fitted, resid, s=18, alpha=0.85, color=pal.data, edgecolor=pal.center, linewidth=0.4)
    axes[0].axhline(0.0, color=pal.ooc, lw=1, ls="--")
    axes[0].set_xlabel("fitted"); axes[0].set_ylabel("residual")
    axes[0].set_title("Residuals vs fitted", fontsize=10)
    # 2. normal QQ of residuals
    (osm, osr), (slope, inter, _r) = stats.probplot(resid, dist="norm")
    axes[1].scatter(osm, osr, s=18, alpha=0.85, color=pal.data, edgecolor=pal.center, linewidth=0.4)
    axes[1].plot(osm, slope * osm + inter, color=pal.center, lw=1.5)
    axes[1].set_xlabel("theoretical quantiles"); axes[1].set_ylabel("ordered residuals")
    axes[1].set_title("Normal QQ of residuals", fontsize=10)
    return axes


def interaction_surface_panel(result, fig, pair=None):
    """The fitted interaction surface over two focal factors (the bilinear twist
    for a two-level design), with the crossing verdict. Other factors are held at
    the coded center (0). Parallel surface = no interaction; a twist = interaction.
    """
    pal = _pal.active()
    twofi = [t for t in result.terms if t.count(":") == 1]
    if not twofi:
        ax = fig.add_subplot(111); ax.axis("off")
        ax.text(0.5, 0.5, "No two-factor interactions in the model.",
                ha="center", va="center", transform=ax.transAxes, color=pal.text)
        return ax
    if pair is None:
        pair = max(twofi, key=lambda t: abs(result.effect[t]))
    a, b = pair.split(":")
    b0 = result.intercept
    ea, eb, eab = result.effect[a], result.effect[b], result.effect.get(pair, 0.0)
    grid = np.linspace(-1, 1, 21)
    X, Y = np.meshgrid(grid, grid)
    # fitted response over the focal pair, other factors at coded 0:
    Z = b0 + ea / 2 * X + eb / 2 * Y + eab / 2 * X * Y
    crossing = bool(result.crossings().get(pair, False))

    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9, linewidth=0, antialiased=True)
    ax.set_xlabel(a); ax.set_ylabel(b); ax.set_zlabel(result.response)
    verdict = "interaction (surface twists)" if crossing else "no interaction (near-planar)"
    ax.set_title(f"Interaction surface {a} x {b}: {verdict}", fontsize=10)
    return ax


def alias_map(result, ax):
    pal = _pal.active()
    labels = list(result.terms)
    mat = result._design_matrix
    words = result._term_words
    cols = []
    for t in words:
        c = np.ones(mat.shape[0])
        for f in t:
            c = c * mat[:, result.factors.index(f)]
        cols.append(c)
    cols = np.column_stack(cols)
    corr = np.corrcoef(cols, rowvar=False) if cols.shape[1] > 1 else np.array([[1.0]])
    im = ax.imshow(np.abs(corr), vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Correlation map (|corr| of model columns)", fontsize=10)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def doe_axes(result, ax, kind, **kwargs):
    if kind is None:
        kind = "halfnormal" if result.fit_kind == "saturated" else "main_effects"
    fn = {
        "pareto": pareto, "normal": normal_plot, "halfnormal": halfnormal_plot,
        "lenth": lenth_plot, "main_effects": main_effects, "interaction": interaction_plot,
        "alias_map": alias_map,
    }.get(kind)
    if fn is None:
        raise ValueError(f"unknown DOE view kind={kind!r}.")
    if kind == "interaction":
        return fn(result, ax, pair=kwargs.get("pair"))
    return fn(result, ax)


def doe_standalone(result, fig, kind, **kwargs):
    if kind == "residuals":
        residuals_panel(result, fig)
        fig.suptitle(result._title(), fontsize=11)
        return
    if kind == "interaction_surface":
        interaction_surface_panel(result, fig, pair=kwargs.get("pair"))
        return
    ax = fig.add_subplot(111)
    doe_axes(result, ax, kind, **kwargs)


def design_view(design, kind="layout", **kwargs):
    import matplotlib.pyplot as plt
    with _pal.using(_pal.LIGHT), _pal.rc_context():
        fig = plt.figure(figsize=kwargs.pop("figsize", (8, 6)))
        ax = fig.add_subplot(111)
        pal = _pal.active()
        if kind == "alias_map" and design.kind == "fractional":
            ax.axis("off")
            ax.text(0.02, 0.98, "Alias structure\n" + "\n".join(design.aliases),
                    va="top", ha="left", family="monospace", fontsize=9, color=pal.data,
                    transform=ax.transAxes)
        else:
            im = ax.imshow(design.matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(design.k)); ax.set_xticklabels(design.factors)
            ax.set_ylabel("standard-order run")
            ax.set_title(f"Design layout ({design.kind})")
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        plt.close(fig)        # one figure, one render (see _result.QCResult.view)
        return fig
