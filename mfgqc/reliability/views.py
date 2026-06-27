"""Reliability plots: the probability plot (the signature view), survival,
hazard, and cdf for a fit; the Kaplan-Meier step function; system R(t)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from .. import palette as _pal


def _grid(result):
    t = result._times
    return np.linspace(max(1e-6, t.min() * 0.5), t.max() * 1.3, 200)


def life_axes(result, ax, kind):
    pal = _pal.active()
    from .life import _linearize, _plot_positions
    if kind == "survival":
        ts = _grid(result)
        ax.plot(ts, result._frozen.sf(ts), color=pal.center, lw=2, label="R(t) fit")
        ax.set_ylabel("R(t)"); ax.set_ylim(0, 1.02)
    elif kind == "cdf":
        ts = _grid(result)
        ax.plot(ts, result._frozen.cdf(ts), color=pal.center, lw=2, label="F(t) fit")
        ax.set_ylabel("F(t)"); ax.set_ylim(0, 1.02)
    elif kind == "hazard":
        ts = _grid(result)
        h = result._frozen.pdf(ts) / np.clip(result._frozen.sf(ts), 1e-300, None)
        ax.plot(ts, h, color=pal.ooc, lw=2, label="h(t) fit")
        ax.set_ylabel("hazard h(t)")
    elif kind == "probability_plot":
        return _probability_plot(result, ax)
    else:
        raise ValueError(f"unknown life view kind={kind!r}.")
    ax.set_xlabel("time"); ax.set_title(result._title()); ax.legend(fontsize=8)
    return ax


def _probability_plot(result, ax):
    pal = _pal.active()
    from .life import _linearize, _plot_positions
    t_pp, F_pp = _plot_positions(result._times, result._events)
    x, y = _linearize(result.dist, t_pp, F_pp)
    ax.scatter(x, y, s=22, color=pal.data, zorder=5, label="plotted points")
    xs = np.linspace(x.min(), x.max(), 100)
    # fitted line in the linearized coordinates
    fr = result._frozen
    tt = np.exp(xs) if result.dist in ("weibull", "lognormal") else xs
    Ffit = fr.cdf(tt)
    Ffit = np.clip(Ffit, 1e-6, 1 - 1e-6)
    _, yfit = _linearize(result.dist, tt, Ffit)
    ax.plot(xs, yfit, color=pal.center, lw=2, label=f"{result.dist} fit (PPCC={result.ppcc:.3f})")
    ax.set_xlabel("linearized time"); ax.set_ylabel("linearized probability")
    ax.set_title(f"Probability plot ({result.dist})")
    ax.legend(fontsize=8, loc="best")
    return ax


def life_view(result, fig, kind):
    ax = fig.add_subplot(111)
    life_axes(result, ax, kind)


def km_view(result, fig, kind):
    pal = _pal.active()
    ax = fig.add_subplot(111)
    t = result.times; R = result.survival
    ax.step(t, R, where="post", color=pal.center, lw=2, label="Kaplan-Meier R(t)")
    if result.lower is not None:
        ax.fill_between(t, result.lower, result.upper, step="post",
                        color=pal.center, alpha=0.18, label="95% bound")
    ax.set_xlabel("time"); ax.set_ylabel("R(t)"); ax.set_ylim(0, 1.02)
    ax.set_title(result._title()); ax.legend(fontsize=8)


def system_view(result, fig, kind):
    pal = _pal.active()
    ax = fig.add_subplot(111)
    r = np.linspace(0, 1, 100)
    ax.plot(r, result._curve(r), color=pal.center, lw=2)
    ax.plot([0, 1], [0, 1], color=pal.limit, ls="--", lw=1, label="component")
    ax.scatter([result.component_r_repr], [result.reliability], color=pal.ooc, zorder=5,
               label=f"system R={result.reliability:.4f}")
    ax.set_xlabel("component reliability"); ax.set_ylabel("system reliability")
    ax.set_title(result._title()); ax.legend(fontsize=8)
