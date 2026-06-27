"""Chart palette and canvas theme for mfgQC's plots.

mfgQC's CRT-phosphor theme: a dark, near-black canvas with bright phosphor data
ink and a sparse status palette (success / warning / danger). The default preset
is ``phosphor`` (blue); ``green`` and ``amber`` presets and a classic ``light``
theme are also available via :func:`set_theme`. The docs site at
mfgqc.brantnersolutions.com uses the same phosphor palette so the published
charts read as one piece with the page around them.

The theme is applied only to mfgQC's own figures (via an ``rc_context`` around the
chart-creation paths in ``view()``), so importing mfgqc does not change a user's
global matplotlib settings.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    """A chart palette: canvas tokens, semantic data roles, and a series cycle."""

    name: str
    bg: str          # figure / axes background
    panel: str       # legend / inset background
    axis: str        # spines
    grid: str        # gridlines
    text: str        # tick labels, axis labels, annotations
    title: str       # axes titles
    muted: str       # secondary annotation
    data: str        # primary data ink (bars, histograms)
    center: str      # center line, fitted curve, plotted points
    target: str      # target / on-spec reference
    limit: str       # control limits (UCL / LCL)
    ooc: str         # out-of-control points, spec violations, FAIL
    amber: str       # warning / secondary series
    accent2: str     # tertiary
    cycle: tuple     # color cycle for multi-series plots

    def rc(self) -> dict:
        """matplotlib rcParams realizing this palette on a dark canvas."""
        from cycler import cycler
        return {
            "figure.facecolor": self.bg, "figure.edgecolor": self.bg,
            "savefig.facecolor": self.bg, "savefig.edgecolor": self.bg,
            "axes.facecolor": self.bg, "axes.edgecolor": self.axis,
            "axes.labelcolor": self.text, "axes.titlecolor": self.title,
            "axes.prop_cycle": cycler(color=list(self.cycle)),
            "text.color": self.text, "xtick.color": self.text, "ytick.color": self.text,
            "grid.color": self.grid, "grid.alpha": 0.5,
            "legend.facecolor": self.panel, "legend.edgecolor": self.axis,
            "legend.labelcolor": self.text, "legend.framealpha": 0.85,
            "boxplot.boxprops.color": self.center, "boxplot.whiskerprops.color": self.text,
            "boxplot.capprops.color": self.text, "boxplot.medianprops.color": self.target,
            "boxplot.flierprops.markeredgecolor": self.ooc,
        }


# Shared status colors (success / warning / danger) per the brand presets.
_GREEN, _AMBER, _RED, _PURPLE = "#40CC70", "#CC9A40", "#CC3355", "#9B59B6"

PHOSPHOR = Palette(
    name="phosphor", bg="#06090F", panel="#0C1018", axis="#143A52", grid="#0C1E30",
    text="#88CCE8", title="#2ECFFF", muted="#5C8AA8",
    data="#88CCE8", center="#2ECFFF", target=_GREEN, limit=_AMBER, ooc=_RED,
    amber=_AMBER, accent2=_PURPLE,
    cycle=("#2ECFFF", "#40CC70", "#CC9A40", "#88CCE8", "#CC3355", "#9B59B6"),
)

PHOSPHOR_GREEN = Palette(
    name="green", bg="#06090F", panel="#0C1018", axis="#13402A", grid="#0C2E1E",
    text="#80D8A0", title="#1BFF80", muted="#5C9878",
    data="#80D8A0", center="#1BFF80", target="#2ECFFF", limit=_AMBER, ooc=_RED,
    amber=_AMBER, accent2=_PURPLE,
    cycle=("#1BFF80", "#2ECFFF", "#CC9A40", "#80D8A0", "#CC3355", "#9B59B6"),
)

PHOSPHOR_AMBER = Palette(
    name="amber", bg="#0B0700", panel="#120D02", axis="#3A2A10", grid="#241A06",
    text="#D8B070", title="#FFB641", muted="#9A7C50",
    data="#D8B070", center="#FFB641", target=_GREEN, limit="#2ECFFF", ooc=_RED,
    amber="#FFB641", accent2=_PURPLE,
    cycle=("#FFB641", "#40CC70", "#2ECFFF", "#D8B070", "#CC3355", "#9B59B6"),
)

LIGHT = Palette(
    name="light", bg="#ffffff", panel="#f5f7fa", axis="#444444", grid="#dddddd",
    text="#222222", title="#111111", muted="#888888",
    data="#9ecae1", center="#1f77b4", target="#2ca02c", limit="#d62728", ooc="#d62728",
    amber="#fdae6b", accent2="#9467bd",
    cycle=("#1f77b4", "#2ca02c", "#ff7f0e", "#9ecae1", "#d62728", "#9467bd"),
)

_PRESETS = {p.name: p for p in (PHOSPHOR, PHOSPHOR_GREEN, PHOSPHOR_AMBER, LIGHT)}
_active = PHOSPHOR


def set_theme(name: str) -> Palette:
    """Select the chart theme: ``phosphor`` (default), ``green``, ``amber``, ``light``."""
    global _active
    if name not in _PRESETS:
        raise ValueError(f"theme must be one of {sorted(_PRESETS)}; got {name!r}.")
    _active = _PRESETS[name]
    return _active


def active() -> Palette:
    """The palette currently in effect."""
    return _active


def rc_context():
    """An ``rc_context`` applying the active palette (for mfgQC's chart paths)."""
    import matplotlib.pyplot as plt
    return plt.rc_context(_active.rc())


import contextlib as _contextlib


@_contextlib.contextmanager
def using(palette):
    """Temporarily make ``palette`` (a Palette or preset name) the active palette.

    Lets one module render with a fixed theme (e.g. the DOE views default to
    ``light``) without changing the package-wide default for everything else."""
    global _active
    prev = _active
    _active = palette if isinstance(palette, Palette) else _PRESETS[palette]
    try:
        yield _active
    finally:
        _active = prev
