"""Pre-control (a.k.a. stoplight control), driven by the SPEC limits, not control
limits.

The tolerance is split into a central GREEN half, two YELLOW quarters, and two
RED zones outside spec. A run qualifies on five consecutive green pieces, then a
running two-piece rule decides continue/adjust/stop.

Surfacing: pre-control assumes a capable, centered process. mfgQC estimates Cpk
from the data and flags when capability has not been established, and never
presents pre-control as a substitute for a control chart on an incapable process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from ._result import QCResult
from .assumptions import AssumptionCheck
from .data import QCData, Step

_GREEN, _YLO, _YHI, _RLO, _RHI = "green", "yellow_low", "yellow_high", "red_low", "red_high"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, repr=False)
class PrecontrolResult(QCResult):
    """Pre-control result (immutable): zone of each point and the dispositions."""

    lower: float
    upper: float
    target: float
    pc_lower: float
    pc_upper: float
    values: np.ndarray
    zones: tuple[str, ...]
    qualified: bool
    qualify_index: int | None
    dispositions: tuple
    cpk: float
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return "Pre-control (stoplight)"

    def _summary_lines(self) -> list[str]:
        from collections import Counter
        z = Counter(self.zones)
        lines = [
            f"spec: [{self.lower:g}, {self.upper:g}]   target {self.target:g}",
            f"pre-control lines: {self.pc_lower:g} | {self.pc_upper:g}   "
            f"(central half green, quarters yellow, outside spec red)",
            f"zones: green {z[_GREEN]}, yellow {z[_YLO] + z[_YHI]}, red {z[_RLO] + z[_RHI]}",
            f"qualification (5 greens in a row): "
            + (f"qualified at piece {self.qualify_index + 1}" if self.qualified else "NOT qualified"),
        ]
        if self.dispositions:
            lines.append("")
            lines.append("running-rule events:")
            for i, action, reason in self.dispositions:
                lines.append(f"  piece {i + 1}: {action} - {reason}")
        return lines

    def summary(self) -> dict:
        from collections import Counter
        z = Counter(self.zones)
        return {"lower": self.lower, "upper": self.upper, "pc_lower": self.pc_lower,
                "pc_upper": self.pc_upper, "qualified": self.qualified,
                "n_green": z[_GREEN], "n_yellow": z[_YLO] + z[_YHI],
                "n_red": z[_RLO] + z[_RHI], "n_stops": len(self.dispositions), "cpk": self.cpk}

    def _render_standalone(self, fig, kind, **kwargs):
        self._render_axes(fig.add_subplot(111), kind, **kwargs)

    def _render_axes(self, ax, kind, **kwargs):
        from . import palette as _pal
        pal = _pal.active()
        x = np.arange(self.values.size)
        cmap = {_GREEN: pal.target, _YLO: pal.amber, _YHI: pal.amber,
                _RLO: pal.ooc, _RHI: pal.ooc}
        ax.axhspan(self.pc_lower, self.pc_upper, color=pal.target, alpha=0.12)
        ax.axhspan(self.lower, self.pc_lower, color=pal.amber, alpha=0.12)
        ax.axhspan(self.pc_upper, self.upper, color=pal.amber, alpha=0.12)
        for lvl in (self.lower, self.upper):
            ax.axhline(lvl, color=pal.ooc, lw=1)
        for lvl in (self.pc_lower, self.pc_upper):
            ax.axhline(lvl, color=pal.amber, lw=1, ls="--")
        ax.scatter(x, self.values, c=[cmap[z] for z in self.zones], zorder=5)
        ax.plot(x, self.values, color=pal.data, lw=0.8, alpha=0.5)
        ax.set_xlabel("piece"); ax.set_ylabel("measurement")
        ax.set_title(self._title())
        return ax


def _zone(v, lo, hi, pclo, pchi) -> str:
    if v < lo:
        return _RLO
    if v > hi:
        return _RHI
    if v < pclo:
        return _YLO
    if v > pchi:
        return _YHI
    return _GREEN


def compute(qc: QCData, *, alpha: float = 0.05) -> PrecontrolResult:
    """Pre-control from the loaded measure against its spec limits."""
    m = qc.meta
    if m.lower is None or m.upper is None:
        from .errors import MissingPrerequisiteError
        raise MissingPrerequisiteError(
            "pre-control needs both spec limits; set them with .spec(lower=, upper=).",
            analysis="precontrol", missing=["spec"])
    lo, hi = float(m.lower), float(m.upper)
    target = float(m.target) if m.target is not None else (lo + hi) / 2.0
    tol = hi - lo
    pclo, pchi = lo + tol / 4.0, hi - tol / 4.0
    vals = qc.values()
    vals = vals[~np.isnan(vals)]
    zones = [_zone(v, lo, hi, pclo, pchi) for v in vals]

    # qualification: first run of 5 consecutive greens
    qualify_index = None
    run = 0
    for i, z in enumerate(zones):
        run = run + 1 if z == _GREEN else 0
        if run >= 5:
            qualify_index = i
            break
    qualified = qualify_index is not None

    # running two-piece rule after qualification
    dispositions = []
    start = (qualify_index + 1) if qualified else 0
    for i in range(start, len(zones) - 1):
        a, b = zones[i], zones[i + 1]
        if a.startswith("red") or b.startswith("red"):
            dispositions.append((i + 1, "STOP", "a piece is out of spec (red)"))
        elif a.startswith("yellow") and b.startswith("yellow"):
            if a == b:
                dispositions.append((i + 1, "ADJUST", "two yellows on the same side (process off-center)"))
            else:
                dispositions.append((i + 1, "STOP", "two yellows on opposite sides (excess variation)"))

    # capability prerequisite (the wedge): estimate Cpk and flag if not established
    mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
    cpk = min((hi - mu), (mu - lo)) / (3 * sd) if sd > 0 else float("inf")
    capable = cpk >= 1.33
    rec = None if capable else (
        f"Estimated Cpk = {cpk:.2f} (< 1.33): capability is not established. Pre-control is "
        "unreliable on an incapable or off-center process - run a capability study and a control "
        "chart first; pre-control is a running check, not a substitute.")
    flag = AssumptionCheck("capability_prerequisite", "estimated Cpk", float(cpk), None,
                           bool(capable), float(cpk), "Cpk", "ok", vals.size, rec)

    step = Step(operation="precontrol",
                params={"lower": lo, "upper": hi, "qualified": qualified, "cpk": cpk},
                n_affected=vals.size, timestamp=_now())
    return PrecontrolResult(
        lower=lo, upper=hi, target=target, pc_lower=pclo, pc_upper=pchi, values=vals,
        zones=tuple(zones), qualified=qualified, qualify_index=qualify_index,
        dispositions=tuple(dispositions), cpk=float(cpk), assumptions=[flag],
        history=qc.history + (step,))
