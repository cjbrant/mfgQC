"""Maintainability and availability: MTTR and the inherent / achieved /
operational availability indices."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .._result import QCResult
from ..data import Step

_KINDS = ("inherent", "achieved", "operational")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, repr=False)
class AvailabilityResult(QCResult):
    """Availability index (immutable)."""

    kind: str
    availability: float
    inputs: dict
    formula: str
    assumptions: list = field(default_factory=list)
    history: tuple[Step, ...] = field(default_factory=tuple)

    def _title(self) -> str:
        return f"Availability ({self.kind})"

    def _summary_lines(self) -> list[str]:
        lines = [f"{self.kind} availability = {self.availability:.6g} "
                 f"({self.availability:.4%})",
                 f"formula: {self.formula}", "", "inputs:"]
        for k, v in self.inputs.items():
            lines.append(f"  {k} = {v:g}")
        return lines

    def summary(self) -> dict:
        return {"kind": self.kind, "availability": self.availability, **self.inputs}

    def _render_standalone(self, fig, kind, **kwargs):
        ax = fig.add_subplot(111)
        ax.barh([self.kind], [self.availability])
        ax.set_xlim(0, 1); ax.set_title(self._title())

    def _render_axes(self, ax, kind, **kwargs):
        ax.barh([self.kind], [self.availability]); ax.set_xlim(0, 1)
        return ax


def availability(mtbf: float, mttr: float, kind: str = "inherent", *,
                 pm_time: float = 0.0, pm_freq: float = 0.0,
                 logistics_delay: float = 0.0) -> AvailabilityResult:
    """Availability index.

    - inherent A_i = MTBF / (MTBF + MTTR): corrective repair only.
    - achieved A_a: adds preventive maintenance (``pm_time`` per action, ``pm_freq``
      actions per unit time).
    - operational A_o: adds mean logistics delay time (``logistics_delay``).
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}; got {kind!r}.")
    if mtbf <= 0 or mttr < 0:
        raise ValueError("mtbf must be positive and mttr non-negative.")
    inputs = {"mtbf": mtbf, "mttr": mttr}
    if kind == "inherent":
        A = mtbf / (mtbf + mttr)
        formula = "MTBF / (MTBF + MTTR)"
    elif kind == "achieved":
        # mean time between maintenance (corrective + preventive) and mean maintenance time
        cm_rate = 1.0 / mtbf
        mtbm = 1.0 / (cm_rate + pm_freq) if (cm_rate + pm_freq) > 0 else mtbf
        m_bar = (cm_rate * mttr + pm_freq * pm_time) / (cm_rate + pm_freq)
        A = mtbm / (mtbm + m_bar)
        inputs.update({"pm_time": pm_time, "pm_freq": pm_freq})
        formula = "MTBM / (MTBM + Mbar)  (corrective + preventive)"
    else:  # operational
        A = mtbf / (mtbf + mttr + logistics_delay)
        inputs.update({"logistics_delay": logistics_delay})
        formula = "MTBF / (MTBF + MTTR + mean logistics delay)"
    step = Step(operation="availability", params={"kind": kind, "availability": float(A)},
                n_affected=None, timestamp=_now())
    return AvailabilityResult(kind=kind, availability=float(A), inputs=inputs,
                             formula=formula, assumptions=[], history=(step,))
