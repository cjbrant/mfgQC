"""Lenth's pseudo standard error for an unreplicated factorial (Lenth 1989).

When a design has no replication, center points, or pooled terms, the saturated
fit leaves zero residual degrees of freedom and there is no pure-error estimate
for t or F tests. Lenth's PSE estimates the noise from the effects themselves,
assuming most are inactive (effect sparsity). It is implemented directly here -
it is small and citable - and its numeric ME/SME are cross-checked against
``BsMD::LenthPlot`` as a Tier 2 secondary check.

The headline active set is the half-normal identification: an effect is reported
active when it clears the simultaneous margin of error (SME), the experiment-wise
threshold that controls the family error rate. The individual margin of error
(ME) is reported as context (the more liberal per-effect threshold); effects
between ME and SME are surfaced as "possibly active" rather than silently
dropped or promoted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class LenthResult:
    """Lenth PSE significance summary for a saturated factorial."""

    pse: float
    me: float                       # individual margin of error  (context)
    sme: float                      # simultaneous margin of error (active threshold)
    pseudo_df: float
    n_effects: int
    labels: dict                    # term -> "active" | "possibly_active" | "inactive"
    pseudo_t: dict                  # term -> effect / PSE

    @property
    def active(self) -> list:
        return [k for k, v in self.labels.items() if v == "active"]

    @property
    def possibly_active(self) -> list:
        return [k for k, v in self.labels.items() if v == "possibly_active"]


def lenth(effects: dict) -> LenthResult:
    """Lenth's PSE, ME, SME and a per-effect verdict from a dict of full effects.

    ``effects`` maps term -> effect (``2 * coefficient``); the intercept must not
    be included. The verdict is keyed to SME (active), with ME as context.
    """
    names = list(effects)
    e = np.array([effects[n] for n in names], dtype=float)
    abs_e = np.abs(e)
    m = e.size
    if m == 0:
        raise ValueError("lenth() needs at least one effect.")

    s0 = 1.5 * float(np.median(abs_e))
    kept = abs_e[abs_e < 2.5 * s0]
    pse = 1.5 * float(np.median(kept)) if kept.size else 1.5 * float(np.median(abs_e))
    d = m / 3.0

    me = float(stats.t.ppf(0.975, d) * pse)
    gamma = (1.0 + 0.975 ** (1.0 / m)) / 2.0
    sme = float(stats.t.ppf(gamma, d) * pse)

    labels: dict = {}
    pseudo_t: dict = {}
    for n in names:
        a = abs(effects[n])
        pseudo_t[n] = float(effects[n] / pse) if pse > 0 else float("nan")
        if a > sme:
            labels[n] = "active"
        elif a > me:
            labels[n] = "possibly_active"
        else:
            labels[n] = "inactive"
    return LenthResult(pse=pse, me=me, sme=sme, pseudo_df=d, n_effects=m,
                       labels=labels, pseudo_t=pseudo_t)
