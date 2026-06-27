"""Sample size and power planning (pre-data), parallel to ``mfgqc.design``.

Each solver fixes ``alpha`` and the operating point of a test and solves for
whichever of ``effect`` / ``n`` / ``power`` is left ``None``::

    import mfgqc
    plan = mfgqc.power.t_test(effect=0.5, power=0.80)      # solve for n per group
    print(plan.report())
    plan.view(kind="power_curve")

Built on the noncentral t and F distributions (and the normal approximation for
proportions); scipy only, no new dependency.
"""

from __future__ import annotations

from .solve import PowerResult, anova, proportion, t_test, variance

__all__ = ["PowerResult", "t_test", "anova", "proportion", "variance"]
