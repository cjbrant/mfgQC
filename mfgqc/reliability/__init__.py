"""Reliability and maintainability.

Life-data analysis with censoring (``qc.life_fit``, ``qc.life_table``), constant-
rate MTBF (``qc.mtbf``), system reliability composition, ISO 281 bearing life,
availability, and demonstration-test sizing::

    qc = mfgqc.load(df, measure="hours").roles(time="hours", event="failed")
    fit = qc.life_fit(dist="weibull")             # MLE with right censoring
    print(fit.report()); fit.view(kind="probability_plot")

    mfgqc.reliability.series([0.99, 0.98, 0.97])   # structural / planning
    mfgqc.reliability.bearing_life(C=26000, P=4000, rpm=1800)

Censoring is first class; the chosen distribution and the constant-rate
assumption are always surfaced, never silently applied.
"""

from __future__ import annotations

from .availability import AvailabilityResult, availability
from .demonstrate import DemonstrationResult, MTBFResult, demonstration_test, mtbf
from .life import LifeFitResult, life_fit
from .nonparametric import KaplanMeierResult, kaplan_meier
from .system import (
    BearingLifeResult,
    SystemReliabilityResult,
    bearing_life,
    k_of_n,
    parallel,
    series,
    system,
)

__all__ = [
    "life_fit", "LifeFitResult", "kaplan_meier", "KaplanMeierResult",
    "series", "parallel", "k_of_n", "system", "SystemReliabilityResult",
    "bearing_life", "BearingLifeResult",
    "availability", "AvailabilityResult", "mtbf", "MTBFResult",
    "demonstration_test", "DemonstrationResult",
]
