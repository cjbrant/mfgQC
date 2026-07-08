"""mfgqc.bayes - Bayesian SPC, capability, and monitoring for manufacturing.

Deterministic conjugate core (closed-form or grid-exact); seeded Monte Carlo only
for pushing draws through nonlinear functions, with the seed recorded in the
provenance chain. No dependency beyond numpy/scipy.

Public surfaces:
  fit_normal, capability_from_values   - Normal-Inverse-chi2 fit and capability
  proportion_capability, rate_capability - Beta-Binomial / Gamma-Poisson attributes
  compare                              - two-analysis posterior comparison
  assurance                            - predictive sample-size sizing
  NormalPrior / BetaPrior / GammaPrior - typed conjugate priors and elicitation

Oracles: BDA3 (Gelman et al., 3rd ed.), Hoff, Colosimo & del Castillo.
"""
from __future__ import annotations

from ._results import BayesNormalResult, fit_normal
from .attributes import (
    BayesProportionResult,
    BayesRateResult,
    proportion_capability,
    rate_capability,
)
from .capability import BayesCapabilityResult, capability_from_values
from .censored import (
    BayesCensoredCapabilityResult,
    Censoring,
    capability_censored,
)
from .comparison import ComparisonResult, compare
from .decisions import AssuranceResult, GuardbandResult, assurance, guardband
from .grid import GridPosterior
from .pooled import (
    HierarchicalResult,
    PooledCapabilityResult,
    hierarchical_normal,
    pooled_capability,
)
from .monitoring import (
    FrozenReference,
    MonitorResult,
    PredictiveCheckResult,
    monitor,
    phase1,
    predictive_check,
)
from .priors import BetaPrior, GammaPrior, NormalPrior
from .shortrun import ShortRunResult, shortrun

__all__ = [
    "fit_normal",
    "BayesNormalResult",
    "capability_from_values",
    "BayesCapabilityResult",
    "capability_censored",
    "BayesCensoredCapabilityResult",
    "Censoring",
    "GridPosterior",
    "proportion_capability",
    "BayesProportionResult",
    "rate_capability",
    "BayesRateResult",
    "compare",
    "ComparisonResult",
    "assurance",
    "AssuranceResult",
    "guardband",
    "GuardbandResult",
    "pooled_capability",
    "PooledCapabilityResult",
    "hierarchical_normal",
    "HierarchicalResult",
    "phase1",
    "monitor",
    "FrozenReference",
    "MonitorResult",
    "predictive_check",
    "PredictiveCheckResult",
    "shortrun",
    "ShortRunResult",
    "NormalPrior",
    "BetaPrior",
    "GammaPrior",
]
