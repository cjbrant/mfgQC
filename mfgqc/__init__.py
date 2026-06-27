"""mfgQC - quality-control analysis for manufacturing practitioners.

Three design pillars: statistical guardrails (every analysis checks and reports
its own assumptions, never silently switching methods), practitioner-oriented
(legible errors, no statistics/programming background assumed), and auditable by
construction (immutable data and results carrying a structured, propagating
provenance history).

Quick start (fluent idiom: verb -> object -> methods)
-----------------------------------------------------
>>> import pandas as pd, mfgqc
>>> clean_df = mfgqc.clean(raw, [mfgqc.fix_names(), mfgqc.coerce_numeric(["width"])])
>>> data = mfgqc.load(clean_df, measure="width", subgroup="batch").spec(lower=1.0, upper=2.0)
>>> mfgqc.overview(data)            # role/spec-aware diagnostic
>>> cap = data.capability()        # uses the attached spec
>>> cc = data.control_chart()
>>> grr = data.gage_rr()
"""

from __future__ import annotations

from .assumptions import AssumptionCheck
from .capability import CapabilityResult
from .control_charts import ControlChartResult, Violation
from .data import (
    Crossed,
    QCData,
    QCMeta,
    Step,
    Subgroups,
    from_wide,
    load,
)
from .gage_rr import GageRRResult
from .hypothesis import (
    HypothesisResult,
    test_anova,
    test_mean,
    test_means,
    test_paired,
    test_proportion,
    test_proportions,
    test_variance,
)
from .ingestion import (
    Overview,
    clean,
    clean_report,
    coerce_numeric,
    drop,
    drop_constant,
    drop_duplicates,
    fix_names,
    normalize_case,
    overview,
    parse_dates,
    recode_empty,
    recode_missing,
    rename,
    select,
    standard_tidy,
)
from .msa import BiasResult, LinearityResult, StabilityResult
from .palette import set_theme
from .pareto_analysis import ContingencyResult, ParetoResult, contingency, pareto, test_independence
from .regression import (
    AnovaResult,
    CorrelationResult,
    RegressionResult,
    correlation,
)
from .sampling import (
    AOQResult,
    LotDisposition,
    OCCurveResult,
    SamplingPlan,
    Z19Disposition,
    Z19Plan,
    find_plan,
    sampling_plan,
    z14_plan,
    z19_plan,
)
from .timeseries_charts import CUSUMResult, EWMAResult
from . import doe as design
from .doe import DOEResult, Design, LenthResult
from . import power
from .power import PowerResult
from .process_sigma import ProcessSigmaResult, compute as process_sigma
from .attribute_agreement import AttributeAgreementResult
from .posthoc import PosthocResult
from .nonparametric import test_medians, test_repeated
from .precontrol import PrecontrolResult
from .regression_ext import LogisticResult, NonlinearResult
from .timeseries import TimeSeriesResult
from .multivari import MultivariResult
from . import reliability
from .reliability.availability import availability
from .reliability import (
    LifeFitResult,
    KaplanMeierResult,
    SystemReliabilityResult,
    BearingLifeResult,
    AvailabilityResult,
    MTBFResult,
    DemonstrationResult,
)
from .errors import MissingPrerequisiteError, PyQCError
from .registry import ANALYSES, ANALYSES_BY_NAME, Analysis, list_analyses

from importlib.metadata import version as _version, PackageNotFoundError as _PkgNotFound
try:
    __version__ = _version("mfgqc")
except _PkgNotFound:  # running from a source checkout without an installed dist
    __version__ = "0.0.0+unknown"

__all__ = [
    # fluent entry points
    "load",
    "from_wide",
    "overview",
    "clean",
    "clean_report",
    # data model
    "QCData",
    "QCMeta",
    "Step",
    "Subgroups",
    "Crossed",
    "Overview",
    "AssumptionCheck",
    # cleaning tasks
    "fix_names",
    "coerce_numeric",
    "parse_dates",
    "recode_missing",
    "recode_empty",
    "drop_constant",
    "drop_duplicates",
    "select",
    "drop",
    "rename",
    "normalize_case",
    "standard_tidy",
    # results
    "CapabilityResult",
    "ControlChartResult",
    "Violation",
    "GageRRResult",
    "HypothesisResult",
    "BiasResult",
    "LinearityResult",
    "StabilityResult",
    "RegressionResult",
    "AnovaResult",
    "CorrelationResult",
    "ParetoResult",
    "ContingencyResult",
    "EWMAResult",
    "CUSUMResult",
    # hypothesis tests
    "test_mean",
    "test_means",
    "test_paired",
    "test_anova",
    "test_variance",
    "test_proportion",
    "test_proportions",
    # regression / ANOVA / correlation
    "correlation",
    # Pareto + contingency
    "pareto",
    "contingency",
    "test_independence",
    # acceptance sampling
    "sampling_plan",
    "find_plan",
    "set_theme",
    "z14_plan",
    "z19_plan",
    "SamplingPlan",
    "OCCurveResult",
    "AOQResult",
    "LotDisposition",
    "Z19Plan",
    "Z19Disposition",
    # design of experiments
    "design",
    "DOEResult",
    "Design",
    "LenthResult",
    # sample size & power
    "power",
    "PowerResult",
    # attributes capability
    "process_sigma",
    "ProcessSigmaResult",
    # attribute agreement (kappa)
    "AttributeAgreementResult",
    # multiple comparisons & nonparametrics
    "PosthocResult",
    "test_medians",
    "test_repeated",
    # SPC additions
    "PrecontrolResult",
    # regression additions
    "LogisticResult",
    "NonlinearResult",
    # time-series & multi-vari
    "TimeSeriesResult",
    "MultivariResult",
    # reliability & maintainability
    "reliability",
    "availability",
    "LifeFitResult",
    "KaplanMeierResult",
    "SystemReliabilityResult",
    "BearingLifeResult",
    "AvailabilityResult",
    "MTBFResult",
    "DemonstrationResult",
    # errors
    "PyQCError",
    "MissingPrerequisiteError",
    # analysis registry (for frontends)
    "ANALYSES",
    "ANALYSES_BY_NAME",
    "Analysis",
    "list_analyses",
    "__version__",
]
