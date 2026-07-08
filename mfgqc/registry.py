"""Machine-readable catalog of mfgQC analyses and what each one needs.

A frontend builds its menu from this catalog instead of hard-coding the method
list (which would drift out of sync as the package grows). ``ANALYSES`` is a
tuple of :class:`Analysis` records; :func:`list_analyses` returns the same thing
as a list of JSON-serializable dicts.

Requirement tokens (the ``requires`` field) use a small fixed vocabulary so a UI
can map them to prompts. They line up with
:attr:`mfgqc.errors.MissingPrerequisiteError.missing`:

* ``"measure"``    -- a numeric measure column (set via ``load(measure=...)``)
* ``"spec"``       -- at least one spec limit (``.spec(lower=/upper=)``)
* ``"spec:both"``  -- both spec limits
* ``"subgroup"``   -- a subgroup/time role or a ``subgroup_size``
* ``"role:NAME"``  -- a specific role (e.g. ``"role:part"``)
* ``"factors"``    -- one or more categorical factor columns (a call argument)
* ``"predictors"`` -- one or more predictor columns (a call argument)
* ``"reference"``  -- a known reference value (a call argument)
* ``"params"``     -- scalar inputs passed directly (no table needed)
* ``"data"``       -- raw sample arrays passed directly
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Analysis:
    """One catalog entry describing an analysis and its inputs."""

    name: str
    """Short identifier, e.g. ``"capability"``."""
    call: str
    """How it is invoked, e.g. ``"QCData.capability"`` or ``"mfgqc.process_sigma"``."""
    kind: str
    """``"fluent"`` (a :class:`~mfgqc.QCData` method) or ``"function"`` (module-level)."""
    requires: tuple[str, ...]
    """Required input tokens (see the module docstring vocabulary)."""
    params: tuple[str, ...]
    """Names of the tunable call arguments (knobs) the analysis accepts."""
    result_type: str
    """The result class name returned (whose ``.summary()``/``.view()`` to render)."""
    description: str
    """One-line human description for the menu."""


# --------------------------------------------------------------------------- #
# Data-driven (fluent) analyses: consume a loaded table via QCData.
# --------------------------------------------------------------------------- #
_FLUENT = (
    Analysis("capability", "QCData.capability", "fluent",
             ("measure", "spec"), ("method", "alpha"),
             "CapabilityResult", "Process capability indices (Cp, Cpk, Pp, Ppk)."),
    Analysis("bayes_capability", "QCData.bayes_capability", "fluent",
             ("measure", "spec"), ("prior", "seed", "draws", "cred_level"),
             "BayesCapabilityResult",
             "Bayesian capability: posterior Pp/Ppk/ppm with credible intervals."),
    Analysis("attribute_capability", "QCData.attribute_capability", "fluent",
             ("measure",), ("defect", "opportunities", "kind"),
             "ProcessSigmaResult", "Attribute (defective/defect) capability and sigma level."),
    Analysis("control_chart", "QCData.control_chart", "fluent",
             ("measure", "subgroup"), ("kind", "rules", "n"),
             "ControlChartResult", "Shewhart control chart (xbar/R/S, I-MR, p/np/c/u)."),
    Analysis("ewma_chart", "QCData.ewma_chart", "fluent",
             ("measure",), ("lam", "L", "mu0", "sigma"),
             "EWMAResult", "EWMA control chart for small sustained shifts."),
    Analysis("cusum_chart", "QCData.cusum_chart", "fluent",
             ("measure",), ("k", "h", "mu0", "sigma"),
             "CUSUMResult", "Tabular CUSUM control chart."),
    Analysis("short_run_chart", "QCData.short_run_chart", "fluent",
             ("measure", "subgroup"), ("by", "target", "rules"),
             "ControlChartResult", "Short-run (deviation-from-target) control chart."),
    Analysis("precontrol", "QCData.precontrol", "fluent",
             ("measure", "spec:both"), (),
             "PrecontrolResult", "Pre-control stoplight zones from the spec."),
    Analysis("gage_rr", "QCData.gage_rr", "fluent",
             ("measure", "role:part", "role:operator", "role:replicate"), ("method", "alpha"),
             "GageRRResult", "Gage repeatability & reproducibility (measurement system)."),
    Analysis("bias_study", "QCData.bias_study", "fluent",
             ("measure", "reference"), ("alpha",),
             "BiasResult", "Measurement bias vs a known reference value."),
    Analysis("linearity_study", "QCData.linearity_study", "fluent",
             ("measure", "reference"), ("alpha",),
             "LinearityResult", "Gage linearity across the measurement range."),
    Analysis("stability_study", "QCData.stability_study", "fluent",
             ("measure", "subgroup"), ("kind", "rules"),
             "StabilityResult", "Measurement-system stability over time."),
    Analysis("attribute_agreement", "QCData.attribute_agreement", "fluent",
             ("role:part", "role:appraiser"), ("rating", "part", "appraiser", "reference",
                                               "trial", "ordinal"),
             "AttributeAgreementResult", "Attribute agreement / kappa across appraisers."),
    Analysis("regress", "QCData.regress", "fluent",
             ("measure", "predictors"), ("on", "select", "criterion", "model", "start"),
             "RegressionResult", "Linear / multiple / nonlinear regression."),
    Analysis("logistic", "QCData.logistic", "fluent",
             ("measure", "predictors"), ("on",),
             "LogisticResult", "Logistic regression for a binary response."),
    Analysis("anova", "QCData.anova", "fluent",
             ("measure", "factors"), ("factors", "interaction"),
             "AnovaResult", "Factorial analysis of variance."),
    Analysis("doe", "QCData.doe", "fluent",
             ("measure", "factors"), ("design", "factors", "order", "reduce"),
             "DOEResult", "Design-of-experiments factorial effect analysis."),
    Analysis("multivari", "QCData.multivari", "fluent",
             ("measure", "factors"), ("factors",),
             "MultivariResult", "Multi-vari chart decomposition of variation."),
    Analysis("timeseries", "QCData.timeseries", "fluent",
             ("measure",), ("lags",),
             "TimeSeriesResult", "Trend (Mann-Kendall), slope and autocorrelation."),
    Analysis("life_fit", "QCData.life_fit", "fluent",
             ("measure",), ("dist", "method", "conf"),
             "LifeFitResult", "Reliability life distribution fit (with censoring)."),
    Analysis("life_table", "QCData.life_table", "fluent",
             ("measure",), ("conf",),
             "KaplanMeierResult", "Kaplan-Meier nonparametric survival estimate."),
)

# --------------------------------------------------------------------------- #
# Parameter-driven analyses: scalar inputs or raw arrays, no loaded table.
# --------------------------------------------------------------------------- #
_FUNCTION = (
    Analysis("process_sigma", "mfgqc.process_sigma", "function",
             ("params",), ("defects", "units", "opportunities", "kind", "alpha"),
             "ProcessSigmaResult", "DPMO and sigma level from defect counts."),
    Analysis("pareto", "mfgqc.pareto", "function",
             ("data",), ("counts",),
             "ParetoResult", "Pareto analysis of category counts."),
    Analysis("contingency", "mfgqc.contingency", "function",
             ("data",), ("table",),
             "ContingencyResult", "Chi-square test of independence on a table."),
    Analysis("correlation", "mfgqc.correlation", "function",
             ("data",), ("df", "cols", "method"),
             "CorrelationResult", "Correlation matrix (Pearson/Spearman)."),
    Analysis("test_means", "mfgqc.test_means", "function",
             ("data",), ("a", "b", "method", "alternative"),
             "HypothesisResult", "Two-sample test of means (auto-routes pooled/Welch/MW)."),
    Analysis("test_mean", "mfgqc.test_mean", "function",
             ("data",), ("x", "mu0", "alternative"),
             "HypothesisResult", "One-sample test of a mean."),
    Analysis("test_anova", "mfgqc.test_anova", "function",
             ("data",), ("groups", "method"),
             "HypothesisResult", "One-way ANOVA / Welch / Kruskal across groups."),
    Analysis("test_medians", "mfgqc.test_medians", "function",
             ("data",), ("groups", "method"),
             "HypothesisResult", "Nonparametric test of medians (Mood's)."),
    Analysis("power_t_test", "mfgqc.power.t_test", "function",
             ("params",), ("effect", "n", "power", "alpha", "kind", "alternative"),
             "PowerResult", "Sample size / power for a t-test."),
    Analysis("power_anova", "mfgqc.power.anova", "function",
             ("params",), ("groups", "effect", "n", "power", "alpha"),
             "PowerResult", "Sample size / power for one-way ANOVA."),
    Analysis("power_proportion", "mfgqc.power.proportion", "function",
             ("params",), ("p1", "p2", "n", "power", "alpha", "kind"),
             "PowerResult", "Sample size / power for a proportion test."),
    Analysis("sampling_plan", "mfgqc.sampling_plan", "function",
             ("params",), ("n", "c", "lot_size", "model"),
             "SamplingPlan", "Acceptance sampling plan and OC curve."),
    Analysis("z19_plan", "mfgqc.z19_plan", "function",
             ("params",), ("lot_size", "aql", "level", "severity"),
             "Z19Plan", "ANSI/ASQ Z1.9 variables sampling plan."),
    Analysis("reliability_system", "mfgqc.reliability.system", "function",
             ("params",), ("structure",),
             "SystemReliabilityResult", "System reliability from a block structure."),
    Analysis("bearing_life", "mfgqc.reliability.bearing_life", "function",
             ("params",), ("C", "P", "rpm", "kind"),
             "BearingLifeResult", "ISO 281 bearing L10 life."),
    Analysis("mtbf", "mfgqc.reliability.mtbf", "function",
             ("params",), ("qc_or_T", "failures", "kind", "conf"),
             "MTBFResult", "MTBF point estimate and confidence bounds."),
    Analysis("demonstration_test", "mfgqc.reliability.demonstration_test", "function",
             ("params",), ("reliability", "confidence", "n", "failures", "dist", "shape"),
             "DemonstrationResult", "Reliability demonstration test sizing."),
    Analysis("availability", "mfgqc.availability", "function",
             ("params",), ("mtbf", "mttr", "kind", "pm_time", "pm_freq", "logistics_delay"),
             "AvailabilityResult", "System availability (inherent/operational)."),
    Analysis("bayes_proportion_capability", "mfgqc.bayes.proportion_capability", "function",
             ("data",), ("n_fail", "n_trials", "max_proportion", "prior", "mapping", "cred_level"),
             "BayesProportionResult",
             "Bayesian proportion capability: Beta-Binomial posterior and P(p <= target)."),
    Analysis("bayes_rate_capability", "mfgqc.bayes.rate_capability", "function",
             ("data",), ("exposures", "max_rate", "prior", "cred_level"),
             "BayesRateResult",
             "Bayesian rate capability: Gamma-Poisson posterior, P(rate <= target), G4 dispersion."),
    Analysis("bayes_compare", "mfgqc.bayes.compare", "function",
             ("params",), ("a", "b", "seed", "draws", "cred_level", "labels"),
             "ComparisonResult",
             "Compare two Bayesian fits: P(B beats A) on mean, sd, Cpk, Ppk."),
    Analysis("bayes_assurance", "mfgqc.bayes.assurance", "function",
             ("params",), ("result", "target", "decide", "n_grid", "sims", "inner_draws", "seed"),
             "AssuranceResult",
             "Predictive sample-size assurance: P(future analysis concludes capable) by n."),
    Analysis("bayes_phase1", "mfgqc.bayes.phase1", "function",
             ("data",), ("prior",),
             "FrozenReference",
             "Freeze an in-control dataset into a digest-verified monitoring reference."),
    Analysis("bayes_monitor", "mfgqc.bayes.monitor", "function",
             ("params",), ("reference", "subgroups", "tests", "alpha", "R", "seed", "labels"),
             "MonitorResult",
             "Screen new subgroups against a phase-1 reference (posterior-predictive p-values)."),
    Analysis("bayes_predictive_check", "mfgqc.bayes.predictive_check", "function",
             ("data",), ("statistic", "prior", "R", "seed"),
             "PredictiveCheckResult",
             "Posterior predictive check: how extreme a statistic is under the model (BDA3 6.3)."),
    Analysis("bayes_capability_censored", "mfgqc.bayes.capability_censored", "function",
             ("data", "spec"),
             ("lower", "upper", "target", "truncation", "censoring", "prior", "seed", "draws", "cred_level"),
             "BayesCensoredCapabilityResult",
             "Censored/truncated capability on a grid posterior (recovers pre-sort process)."),
    Analysis("bayes_pooled_capability", "mfgqc.bayes.pooled_capability", "function",
             ("data", "spec"),
             ("lower", "upper", "target", "prior", "seed", "draws", "cred_level", "tau_points"),
             "PooledCapabilityResult",
             "Hierarchical pooled capability across positions; P(min_j Cpk_j >= target)."),
    Analysis("bayes_guardband", "mfgqc.bayes.guardband", "function",
             ("params",), ("result", "sigma_gauge", "c_scrap", "c_escape", "grid", "ndraws", "seed"),
             "GuardbandResult",
             "Cost-optimal acceptance limits under gauge error (scrap vs escape trade-off)."),
    Analysis("bayes_shortrun", "mfgqc.bayes.shortrun", "function",
             ("data",),
             ("target", "d", "lower", "upper", "prior", "p_star", "allow_vague", "cred_level"),
             "ShortRunResult",
             "Short-run sequential chart: chained N-Inv-chi2 posteriors, P(|mu-target|>d) per stage."),
)

ANALYSES: tuple[Analysis, ...] = _FLUENT + _FUNCTION
"""The full catalog of analyses, ordered fluent-first then parameter-driven."""

ANALYSES_BY_NAME: dict[str, Analysis] = {a.name: a for a in ANALYSES}
"""Lookup of :class:`Analysis` records by ``name``."""


def list_analyses() -> list[dict]:
    """Return the catalog as a list of JSON-serializable dicts (for a UI)."""
    return [asdict(a) for a in ANALYSES]
