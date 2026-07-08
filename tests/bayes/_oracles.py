"""Transcribed oracle constants for mfgqc.bayes tests.

Sourcing rule (matches tests/correctness): every value is either a published
input->output transcribed from a cited source, or marked [LOCK] pending
transcription. No value is ever taken from a prior mfgqc run.

Sources (PDFs held by the build owner, not committed):
  BDA3 = Gelman et al., Bayesian Data Analysis, 3rd ed.
  Hoff = Hoff, A First Course in Bayesian Statistical Methods.
  CdC  = Colosimo & del Castillo, Bayesian Process Monitoring, Control and Optimization.

Transcription notes record the source section and that the build agent typed them.
"""
from __future__ import annotations

# T2.1 - BDA3 sec 2.4, placenta previa (uniform prior Beta(1,1); y=437, n=980).
# Transcribed by the build agent from BDA3 sec 2.4 text and Table 2.1 (row
# alpha/(alpha+beta)=0.500, alpha+beta=2): posterior Beta(438,544), mean 0.446,
# sd 0.016, median 0.446, central 95% interval [0.415, 0.477].
PLACENTA = {
    "y": 437,
    "n": 980,
    "prior": (1.0, 1.0),
    "posterior_mean": 0.446,
    "posterior_sd": 0.016,
}
PLACENTA_CI95 = (0.415, 0.477)

# T2.3 - BDA3 sec 2.6, asthma mortality rate. Transcribed by the build agent from
# BDA3 sec 2.6 text: y=3 deaths, exposure x=2.0 (per 100,000); prior Gamma(3.0,5.0)
# (shape, rate) with mean 0.6 and 97.5% mass below 1.44; posterior Gamma(6.0,7.0)
# with mean 0.86; posterior P(theta > 1.0) = 0.30.
RATE_EXAMPLE = {
    "y": 3,
    "x": 2.0,
    "prior": (3.0, 5.0),          # (shape a, rate b)
    "prior_mean": 0.6,
    "prior_p975": 1.44,
    "posterior": (6.0, 7.0),
    "posterior_mean": 0.86,
    "p_theta_gt_1": 0.30,
}

# T2.2 - BDA3 sec 3.2, speed of light (Newcomb). Transcribed by the build agent
# from BDA3 sec 3.2 text: n=66, ybar=26.2, s=10.8, noninformative prior; the t_65
# 0.975 multiplier is 1.997 and the 95% central interval for mu is [23.6, 28.8].
# NOTE: BDA3 shows the 66 raw measurements only as a histogram (Figure 3.1), not
# as a data table (its "Table 3.1" is the unrelated bioassay data), so the raw
# vector needed to reproduce [23.6, 28.8] exactly is not available in BDA3. The
# full interval reproduction is a P1 item (noninformative fit); here only the
# t-multiplier is checked.
SPEED_OF_LIGHT = {"n": 66, "ybar": 26.2, "s": 10.8, "t_mult": 1.997, "mu_ci95": (23.6, 28.8)}
NEWCOMB_DATA: list | None = None

# T2.4 - BDA3 sec 6.3, posterior predictive check on the speed-of-light data with
# T(y)=min(y). Transcribed by the build agent from BDA3 sec 6.3 text and Figure
# 6.3: Newcomb's smallest observation is -44, and the smallest values of all 20
# posterior predictive replications are "much larger", so the normal model "clearly
# does not capture the variation" - a blatant misfit. BDA3 reports this check
# GRAPHICALLY (Fig 6.3), printing no numeric Bayesian p-value here, and the 66 raw
# values are not tabulated (only the min, -44, and the sec 3.2 summaries). So the
# exact p-value is not reproducible; the recoverable facts are the min and the
# qualitative verdict (min-check extreme, mean-check not).
NEWCOMB_MIN = -44.0

# T2.6 - Hoff sec 3.2.2, birth rates (Gamma-Poisson). Transcribed by the build
# agent from Hoff sec 3.2.2 (data sums, gamma(2,1) prior, and the R output for the
# posterior means, modes, and 95% quantile intervals; Pr(theta1>theta2)=0.97).
BIRTH_RATES = {
    "prior": (2.0, 1.0),  # gamma(a=2, b=1)
    "group1": {"n": 111, "sum_y": 217, "posterior": (219, 112),
               "mean": 1.955357, "mode": 1.946429, "ci95": (1.704943, 2.222679)},
    "group2": {"n": 44, "sum_y": 66, "posterior": (68, 45),
               "mean": 1.511111, "mode": 1.488889, "ci95": (1.173437, 1.890836)},
    "prob_g1_gt_g2": 0.97,
}

# T2.7 - Hoff sec 8.1, two-group math scores. Transcribed by the build agent from
# Hoff sec 8.1 (frequentist summaries; the raw scores are not tabulated there).
MATH_SCORES = {"ybar1": 50.81, "n1": 31, "ybar2": 46.15, "n2": 28,
               "sp": 10.44, "t": 1.74, "p_two_sided": 0.087}

# T2.9 - BDA3 sec 5.5, eight schools (hierarchical normal, KNOWN sigma_j).
# Transcribed by the build agent from BDA3 Table 5.2 (data) and Table 5.3
# (posterior quantiles), cross-checked against three independent reads.
# CAVEAT: Table 5.3 is explicitly a "Summary of 200 simulations" and the book
# calls the jaggedness "an artifact caused by sampling variability from using
# only 200 random draws" - so those quantiles are COARSE (integer-rounded), a
# loose ballpark reference, not a tight oracle. The exact analytic anchors are
# the complete-pooling estimate and its interval. Also the model uses known
# sigma_j, whereas mfgqc's pooled_capability estimates the within-position sigma;
# T2.9 therefore validates the hierarchical-normal core with that delta noted.
EIGHT_SCHOOLS = {
    "data": [  # (school, effect y_j, standard error sigma_j) - BDA3 Table 5.2
        ("A", 28.0, 15.0), ("B", 8.0, 10.0), ("C", -3.0, 16.0), ("D", 7.0, 11.0),
        ("E", -1.0, 9.0), ("F", 1.0, 11.0), ("G", 18.0, 10.0), ("H", 12.0, 18.0),
    ],
    # complete-pooling common-effect estimate (exact analytic; BDA3 sec 5.5 text)
    "pooled_mean": 7.7, "pooled_var": 16.6, "pooled_se": 4.1,
    "pooled_ci95": (-0.5, 15.9),
    # Table 5.3 posterior medians of theta_j (200-draw, integer-rounded, LOOSE)
    "posterior_median": {"A": 10, "B": 8, "C": 7, "D": 8, "E": 5, "F": 6, "G": 10, "H": 8},
    "prior_on_tau": "uniform on tau >= 0 (noninformative)",
}
