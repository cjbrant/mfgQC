# Correctness test suite

These tests pin mfgQC to **independent sources it was not built against**. They are
distinct from the regression oracle tests in `tests/` (which re-run the build
oracles: Montgomery, AIAG, Lawson, R `survreg`). The point of correctness tests is
to catch the case where mfgQC and its build oracle are *both* wrong in the same way.

## Sourcing rules

1. No expected value here is ever taken from a prior mfgQC run. Either a published
   authority states input → output, or an independent engine computes the answer
   at test time.
2. Every test's docstring **names its source**.
3. NIST is preferred for the QC charts and the core inference examples.
4. R-backed tests run `Rscript` as a subprocess and have the R engine emit **both**
   the input data and the oracle answer, so mfgQC is fed R's data and compared to
   R's answer. If `Rscript` or a package is missing, the test **skips** (it never
   silently passes).
5. Lawson examples are only used here when they were **not** mfgQC build oracles.
   The DOE build oracles are Lawson's `volt` (2³, p. 91), `chem` (2⁴, p. 97) and
   `soup` (half-fraction, p. 201) datasets; those are re-verified in `tests/doe/`,
   not here. The correctness suite uses Lawson's independent `COdata` factorial.

## Independent sources

| Source | Used for |
| --- | --- |
| NIST/SEMATECH e-Handbook (itl.nist.gov/div898/handbook) | capability, individuals & c charts, EWMA, CUSUM, two-sample t, one-way ANOVA |
| NIST StRD certified datasets (itl.nist.gov/div898/strd) | linear regression (Norris) |
| Lawson, *Design and Analysis of Experiments with R* (2017), printed examples | two-factor factorial ANOVA (Hunter CO / `COdata`) |
| R `qcc` (live) | X-bar/R charts (pistonrings), revised p-chart (orangejuice / Lawson) |
| R `SixSigma` (live) | capability (ss.data.ca) |
| scipy / statsmodels / scikit-learn (in-test) | chi-square, correlation, process-sigma Z, power, Tukey HSD, Mood's median, Cohen's kappa, Weibull MLE, DOE effects, Mann-Kendall, OLS trend slope |

## Module → source coverage

| Analysis module | Independent source | File |
| --- | --- | --- |
| capability | NIST 6.1.6; R SixSigma `ss.data.ca` | `test_capability_nist.py`, `test_sixsigma_capability.py` |
| control charts (variables) | NIST 6.3.2.2 individuals; R qcc pistonrings X-bar/R | `test_control_charts_nist.py`, `test_qcc_charts.py` |
| control charts (attribute) | NIST 6.3.3.1 c-chart; R qcc/Lawson revised p-chart | `test_control_charts_nist.py`, `test_qcc_charts.py` |
| EWMA / CUSUM | NIST 6.3.2.4 / 6.3.2.3 | `test_ewma_cusum_nist.py` |
| regression | NIST StRD Norris (certified) | `test_regression_nist.py` |
| hypothesis (t-test, ANOVA) | NIST AUTO83B t-test; NIST 7.4.3 ANOVA | `test_hypothesis_nist.py` |
| contingency / chi-square | scipy.stats.chi2_contingency | `test_inference_scipy.py` |
| correlation | scipy.stats.pearsonr / spearmanr | `test_inference_scipy.py` |
| process_sigma | scipy.stats.norm (DPMO↔Z identity) | `test_inference_scipy.py` |
| power | statsmodels TTestIndPower / FTestAnovaPower | `test_inference_scipy.py` |
| posthoc | statsmodels pairwise_tukeyhsd | `test_inference_scipy.py` |
| nonparametric | scipy.stats.median_test (Mood's) | `test_inference_scipy.py` |
| attribute_agreement | scikit-learn cohen_kappa_score (incl. weighted) | `test_inference_scipy.py` |
| reliability (life fit) | scipy.stats.weibull_min MLE | `test_reliability_scipy.py` |
| doe | statsmodels OLS (effect = 2·coef); Lawson Hunter-CO factorial ANOVA | `test_doe_statsmodels.py`, `test_factorial_lawson.py` |
| timeseries | Mann-Kendall by construction; scipy.linregress | `test_timeseries_construct.py` |

## Not yet covered by an independent source

These modules are still verified only against their build oracle (regression
tests), pending a suitable independent example: gage R&R / MSA, acceptance
sampling (Z1.4/Z1.9), precontrol, multi-vari, and the reliability system/
availability/MTBF helpers. Candidates: NIST MSA examples, published Z1.4 OC
points, and reliawiki worked examples.

## Running

```bash
.venv/bin/pytest tests/correctness/ -q          # all (R tests skip if R absent)
.venv/bin/pytest tests/correctness/ -v -k qcc   # just the qcc-backed ones
```

R packages for the live-engine tests:

```bash
Rscript -e 'install.packages(c("qcc","SixSigma","jsonlite"), repos="https://cloud.r-project.org")'
```
