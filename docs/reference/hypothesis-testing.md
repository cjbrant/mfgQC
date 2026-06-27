---
title: Hypothesis testing
---

# Hypothesis testing and association

A hypothesis test asks a yes-or-no question about a process and answers it with a
number. Did the new supplier shift the mean? Do three lines fill to the same level?
Is the scratch rate different on the night shift? You state a null hypothesis (the
"nothing changed" position), compute a test statistic from the data, and read off a
p-value: the probability of seeing a result at least this extreme if the null were
true. A small p-value (conventionally below 0.05) is evidence against the null.

The hard part is not the arithmetic. scipy computes the statistics. The hard part is
picking the *right* test, because most tests assume something about the data, and the
wrong test on the wrong data gives a confident wrong answer. mfgQC's tests of means
and variances make that choice for you and tell you why. They check the assumptions
first (normality of each group, equal variance across groups), then select the test
those checks call for, and report the selection in the result. This is the
"statistical guardrails" pillar applied to inference: the assumption check *is* the
routing logic. You can override the choice with `method=`; overriding still prints
the assumption checks so a forced wrong choice is visible.

Every function on this page returns a frozen result object with `.report()` (the full
text below), `.summary()` (a flat dict), `.to_dict()` (JSON), and `.view()` (a chart).
The source for the routing and statistics is `mfgqc/hypothesis.py`,
`mfgqc/nonparametric.py`, and `mfgqc/posthoc.py`.

## 1. The vocabulary

A few terms recur. Define them once.

- **Normality.** The data follow a normal (bell-shaped) distribution. The t-test and
  ANOVA assume each group is approximately normal. mfgQC checks this with the
  **Anderson-Darling** test (`mfgqc/assumptions.py`, `check_normality`), a goodness-of-fit
  test that is sensitive in the tails. A p-value below 0.05 means "reject normality."
- **Homogeneity of variance** (also "equal variance"). The groups have the same spread.
  The pooled t-test and classic ANOVA assume this. mfgQC checks it with **Levene's**
  test centered at the median (`check_homogeneity`), which is itself tolerant of
  non-normal data.
- **Parametric vs non-parametric.** A parametric test (t, ANOVA) assumes a distribution
  shape and tests the mean. A non-parametric test (Mann-Whitney, Kruskal-Wallis,
  Mood's median) ranks the data instead and tests a location shift or the median, which
  needs no normality assumption. Non-parametric tests are safer when normality fails
  and slightly less sensitive when it holds.
- **Effect size.** The p-value tells you whether a difference is detectable; the effect
  size tells you how big it is. mfgQC reports Cohen's d for mean tests (the difference
  in standard-deviation units), $\eta^2$ / $\epsilon^2$ for ANOVA (the fraction of
  variation the groups explain), and the rank-biserial correlation for Mann-Whitney.

## 2. One-sample and two-sample tests of means

### `test_mean(values, target)`: one sample vs a target

Use this to compare one set of measurements against a fixed number: a nominal
dimension, a contract limit, a target fill weight. The null is that the true mean
equals `target`.

The statistic is the one-sample t:

$$
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}, \qquad \text{df} = n - 1
$$

where $\bar x$ is the sample mean, $\mu_0$ is `target`, $s$ is the sample standard
deviation, and $n$ is the count. The effect size is Cohen's
$d = (\bar x - \mu_0)/s$. mfgQC checks normality with Anderson-Darling. The default is
`auto=False`: if the data fail normality it **still runs the t-test** but adds a
recommendation to rerun with `auto=True`, which switches to the **Wilcoxon
signed-rank** test on $x - \mu_0$. This is the one place mfgQC will not silently change
the method; the one-sample router is opt-in by design.

```python
import numpy as np, mfgqc
x = np.array([10.2, 9.8, 10.1, 10.4, 9.9, 10.0, 10.3, 9.7, 10.2, 10.1])
mfgqc.test_mean(x, 10.0).report()
```

```
Hypothesis Test: one-sample t
=============================
H0: mean = 10.0
H1: mean != 10.0  (alternative=two-sided)

one-sample t: statistic=1  df=9  p=0.3434
effect size (Cohen's d) = 0.3162
95% CI = (9.9116, 10.228)
decision at alpha=0.05: fail to reject H0

Assumption checks:
  [PASS] normality (Anderson-Darling): AD=0.176, p=0.895; skew 0.249; n=10 [low power]
```

The `[low power]` tag is honest about small n: with ten points the normality test has
limited resolving power, so a "pass" is weak evidence of normality, not proof of it.

### `test_means(a, b)`: two independent samples, routed

Use this to compare two groups: two machines, two suppliers, before and after a change
measured on different parts. (For the same parts measured twice, use the paired test,
`test_paired`.) The null is that the two means are equal.

`test_means` routes by default. It runs Anderson-Darling on each group and Levene
across the two, then selects:

| Both groups normal? | Equal variance? | Test selected |
|---|---|---|
| yes | yes | **Student's pooled t** |
| yes | no | **Welch's t** |
| no | (not reached) | **Mann-Whitney U** |

The pooled and Welch statistics share the form $t = (\bar x_a - \bar x_b)/\text{SE}$ but
differ in the standard error and degrees of freedom. Pooled uses one combined variance
and $\text{df}=n_a+n_b-2$; Welch uses the separate variances and the
Welch-Satterthwaite df, which does not assume equal spread. When either group fails
normality the router drops to Mann-Whitney U, which ranks all the values together and
tests whether one group tends to sit above the other. Effect size is Cohen's d (pooled
variance) for the t routes and the rank-biserial correlation
$1 - 2U/(n_a n_b)$ for Mann-Whitney.

The routing decision is exactly the `if/elif` in `test_means`: not normal goes to
Mann-Whitney; normal but unequal variance goes to Welch; normal and equal variance
goes to pooled Student's t. Force a specific test with `method='pooled'`, `'welch'`, or
`'mannwhitney'`. Forcing still prints all three assumption checks, and if your forced
choice disagrees with the routed one, the result adds a recommendation saying so.

Worked example: fill weights from two lines.

```python
a = np.array([20.1, 20.3, 19.8, 20.0, 20.2, 19.9, 20.4, 20.1, 20.0, 19.7])
b = np.array([20.5, 20.7, 20.4, 20.6, 20.8, 20.3, 20.9, 20.5, 20.6, 20.4])
mfgqc.test_means(a, b, labels=("line A", "line B")).report()
```

```
Hypothesis Test: Student's t (pooled)
=====================================
H0: mean(line A) = mean(line B)
H1: mean(line A) != mean(line B)  (alternative=two-sided)

selected Student's t (pooled): both groups ~normal with equal variance -> pooled Student's t
Student's t (pooled): statistic=-5.712  df=18  p=2.045e-05
effect size (Cohen's d) = -2.554
95% CI = (-0.71127, -0.32873)
decision at alpha=0.05: reject H0

Assumption checks:
  [PASS] normality (Anderson-Darling): AD=0.131, p=0.97; skew 1.77e-14; n=10 [low power]
  [PASS] normality (Anderson-Darling): AD=0.208, p=0.811; skew 0.351; n=10 [low power]
  [PASS] homogeneity_of_variance (Levene): variance ratio 1.32, p=0.701; n=20
```

Both groups passed normality and Levene passed (variance ratio 1.32), so the router
chose the pooled t. The line means differ by about 0.52 units, the difference is
detectable (p around $2\times10^{-5}$), and the effect is large (d about 2.6 standard
deviations).

When a group is skewed, the route changes. With one heavy-tailed group the same call
selects Mann-Whitney:

```
selected Mann-Whitney U: normality failed (AD p=0.000212) -> non-parametric Mann-Whitney U
Mann-Whitney U: statistic=17  p=0.01334
effect size (rank-biserial) = 0.66
...
  [FAIL] normality (Anderson-Darling): AD=1.57, p=0.000212; skew 2.11; n=10 [low power]
```

Note the order of precedence: normality is checked first. If either group fails it, the
result is Mann-Whitney regardless of what Levene says (the equal-variance branch is
never reached).

Source for the t-tests and the routing logic: scipy (`ttest_1samp`, `ttest_ind`,
`mannwhitneyu`, `wilcoxon`) for the statistics, with the assumption checks and routing
added in `mfgqc/hypothesis.py`. The Student/Welch distinction and the pooled-variance
Cohen's d follow Montgomery, *Introduction to Statistical Quality Control*
([bibliography](bibliography.md)).

## 3. One-way ANOVA: `test_anova`

When you have three or more groups, you do not run a t-test on every pair: each test
carries its own false-positive risk and they accumulate. The analysis of variance
(ANOVA) asks one question across all groups at once: **are all the group means equal,
or does at least one differ?** It compares the variation *between* group means to the
variation *within* groups. If the between-group spread is large relative to the
within-group noise, the F statistic is large and the means are not all equal.

The classic one-way F statistic is

$$
F = \frac{\text{SS}_{\text{between}}/(k-1)}{\text{SS}_{\text{within}}/(N-k)},
$$

with $k$ groups and $N$ total observations. mfgQC reports
$\eta^2 = \text{SS}_{\text{between}}/\text{SS}_{\text{total}}$ as the effect size: the
fraction of total variation explained by the grouping.

`test_anova` routes the same way `test_means` does, one tier up:

| All groups normal? | Equal variance? | Test selected | Effect size |
|---|---|---|---|
| yes | yes | **classic one-way ANOVA** | $\eta^2$ |
| yes | no | **Welch's ANOVA** | (none) |
| no | (not reached) | **Kruskal-Wallis** | $\epsilon^2$ |

Welch's ANOVA (computed in `_welch_anova`) weights each group by $n_i/s_i^2$ and
adjusts the denominator degrees of freedom, so it does not assume equal spread.
Kruskal-Wallis is the rank-based, non-parametric analogue: it replaces the values with
their ranks and tests for a location shift among the groups. The omnibus normality
check shown in the report is the *worst-case* group (the one with the smallest
normality p-value) plus the single Levene check across all groups. Force a route with
`method='anova'`, `'welch'`, or `'kruskal'`.

Worked example: tensile strength from three suppliers.

```python
s1 = np.array([85.2, 86.1, 84.8, 85.5, 85.9, 84.7, 86.3, 85.1])
s2 = np.array([88.4, 89.1, 87.9, 88.7, 89.3, 88.0, 87.6, 88.9])
s3 = np.array([85.7, 86.4, 85.2, 85.9, 86.1, 85.0, 86.6, 85.4])
res = mfgqc.test_anova(s1, s2, s3, labels=("supplier 1", "supplier 2", "supplier 3"))
res.report()
```

```
Hypothesis Test: one-way ANOVA
==============================
H0: all group means are equal
H1: at least one group mean differs  (alternative=two-sided)

selected one-way ANOVA: groups ~normal with equal variance -> classic one-way ANOVA
one-way ANOVA: statistic=62.78  df=2  p=1.381e-09
effect size (eta^2) = 0.8567
decision at alpha=0.05: reject H0

Assumption checks:
  [PASS] normality (Anderson-Darling): AD=0.243, p=0.664; skew 0.159; n=8 [low power]
  [PASS] homogeneity_of_variance (Levene): variance ratio 1.16, p=0.936; n=24 [low power]
```

The means are not all equal (p about $1.4\times10^{-9}$), and the grouping explains
about 86% of the total variation ($\eta^2 = 0.857$). But the omnibus test does not say
*which* suppliers differ. That is the post-hoc step.

## 4. Post-hoc: which pairs differ?

A significant ANOVA tells you a difference exists somewhere. To find the specific
pairs, call `.posthoc()` on the ANOVA result. The catch is multiplicity: with three
groups there are three pairwise comparisons, and testing each at 0.05 inflates the
family-wise error rate. The post-hoc methods control that family-wise rate. mfgQC
routes the choice from the same assumptions the ANOVA used (`mfgqc/posthoc.py`,
`_route`):

| Situation | Method | What it controls |
|---|---|---|
| equal variances, normal (Tukey route) | **Tukey HSD** | all pairwise, family-wise, via the studentized range |
| unequal variances (Welch route) | **Games-Howell** | all pairwise, Welch-corrected per pair |
| Kruskal-Wallis route (non-normal) | **Dunn's test** | rank-based pairs, Holm-adjusted p-values |
| you pass `control=` | **Dunnett** | each group against one reference level |

Use Tukey when the ANOVA went the classic route and you want every pair. Use
Games-Howell when variances are unequal, because Tukey's pooled error term is then
wrong. Use Dunn after a Kruskal-Wallis, because the comparisons must stay on the rank
scale. Use Dunnett when one level is a control or baseline and the only comparisons of
interest are "each treatment vs the control" (fewer comparisons, more sensitivity than
all-pairs). The routing prefers `control=` over everything: passing it forces Dunnett.

Tukey reports each pair's mean difference, a studentized-range confidence interval, and
an adjusted p-value:

```python
res.posthoc().report()
```

```
Post-hoc multiple comparisons (tukey)
=====================================
family-wise control: Tukey HSD (family-wise across all pairs)
routed: equal variances and normal: Tukey HSD

pair                        diff    95% CI low   95% CI high     p adj
supplier 1 - supplier 2      -3.038        -3.787        -2.288  3.87e-09 *
supplier 1 - supplier 3     -0.3375        -1.087        0.4115     0.503
supplier 2 - supplier 3         2.7         1.951         3.449  2.96e-08 *

significant pairs (p < 0.05): supplier 1 vs supplier 2, supplier 2 vs supplier 3
```

Supplier 2 differs from both 1 and 3; suppliers 1 and 3 are statistically
indistinguishable (p = 0.503, and the interval for their difference spans zero).

Passing `control=` switches to Dunnett. Here, comparing each supplier to supplier 1:

```python
res.posthoc(control="supplier 1").report()
```

```
Post-hoc multiple comparisons (dunnett)
=======================================
family-wise control: Dunnett vs control='supplier 1' (family-wise)
routed: control='supplier 1' given: Dunnett against the control level

pair                        diff    95% CI low   95% CI high     p adj
supplier 2 - supplier 1       3.038         2.333         3.742  5.53e-10 *
supplier 3 - supplier 1      0.3375       -0.3665         1.042     0.431

significant pairs (p < 0.05): supplier 2 vs supplier 1
```

The post-hocs shown so far (Tukey, Games-Howell, Dunnett) report a confidence interval on
the original measurement scale, as the output above shows. Dunn's test is the exception, and
it is a different test: it is the post-hoc for the Kruskal-Wallis route, so it compares
*mean ranks* rather than means. mfgQC leaves its confidence interval as `nan`, because a
difference in ranks is not on the measurement scale. For Dunn, the Holm-adjusted p-value is
the number to act on.

## 5. Non-parametric location: `test_medians`

`test_medians` runs **Mood's median test** for $k \ge 2$ groups
(`mfgqc/nonparametric.py`). It is a location test like Kruskal-Wallis, but built around
the **grand median**: it counts how many points in each group fall above and below the
pooled median and runs a chi-square on that 2-by-k table. The null is a common median.

Mood's median test is the most outlier-tolerant of the location tests, because it only
uses above/below counts and ignores how far above or below a point sits. That tolerance
costs sensitivity, and the result says so directly:

```python
g1 = np.array([2.1, 2.3, 2.0, 2.2, 2.4])
g2 = np.array([3.0, 3.2, 2.9, 3.1, 3.3])
g3 = np.array([2.5, 2.6, 2.4, 2.7, 2.5])
mfgqc.test_medians(g1, g2, g3, labels=("L1", "L2", "L3")).report()
```

```
Hypothesis Test: mood_median
============================
H0: all groups share a common median
H1: at least one median differs  (alternative=two-sided)

requested medians; ran mood_median
mood_median: statistic=10.18  df=2  p=0.006162
decision at alpha=0.05: reject H0

Assumption checks:
  (none)

Recommendations:
  - Mood's median test is robust to outliers but LESS powerful than Kruskal-Wallis; prefer Kruskal-Wallis when a location-shift model holds.
```

This function does not route; it always runs Mood's median test. The other rank-based
location tests are reached through `test_anova` (Kruskal-Wallis for $k$ groups) and
`test_means` (Mann-Whitney U for two groups), where they sit as the non-normal branch
of the router described above. The statistic delegates to scipy's `median_test`.

## 6. Association

"Association" asks whether two variables move together, rather than whether a mean
shifted. mfgQC has two functions, one for categorical variables and one for numeric.

### `contingency(table)`: chi-square test of independence

Use this for two categorical variables: defect type by shift, pass/fail by tool,
supplier by disposition. You build a table of counts (rows by columns) and ask whether
the row classification and the column classification are independent. The statistic is
Pearson's chi-square,

$$
\chi^2 = \sum_{\text{cells}} \frac{(O - E)^2}{E},
$$

where $O$ is the observed count and $E$ the count expected under independence. mfgQC
computes this with scipy's `chi2_contingency` and **no Yates continuity correction**
(`correction=False`), so a 2-by-2 table is the raw Pearson statistic. It also reports
**Cramer's V**, $\sqrt{\chi^2/(N\,k)}$ with $k=\min(r-1,c-1)$, as an effect size on a
0-to-1 scale.

The chi-square approximation needs every expected cell count to be at least 5. mfgQC
checks this and surfaces it as an assumption (`expected_count`). Note what it does *not*
do: when the minimum expected count drops below 5 in a 2-by-2 table, mfgQC
**recommends** Fisher's exact test in the assumption message but does not run it. There
is no auto-route to Fisher here; `contingency` always computes chi-square. (Fisher's
exact is available via `test_proportions(..., auto=True)` for the two-proportion case.)

```python
import pandas as pd
table = pd.DataFrame({"day": [42, 30], "night": [18, 40]}, index=["scratch", "dent"])
mfgqc.contingency(table).report()
```

```
Chi-Square Test of Independence
===============================
table         = 2 x 2   N = 130
chi-square    = 9.633   dof = 1
p-value       = 0.001912
Cramer's V    = 0.2722
min expected  = 26.77

Assumption checks:
  [PASS] expected_count (min expected cell count >= 5): min expected count 26.77; n=130
```

The defect mix depends on the shift (p about 0.002), with a small-to-moderate
association (Cramer's V = 0.27). The smallest expected cell is 26.8, well above 5, so
the approximation is sound. A sibling function, `test_independence(df, row, col)`,
builds the table from two columns of a tidy DataFrame via `pandas.crosstab` and then
runs the same computation.

### `correlation(df, cols, method)`: Pearson or Spearman

Use this for numeric variables: does yield track furnace temperature, does runout track
spindle speed? `correlation` returns a full correlation matrix plus a p-value for every
off-diagonal pair (`mfgqc/regression.py`). Two methods:

- `method="pearson"` (default) measures the strength of a *linear* relationship. It
  assumes the relationship is roughly straight-line.
- `method="spearman"` correlates the *ranks* instead. It detects any monotonic
  relationship (consistently increasing or decreasing, even if curved) and is less
  sensitive to outliers.

Each pair's p-value tests the null of zero correlation, from scipy's `pearsonr` /
`spearmanr`. The function keeps only complete rows (it drops any row with a missing
value across the chosen columns) and needs at least 3 complete rows and 2 numeric
columns.

```python
df = pd.DataFrame({
    "temp":     [200, 205, 210, 215, 220, 225],
    "yield":    [78, 80, 83, 85, 88, 90],
    "pressure": [12, 11, 13, 12, 14, 13],
})
mfgqc.correlation(df, cols=["temp", "yield", "pressure"], method="pearson").report()
```

```
Correlation (pearson)
=====================
n = 6   variables: temp, yield, pressure

pair                                 r           p
temp ~ yield                    0.9984    3.93e-06
temp ~ pressure                 0.6625       0.152
yield ~ pressure                0.7041       0.118
```

Temperature and yield are nearly collinear (r = 0.998, p about $4\times10^{-6}$).
Pressure correlates positively with both, but at n = 6 neither pressure pair clears
0.05: the relationship may be real, but six points cannot resolve it. Correlation is
association, not cause; a strong r is a starting point for a designed experiment, not a
conclusion about mechanism.

## 7. Choosing a test

| You have | You want to know | Function |
|---|---|---|
| one sample, a target | is the mean on target? | `test_mean` |
| two independent groups | do the means differ? | `test_means` (routes) |
| the same parts, measured twice | did the change move them? | `test_paired` |
| three or more groups | are all means equal? | `test_anova` (routes), then `.posthoc()` |
| groups, outliers, want the median | do the medians differ? | `test_medians` |
| two categorical variables | are they independent? | `contingency` / `test_independence` |
| two or more numeric variables | do they move together? | `correlation` |

## 8. Assumptions, sources, and limits

What mfgQC checks for you, and where the honesty boundaries are:

- **Mean and variance tests** check normality (Anderson-Darling, per group) and equal
  variance (Levene, across groups) and route on the outcomes. `test_means` and
  `test_anova` route by default; `test_mean` and `test_paired` are opt-in (`auto=True`)
  and otherwise warn. The Anderson-Darling p-value uses the small-sample correction in
  `mfgqc/assumptions.py`; at small n the `[low power]` tag flags that a "pass" is weak.
- **Post-hoc** inherits the route from the ANOVA result it was called on. It does not
  re-test assumptions; it trusts the omnibus checks.
- **Mood's median test** makes no distributional assumption and does no routing; its
  trade-off (outlier-tolerant, lower power) is stated in the result.
- **Contingency** checks the minimum expected cell count (>= 5) and *recommends* Fisher's
  exact when that fails, but does not run Fisher. It always computes chi-square, without
  the Yates correction.
- **Correlation** does not test the normality or linearity that Pearson assumes; if you
  suspect a curved or outlier-driven relationship, use `method="spearman"`.

The statistics delegate to scipy so they match that reference exactly; mfgQC adds the
routing, effect sizes, confidence intervals, and the provenance record. The
Student/Welch t distinction, the one-way ANOVA decomposition, and the multiple-comparison
framing follow Montgomery, *Introduction to Statistical Quality Control*
([bibliography](bibliography.md)). The specific small-sample p-value approximations
(Anderson-Darling, Games-Howell, Dunn) are standard formulas, also found in scipy, rather than a single cited standard in the bibliography.

## See also

- [API reference](api.md) for the full signatures and parameters.
- [Power and sample size](../reference/power.md) to size a study *before* you run the
  test, so the experiment can detect a difference worth acting on.
- [Bibliography](bibliography.md) for the cited sources.
