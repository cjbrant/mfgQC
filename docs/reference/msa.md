---
title: Measurement systems analysis
---

# Measurement systems analysis (beyond gage R&R)

A gage R&R study answers one question: of the variation you see, how much is the part
and how much is the measurement system? That study has its own page,
[Gage R&R](gage-rr.md). But repeatability and reproducibility are not the whole picture.
A gage can be precise and still wrong. It can read true near the middle of its range and
drift at the ends. It can agree with itself today and disagree next month. And when the
"measurement" is a pass/fail call by a human, the question changes shape entirely.

This page documents the four MSA studies mfgQC computes outside the variance study:

- **Bias** (`bias_study`): is the average reading off from a known truth?
- **Linearity** (`linearity_study`): does that bias change across the operating range?
- **Stability** (`stability_study`): does the system hold steady over time?
- **Attribute agreement** (`attribute_agreement`): do appraisers making categorical calls
  agree with themselves, with each other, and with the standard?

Every formula and default below is pinned to the implementation in `mfgqc/msa.py` and
`mfgqc/attribute_agreement.py`. The source standard for all four is AIAG, *Measurement
Systems Analysis (MSA) Reference Manual,* 4th ed. (see the [bibliography](bibliography.md)).

---

## 1. Bias study

### What it is and when to use it

Bias is the difference between the average of repeated measurements and a known reference
value. You measure one part many times on one gage. The part has a reference value you
trust, from a more accurate instrument or a certified standard. If your gage reads high or
low on average, that offset is the bias. Use a bias study when you have a master part with
a known value and you want to know whether your gage is centered on the truth.

### The procedure

Let the gage produce $n$ readings $y_1, \dots, y_n$ of a part with reference $r$. mfgQC
computes:

$$
\bar y = \frac{1}{n}\sum_i y_i, \qquad
s = \sqrt{\frac{1}{n-1}\sum_i (y_i - \bar y)^2}, \qquad
\text{bias} = \bar y - r .
$$

It then runs a one-sample $t$-test of the null hypothesis that the bias is zero:

$$
\text{SE} = \frac{s}{\sqrt n}, \qquad
t = \frac{\text{bias}}{\text{SE}}, \qquad
\text{df} = n - 1 ,
$$

with a two-sided $p$-value from Student's $t$. The confidence interval on the bias at
confidence $1-\alpha$ (default $\alpha = 0.05$, so 95%) is

$$
\text{bias} \pm t_{1-\alpha/2,\, n-1}\,\text{SE} .
$$

The verdict is **acceptable** when $0$ lies inside that interval (the gage could be
unbiased, and you cannot reject it at $\alpha$), and **not acceptable** when $0$ lies
outside it.

A note on degrees of freedom. mfgQC uses the standard $n-1$. The AIAG manual prints a
$d_2^*$-effective df of 1.993 for its worked example, but that value is inconsistent with
the interval AIAG itself publishes: df = 1.993 implies $t_{.975} = 4.32$ and a much wider
interval than the one printed. The $n-1$ df is what reproduces AIAG's stated $t$ and CI,
so mfgQC uses it.

### Assumptions and what mfgQC checks

The $t$-test assumes the $n$ readings are independent and approximately normal. mfgQC
checks normality with the Anderson-Darling test and attaches the result as an
[assumption check](../guide/assumption-report.ipynb). Following the guardrail design, a
failed check is reported with a recommendation; it does not change the bias, the $t$, or
the verdict. Independence (taking the readings in a way that does not let one reading
influence the next) is on you; mfgQC does not test it.

### Worked example

This is the AIAG MSA 4th ed. bias example (Table III-B.3): reference 6.01, 100 readings
with mean 6.021 and standard deviation 0.2048. The $t$ and the interval depend only on the
bias, the standard deviation, and $n$, so a sample matched to that mean and standard
deviation reproduces AIAG's published numbers.

```python
import numpy as np, pandas as pd, mfgqc

rng = np.random.default_rng(1)
z = rng.normal(0, 1, 100)
z = (z - z.mean()) / z.std(ddof=1)          # standardize, then scale to AIAG's moments
vals = z * 0.2048 + 6.021
qc = mfgqc.load(pd.DataFrame({"m": vals}), measure="m")

b = qc.bias_study(reference=6.01)
print(b.report())
```

```text
MSA Bias Study
==============
n = 100   reference = 6.01   mean = 6.021
bias = +0.011   (sigma_repeat = 0.2048)
t = 0.537   p = 0.592   df = 99
95% CI on bias: (-0.02964, +0.05164)
Verdict: bias is acceptable (0 within the CI).

Assumption checks:
  [PASS] normality (Anderson-Darling): AD=0.671, p=0.0773; skew 0.403; n=100
```

The bias is +0.011, $t = 0.537$, and the 95% interval $(-0.030, +0.052)$ straddles zero,
so the gage's centering cannot be rejected. These reproduce AIAG's published $t = 0.537$
and CI exactly.

The structured fields are on `b.summary()`:

```python
>>> b.summary()
{'reference': 6.01, 'mean': 6.021, 'bias': 0.011, 'sigma_repeat': 0.2048,
 't': 0.5371, 'p_value': 0.5924, 'CI_low': -0.02964, 'CI_high': 0.05164,
 'df': 99, 'n': 100, 'verdict': 'acceptable', 'confidence': 95}
```

(Values shown rounded.) Build reports and dashboards from `summary()` or `to_dict()`,
never by parsing the `report()` text.

---

## 2. Linearity study

### What it is and when to use it

Linearity asks whether the bias is the same everywhere in the gage's operating range. A
gage might read true near the low end of its scale and drift high near the top. To detect
that, you measure several master parts spanning the range, each with its own known
reference, and you look at how the bias changes as the reference value grows. Use a
linearity study when your gage is used across a wide measurement range and you want to know
whether the bias is constant or reference-dependent.

### The procedure

For every measurement, mfgQC forms the bias at that point, $\text{bias}_i = y_i - r_i$,
where $r_i$ is the reference for the part being measured. It then fits a straight line of
bias on reference by ordinary least squares over all measurements:

$$
\text{bias} = (\text{slope})\cdot r + \text{intercept}.
$$

The slope is the linearity. A nonzero slope means the bias changes across the range. mfgQC
tests two hypotheses, $H_0:\text{slope}=0$ and $H_0:\text{intercept}=0$, each with a
two-sided $t$-test on $\text{df} = n - 2$:

$$
s^2 = \frac{\sum_i (\text{bias}_i - \widehat{\text{bias}}_i)^2}{n-2}, \qquad
\text{SE}_\text{slope} = \frac{s}{\sqrt{S_{xx}}}, \qquad
\text{SE}_\text{intercept} = s\sqrt{\frac{1}{n} + \frac{\bar x^2}{S_{xx}}},
$$

where $x$ is the reference, $\bar x$ its mean, and $S_{xx} = \sum_i (x_i - \bar x)^2$. The
$t$ statistics are slope/$\text{SE}_\text{slope}$ and intercept/$\text{SE}_\text{intercept}$.
mfgQC also reports $R^2$ of the fit and the mean bias at each distinct reference value.

The verdict is **acceptable** when neither slope nor intercept is significant at $\alpha$
(default 0.05), which is the case where the bias = 0 line lies within the regression's
confidence bands. It is **not acceptable** when either is rejected, meaning the bias varies
across the range or is nonzero overall.

### Assumptions and what mfgQC checks

The regression assumes the residuals are independent and approximately normal with constant
variance. mfgQC runs an Anderson-Darling normality check on the residuals and attaches it.
As with the bias study, a failed check is reported with a recommendation and does not alter
the fitted statistics or the verdict. Constant variance and independence across the range
are not tested.

### Worked example

This is AIAG MSA 4th ed.'s deliberately failing linearity example (Table III-B.4): five
reference parts at 2, 4, 6, 8, 10, each measured 12 times.

```python
import pandas as pd, mfgqc

AIAG = {
    2.00:[2.70,2.50,2.40,2.50,2.70,2.30,2.50,2.50,2.40,2.40,2.60,2.40],
    4.00:[5.10,3.90,4.20,5.00,3.80,3.90,3.90,3.90,3.90,4.00,4.10,3.80],
    6.00:[5.80,5.70,5.90,5.90,6.00,6.10,6.00,6.10,6.40,6.30,6.00,6.10],
    8.00:[7.60,7.70,7.80,7.70,7.80,7.80,7.80,7.70,7.80,7.50,7.60,7.70],
    10.00:[9.10,9.30,9.50,9.30,9.40,9.50,9.50,9.50,9.60,9.20,9.30,9.40],
}
rows = [{"ref": r, "m": v} for r, vals in AIAG.items() for v in vals]
qc = mfgqc.load(pd.DataFrame(rows), measure="m")

lin = qc.linearity_study(reference="ref")
print(lin.report())
```

```text
MSA Linearity Study
===================
n = 60   references = 5   df = 58
bias = -0.1317 * ref + +0.7367   (R^2 = 0.7143)
slope:     -0.1317  (t = -12.043, p = 2.04e-17)
intercept: +0.7367  (t = 10.158, p = 1.73e-14)
Verdict: linearity is not acceptable.
  (slope and/or intercept != 0 -> bias varies across the range)

Assumption checks:
  [FAIL] normality (Anderson-Darling): AD=1.37, p=0.0014; skew 1.29; n=60

Recommendations:
  - Data are not normal (AD=1.37, p=0.0014); the appropriate remedy depends on the analysis (transform, non-parametric test, or a non-normal capability method).
```

The slope is $-0.13$ bias units per reference unit and is strongly significant
($t = -12.0$), so the gage reads progressively low as the reference rises. The verdict is
not acceptable. These reproduce AIAG's published slope, intercept, $t$ statistics, and
$R^2 = 0.714$. Note the normality check on the residuals fails and is reported as a
recommendation; it does not change the linearity verdict.

The `reference` argument is either a column name (as above) or a `{group: ref}` mapping
keyed on the `part` or `subgroup` role, for the case where the reference lives in a lookup
rather than a column.

---

## 3. Stability study

### What it is and when to use it

Stability is consistency over time. You take one master part and measure it on a schedule:
once a shift, once a day, once a week. If the measurement system is stable, those readings
behave like a process that is in statistical control. If it drifts, jumps, or develops a
trend, the system is changing and the gage needs attention. Use a stability study to
monitor a gage between formal calibrations.

### The procedure

mfgQC puts the over-time readings on a control chart and checks for special-cause signals.
This is a thin specialization of [control charts](control-charts.md): "stable" means the
chart shows no out-of-control points under the chosen rule set.

The common case is one master part measured once per period, with no subgroup structure.
When the data carries no `subgroup` role, no `time` role, and no `subgroup_size`, the study
defaults to an **individuals (I-MR) chart** over the row sequence rather than raising an
error. If you measure subgroups instead (several readings per period), declare a `subgroup`
role or a `subgroup_size` and mfgQC charts the subgroups. The rule set defaults to
`rules="nelson"` (the eight [Nelson rules](run-rules.md)); pass `rules=` to change it and
`kind=` to force a specific chart.

The result reports the chart, the number of out-of-control signals, and the verdict:
**stable** when the signal count is zero, **unstable** otherwise. When unstable, the
report lists the violations.

```python
import numpy as np, pandas as pd, mfgqc

# one master part measured 30 times in order; no subgroup structure -> I-MR
rng = np.random.default_rng(0)
qc = mfgqc.load(pd.DataFrame({"m": rng.normal(50, 1, 30)}), measure="m")

st = qc.stability_study()
print(st.summary())     # {'chart_kind': 'i_mr', 'n_signals': ..., 'stable': ..., 'verdict': ...}
```

### Assumptions and what mfgQC checks

The stability verdict inherits the assumptions of the underlying control chart, and the
chart's assumption checks pass through onto the stability result. A control chart assumes
the in-control measurements are independent and, for the individuals chart, approximately
normal. mfgQC reports the chart's checks; it does not act on them. The reference part is
assumed to be itself stable over the study, so that any signal you see is the gage and not
the part.

---

## 4. Attribute agreement

### What it is and when to use it

Sometimes the "measurement" is a judgment, not a number: pass or fail, good or scrap, an
ordinal grade. The gage is a person, or a person plus a go/no-go fixture. Variance-based
gage R&R does not apply. Instead you ask whether appraisers agree. An attribute agreement
study has appraisers rate the same items over repeated trials, and reports three things:

- **Within-appraiser agreement** (repeatability): does each appraiser give the same call to
  the same item across their own trials?
- **Between-appraiser agreement** (reproducibility): do the appraisers agree with each
  other?
- **Versus reference** (accuracy): does each appraiser agree with the known correct call,
  when a reference standard is supplied?

Use it whenever the inspection is a categorical human call and you need to know whether the
call is consistent and correct.

### The procedure

mfgQC reports two kinds of number for each comparison: a **percent agreement** and a
**kappa**. Percent agreement is the fraction of items on which the relevant ratings all
match. Kappa corrects agreement for what you would expect by chance.

**Within-appraiser.** For each appraiser, mfgQC takes that appraiser's repeated trials on
each part and computes the fraction of parts where all of their own trials agree, plus a
Fleiss kappa across the trial replicates.

**Between-appraiser.** Each appraiser's per-part consensus rating is the mode of their
trials. mfgQC computes the fraction of parts where all appraisers' consensus ratings agree.
For the kappa it uses **Cohen's kappa** when there are exactly two appraisers and
**Fleiss' kappa** when there are three or more. The report labels which method was used.

**Versus reference.** When you pass a `reference` (a column or a `{part: ref}` map), mfgQC
compares each appraiser's consensus against the standard with percent agreement and Cohen's
kappa.

**Weighted kappa.** Yes, it is implemented. When you pass `ordinal=True`, the kappa
calculations use **linearly weighted** kappa, which gives partial credit for near-misses
between adjacent ordinal categories instead of scoring every disagreement equally. The
weighting is wired into `cohen_kappa(..., weights="linear")`; the `cohen_kappa` function
also supports `weights="quadratic"` directly, though the attribute study uses linear
weights for ordinal data. Binary studies (`ordinal=False`, the default) use unweighted
kappa.

The kappa formulas are the standard ones. For Cohen's kappa with observed agreement
$p_o$ and chance agreement $p_e$ from the marginals,

$$
\kappa = \frac{p_o - p_e}{1 - p_e}.
$$

Fleiss' kappa generalizes this to many raters per item using the per-item agreement
proportions. Percent-agreement confidence intervals use the Wilson score interval.

### How kappa is interpreted

mfgQC labels each kappa with the **Landis-Koch** band, but it treats those bands as context,
not as the verdict:

| kappa | band |
|---|---|
| below 0.20 | slight |
| 0.20 to 0.40 | fair |
| 0.40 to 0.60 | moderate |
| 0.60 to 0.80 | substantial |
| 0.80 and above | almost perfect |

(Values below 0 are reported as "poor".) The binary adequacy flag is **not** the kappa
band. It is percent agreement against a threshold of 90% (an AIAG-style acceptability
level). The reasoning is the kappa paradox: when one category dominates the ratings, a
system that agrees almost all of the time can still post a low kappa, because chance
agreement is already high. mfgQC flags that situation (high percent agreement, kappa below
0.6, and one category making up more than 85% of ratings) with a recommendation to judge
the system on agreement rather than kappa alone, rather than silently failing an agreeing
system.

### Assumptions and what mfgQC checks

The kappa statistics assume the items are independent and rated under a fixed operational
definition. mfgQC's structured checks here are not normality tests; they are two adequacy
flags attached to the result:

- **agreement**: passes when between-appraiser percent agreement is at least 90%; otherwise
  it carries a recommendation that the rating system is not reproducible enough, and to
  retrain or refine the operational definition.
- **kappa_marginal_skew**: the kappa-paradox flag described above.

These are reported, not enforced.

### Worked example

A synthetic crossed study: 30 parts, 3 appraisers, 3 trials each, binary pass/fail, with a
known reference call per part and a small flip rate.

```python
import numpy as np, pandas as pd, mfgqc

rng = np.random.default_rng(0)
n_parts, appraisers, trials, p_flip = 30, ("A", "B", "C"), 3, 0.05
truth = rng.integers(0, 2, n_parts)
rows = []
for p in range(n_parts):
    for a in appraisers:
        for t in range(trials):
            r = truth[p] if rng.random() > p_flip else 1 - truth[p]
            rows.append({"part": p, "appraiser": a, "trial": t,
                         "y": int(r), "ref": int(truth[p])})
df = pd.DataFrame(rows)

res = mfgqc.load(df, measure="y").attribute_agreement(
    rating="y", part="part", appraiser="appraiser", reference="ref")
print(res.report())
```

```text
Attribute agreement (binary): 30 parts x 3 appraisers x 3 trials
================================================================
appraiser     within %  within k          band
A                90.0%     0.852almost perfect
B                86.7%     0.806almost perfect
C                96.7%     0.952almost perfect

between appraisers (Fleiss): 96.7% all-agree, kappa = 0.951 (almost perfect)

appraiser     vs ref %  vs ref k          band
A               100.0%     1.000almost perfect
B               100.0%     1.000almost perfect
C                96.7%     0.927almost perfect

kappa bands are Landis-Koch context, not the verdict; the adequacy flag is percent-agreement vs 90%.

Assumption checks:
  [PASS] agreement (percent agreement vs threshold): percent=0.967; n=30
  [PASS] kappa_marginal_skew (category prevalence): category=0.648; n=30
```

Between-appraiser agreement is 96.7% with Fleiss kappa 0.951, so the three appraisers
reproduce each other's calls. There are three appraisers, so the between-appraiser kappa
uses Fleiss; with two appraisers it would report Cohen. Each appraiser's accuracy against
the reference is at or near 100%. Both adequacy flags pass.

As with every result, build downstream reports from `res.summary()` or `res.to_dict()`,
which expose `between_pct`, `between_kappa`, and the per-appraiser within and reference
figures as flat scalars.

## Source standard

The bias, linearity, stability, and attribute methods on this page follow:

> **AIAG.** *Measurement Systems Analysis (MSA) Reference Manual,* 4th ed. Automotive
> Industry Action Group, 2010.

See the [bibliography](bibliography.md). The bias and linearity worked examples reproduce
AIAG Tables III-B.3 and III-B.4. The kappa interpretation bands are Landis & Koch (1977),
which the AIAG manual also references. The Wilson score interval used for percent-agreement
confidence is a standard result, not from AIAG.

## See also

- [Gage R&R](gage-rr.md): the repeatability and reproducibility variance study.
- [Control charts](control-charts.md): the chart the stability study is built on.
- [Reading the assumption report](../guide/assumption-report.ipynb): what the guardrail checks mean.
- [API reference](api.md): `QCData.bias_study`, `linearity_study`, `stability_study`, `attribute_agreement`.
- [Bibliography](bibliography.md): the AIAG MSA reference.
