# Gage R&R

Gage repeatability and reproducibility (gage R&R) asks one question: **how much of the
variation you see is the part, and how much is the measurement system?** mfgQC's default
method is the **crossed ANOVA** decomposition from the AIAG MSA manual. Unlike the
older Average-and-Range method, it estimates the part×operator interaction explicitly and
pools it into error only when it is not significant.

This page documents what `gage_rr(method="anova")` actually computes, formula by formula,
pinned to the implementation in `mfgqc/gage_rr.py`. For the end-to-end study workflow
(planning the design, attaching roles, reading the verdict) see the
[Gage R&R workflow](../guide/gage-rr.ipynb).

---

## 1. The model and its decomposition

The crossed study has $p$ **parts** (random), $o$ **operators** (random), and $r$
**replicate** trials in every part×operator cell. The two-way random-effects model is

$$
y_{ijk} = \mu + P_i + O_j + (PO)_{ij} + \varepsilon_{ijk},
\qquad i=1\dots p,\; j=1\dots o,\; k=1\dots r .
$$

### Sums of squares and the ANOVA table

mfgQC builds a textbook balanced two-way ANOVA (`_anova`). With $\bar y$ the grand mean,
$\bar y_{i\cdot\cdot}$ the part means, $\bar y_{\cdot j\cdot}$ the operator means, and
$\bar y_{ij\cdot}$ the cell means:

| Source | Sum of squares (code) | df |
|---|---|---|
| Parts | $SS_P = or\sum_i(\bar y_{i\cdot\cdot}-\bar y)^2$ | $p-1$ |
| Operators | $SS_O = pr\sum_j(\bar y_{\cdot j\cdot}-\bar y)^2$ | $o-1$ |
| Part×Operator | $SS_{PO} = SS_\text{cells}-SS_P-SS_O$ | $(p-1)(o-1)$ |
| Equipment (error) | $SS_E = SS_\text{total}-SS_\text{cells}$ | $po(r-1)$ |
| Total | $SS_\text{total}=\sum(y_{ijk}-\bar y)^2$ | $por-1$ |

where $SS_\text{cells}=r\sum_{ij}(\bar y_{ij\cdot}-\bar y)^2$. Each mean square is
$MS = SS/df$. The F-ratios are taken **against the equipment (error) mean square**
($F=MS_\text{source}/MS_E$), and the interaction p-value is
$p_\text{int}=\Pr\!\big(F_{(p-1)(o-1),\,po(r-1)} > F_{PO}\big)$ (scipy `f.sf`).

### Variance components from the expected mean squares

The components are solved from the expected mean squares of the random model. **Which
solution is used depends on the pooling decision** (see [§2](#2-the-interaction-pooling-rule)).

=== "Interaction pooled (default when not significant)"

    A pooled error mean square is formed and the no-interaction expected mean squares apply:

    $$
    MS_\text{err} = \frac{SS_{PO}+SS_E}{(p-1)(o-1)+po(r-1)}
    $$

    $$
    \sigma^2_\text{repeat}=MS_\text{err},\quad
    \sigma^2_\text{oper}=\frac{MS_O-MS_\text{err}}{pr},\quad
    \sigma^2_\text{part}=\frac{MS_P-MS_\text{err}}{or},\quad
    \sigma^2_\text{int}=0 .
    $$

=== "Interaction retained (significant)"

    The full model uses $MS_E$ for repeatability and $MS_{PO}$ for the interaction:

    $$
    \sigma^2_\text{repeat}=MS_E,\quad
    \sigma^2_\text{int}=\frac{MS_{PO}-MS_E}{r},\quad
    \sigma^2_\text{oper}=\frac{MS_O-MS_{PO}}{pr},\quad
    \sigma^2_\text{part}=\frac{MS_P-MS_{PO}}{or}.
    $$

Every component is floored at zero (`max(..., 0.0)`). Negative variance estimates from
the method-of-moments solution are reported as exactly $0$, never as a negative number.

### Rolling the components up

The reported quantities are **standard deviations** (the square roots), assembled exactly
as the code does in `compute`:

$$
\begin{aligned}
\text{EV (repeatability)} &= \sigma_\text{repeat} \\
\text{AV (reproducibility)} &= \sqrt{\sigma^2_\text{oper} + \sigma^2_\text{int}}
   \;=\; \sigma_\text{oper}\ \text{(pooled case, } \sigma^2_\text{int}=0) \\
\text{GRR} &= \sqrt{\sigma^2_\text{repeat}+\sigma^2_\text{oper}+\sigma^2_\text{int}}
   \;=\;\sqrt{\text{EV}^2+\text{AV}^2} \\
\text{PV (part variation)} &= \sigma_\text{part} \\
\text{TV (total variation)} &= \sqrt{\text{GRR}^2 + \text{PV}^2}
\end{aligned}
$$

!!! note "Reproducibility folds the interaction in"
    mfgQC defines $\text{AV}^2 = \sigma^2_\text{oper}+\sigma^2_\text{int}$, so the operator
    *and* the part×operator interaction both count as reproducibility, the appraiser's
    contribution to spread. In the pooled (default) case $\sigma^2_\text{int}=0$ and AV is
    just the operator term. This matches the AIAG convention that GRR $=\sqrt{\text{EV}^2+\text{AV}^2}$.

---

## 2. The interaction-pooling rule

mfgQC follows the AIAG MSA 4th ed. convention: pool the part×operator interaction into
error when its F-test is **not significant at $\alpha = 0.25$** (that is, when
$p > 0.25$) and retain it otherwise.

```python
_INTERACTION_POOL_ALPHA = 0.25
...
pooled = p_int > _INTERACTION_POOL_ALPHA   # pool when interaction is NOT significant (p > 0.25)
```

!!! note "Why 0.25 and not 0.05"
    AIAG uses a deliberately permissive $\alpha = 0.25$ rather than the conventional 5%.
    The interaction term, when real, inflates reproducibility (AV); pooling a genuine but
    weakly-powered interaction into error would understate the appraiser's contribution.
    The high $\alpha$ retains the interaction unless the data give fairly strong evidence
    it is absent ($p > 0.25$), guarding against that understatement.

The `alpha` parameter of `gage_rr()` is **not** the pooling threshold. It controls the
confidence level of the component intervals only (see [§4](#4-confidence-intervals)). The
pooling cutoff is the fixed constant `_INTERACTION_POOL_ALPHA = 0.25`.

Whether pooling happened is reported in the text (`interaction pooled into error
(not significant)` vs `retained (significant)`) and exposed as `result.pooled`.

---

## 3. The reported metrics

Let $\sigma_c$ be the standard deviation of component $c$ (EV, AV, GRR, PV) and
$\sigma_\text{TV}$ the total. mfgQC reports three families of percentages.

### %study variation, a ratio of standard deviations

$$
\%\text{study}_c = 100\cdot\frac{\sigma_c}{\sigma_\text{TV}}
$$

This is a ratio of **standard deviations**, not variances. The AIAG study-variation
multiplier (commonly $6\sigma$ or $5.15\sigma$) is applied to *every* component including
the denominator TV, so it **cancels**: the code computes the ratio directly
(`100.0 * sd / tv`). The percentages therefore do **not** add to 100% (standard deviations
don't add in quadrature).

### %contribution, a ratio of variances

$$
\%\text{contribution}_c = 100\cdot\frac{\sigma_c^2}{\sigma^2_\text{TV}}
$$

Because variances *do* add, the %contribution column **does** sum to 100% (EV + AV(+INT) +
PV). This is the column to use when you want shares that partition the total.

### %tolerance, GRR against the spec window

When a two-sided spec is attached (`tol = upper − lower`), each component is also expressed
against the engineering tolerance, using a **fixed $6\sigma$ multiplier**:

$$
\%\text{tolerance}_c = 100\cdot\frac{6\,\sigma_c}{\text{USL}-\text{LSL}}
$$

`pct_tol` is `None` when no two-sided spec is present. The headline number practitioners
quote, `%tolerance (GRR)`, is the GRR entry of this dict.

### Number of distinct categories (ndc)

$$
\text{ndc} = \Big\lfloor 1.41\cdot\frac{\text{PV}}{\text{GRR}}\Big\rfloor
$$

The factor $1.41\approx\sqrt 2$, the ratio is **truncated** to an integer (`int(...)`,
toward zero), and the result is floored at 1 (`max(1, ...)`); if GRR $\le 0$ it returns 0.
ndc estimates how many non-overlapping part groups the gage can reliably tell apart.

---

## 4. Confidence intervals

For the ANOVA method, mfgQC attaches component confidence limits (EV, AV, GRR, PV) on the
**standard-deviation scale**, computed by the Modified Large Sample (MLS / Burdick–Larsen)
method that reproduces AIAG MSA Table III-B.9. The confidence level is
$100(1-\alpha)\%$, and **`alpha` defaults to `0.10`** (i.e. **90% limits**), the AIAG
convention. (Note this differs from capability's 95% default.) The `xbar_r` method reports
point estimates only.

---

## 5. The verdict

mfgQC renders a single AIAG-style acceptability verdict from %study(GRR) and ndc
(`_verdict`):

```python
def _verdict(pct_grr, ndc):
    if pct_grr < 10 and ndc >= 5:
        return "acceptable"
    if pct_grr > 30 or ndc < 2:
        return "unacceptable"
    return "marginal (conditionally acceptable)"
```

| %study(GRR) | ndc | Verdict string |
|---|---|---|
| $< 10\%$ | $\ge 5$ | `acceptable` |
| $> 30\%$ **or** ndc $< 2$ | n/a | `unacceptable` |
| anything else | n/a | `marginal (conditionally acceptable)` |

!!! note "How this maps to AIAG canon"
    AIAG's headline %GRR bands are: **< 10% acceptable, 10–30% conditionally
    acceptable, > 30% unacceptable**, with **ndc ≥ 5 desirable**. mfgQC encodes those
    bands but **couples ndc into the verdict**: a system with %GRR < 10% but ndc < 5 is
    reported as *marginal*, not *acceptable*; and ndc < 2 alone forces *unacceptable*.
    The standalone ndc adequacy check (`ndc_adequacy`, AIAG ndc ≥ 5) is also reported as a
    separate assumption so you can see the ndc shortfall even when %GRR is fine.

---

## 6. Assumptions checked

The ANOVA method populates these structured
[assumption checks](../guide/assumption-report.ipynb) and *reports* them. It never silently changes the analysis:

- **Normality of residuals** (Anderson–Darling) on the cell-centered residuals
  $y_{ijk}-\bar y_{ij\cdot}$.
- **Homogeneity of variance across operators** (Levene): equal measurement spread per
  appraiser.
- **ndc adequacy**: passes when ndc $\ge 5$ (`NDC_MIN`); the failing recommendation tells
  you the gage cannot reliably distinguish parts.

Two further assumptions are structural to the design and enforced as hard errors, not soft
checks:

!!! warning "Balanced crossed design required"
    The study must be **balanced** (every part×operator cell must have the **same number
    of trials**) and have **$r\ge 2$** replicates. An unbalanced design raises a
    `ValueError` that names the trials-per-part for each operator; $r<2$ also raises. The
    `part`, `operator`, and `replicate` roles must all be attached.

Beyond what the code can test, the study is only meaningful if the **parts span the actual
process range** (parts are a random sample of production, not hand-picked) and operators
are representative appraisers, assumptions of the random-effects model that no software
can verify for you.

---

## 7. Worked example

The canonical AIAG MSA 4th-ed. study: **10 parts × 3 operators × 3 trials**. Here we attach
a spec of $[-3, 3]$ so the %tolerance line appears.

```python
import pandas as pd
import mfgqc

# AIAG MSA 4th ed. data: (operator, trial) -> measurement for parts 1..10
_AIAG = {
    ("A", 1): [0.29, -0.56, 1.34, 0.47, -0.80, 0.02, 0.59, -0.31, 2.26, -1.36],
    ("A", 2): [0.41, -0.68, 1.17, 0.50, -0.92, -0.11, 0.75, -0.20, 1.99, -1.25],
    ("A", 3): [0.64, -0.58, 1.27, 0.64, -0.84, -0.21, 0.66, -0.17, 2.01, -1.31],
    ("B", 1): [0.08, -0.47, 1.19, 0.01, -0.56, -0.20, 0.47, -0.63, 1.80, -1.68],
    ("B", 2): [0.25, -1.22, 0.94, 1.03, -1.20, 0.22, 0.55, 0.08, 2.12, -1.62],
    ("B", 3): [0.07, -0.68, 1.34, 0.20, -1.28, 0.06, 0.83, -0.34, 2.19, -1.50],
    ("C", 1): [0.04, -1.38, 0.88, 0.14, -1.46, -0.29, 0.02, -0.46, 1.77, -1.49],
    ("C", 2): [-0.11, -1.13, 1.09, 0.20, -1.07, -0.67, 0.01, -0.56, 1.45, -1.77],
    ("C", 3): [-0.15, -0.96, 0.67, 0.11, -1.45, -0.49, 0.21, -0.49, 1.87, -2.16],
}
rows = [
    {"part": p, "operator": op, "trial": t, "y": v}
    for (op, t), parts in _AIAG.items()
    for p, v in enumerate(parts, start=1)
]
df = pd.DataFrame(rows)

qc = (mfgqc.load(df, measure="y",
                 roles={"part": "part", "operator": "operator", "replicate": "trial"})
          .spec(lower=-3.0, upper=3.0))

print(qc.gage_rr().report())
```

```text
Gage R&R (method=anova)
=======================
Design: 10 parts x 3 operators x 3 trials
Verdict: marginal (conditionally acceptable)   ndc = 4

component          std dev   lower 90%   upper 90%  %study var   %contrib
Repeatability(EV)   0.19993       0.177       0.231      18.42%      3.39%
Reproducib.(AV)    0.22684       0.128       1.014      20.90%      4.37%
GRR                0.30237       0.235       1.033      27.86%      7.76%
Part(PV)            1.0423       0.759       1.717      96.04%     92.24%
Total(TV)           1.0853                             100.00%    100.00%

interaction pooled into error (not significant)

%tolerance (GRR) = 30.24%

Assumption checks:
  [PASS] normality (Anderson-Darling): AD=0.64, p=0.0924; skew 0.386; n=90
  [PASS] homogeneity_of_variance (Levene): variance ratio 1.16, p=0.994; n=90
  [FAIL] ndc_adequacy (ndc (AIAG ndc>=5)): ndc 4; n=90

Recommendations:
  - ndc = 4 (< 5 AIAG): the measurement system cannot reliably distinguish parts. Improve gage resolution/repeatability.
```

Reading the verdict:

- **Interaction pooled.** The interaction p-value here is $\approx 0.974$, far above the
  AIAG $0.25$ cutoff, so the part×operator term is folded into error and AV is the operator
  term alone.
- **%GRR = 27.86%** of study variation, inside the 10–30% band, so *conditionally
  acceptable*, **not** good enough to wave through.
- **ndc = 4** ($\lfloor 1.41\cdot 1.0423/0.30237\rfloor = \lfloor 4.86\rfloor = 4$), below
  the AIAG target of 5. The gage struggles to resolve the parts, and the `ndc_adequacy`
  check fails.
- **PV dominates** (96% of study variation; 92% of variance contribution): most of what
  the study sees is genuine part-to-part difference, which is the healthy part of this
  result.

The headline numbers are available as a flat dict via `.summary()`. Consume that or
`.to_dict()` from code; never parse `.report()` text.

```python
g = qc.gage_rr()
g.summary()["pct_study_GRR"]   # 27.860650881142313
g.summary()["ndc"]             # 4
g.summary()["verdict"]         # 'marginal (conditionally acceptable)'
g.pooled                       # True
```

---

## Source standard

The ANOVA decomposition, the variance-component expected-mean-square solution, the
%study / %contribution / %tolerance metrics, the ndc formula, the component confidence
limits (Table III-B.9), and the acceptability bands all follow:

> **AIAG.** *Measurement Systems Analysis (MSA) Reference Manual,* 4th ed. Automotive
> Industry Action Group, 2010.

See the [bibliography](bibliography.md). The interaction-pooling threshold follows AIAG's
$p>0.25$ convention (see [§2](#2-the-interaction-pooling-rule)). One documented refinement
remains: the verdict **couples ndc into the %GRR acceptability bands** (a system with
%GRR < 10% but ndc < 5 is reported *marginal*, not *acceptable*), pinned to the
implementation in [§5](#5-the-verdict).

## See also

- [Gage R&R workflow](../guide/gage-rr.ipynb): running a study end to end.
- [Reading the assumption report](../guide/assumption-report.ipynb): what the guardrail checks mean.
- [Bibliography](bibliography.md): the AIAG MSA reference.
