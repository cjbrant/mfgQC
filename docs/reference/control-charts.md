# Control charts

This page specifies the **control-chart constants** mfgQC uses and the **limit
formulas** for every chart family it computes: variables (I-MR, X-bar R, X-bar S),
attributes (p, np, c, u), the time-weighted charts (EWMA, CUSUM), and the
standardized short-run chart. The run-rules engine that flags out-of-control
points has its own page: **[Run rules](run-rules.md)**.

Everything below is pinned to the implementation in `mfgqc/constants.py`
(`_CONTROL_TABLE`, `control_constant()`) and `mfgqc/control_charts.py`. Where the
code makes a choice a textbook might make differently, that choice is noted.

## How subgrouping works

A control chart needs to know which measurements form a rational subgroup, and the
within-subgroup sigma in [capability](capability.md) needs the same thing. You declare
that structure when you load the data, in one of two ways. mfgQC does not infer it from
the numbers on its own.

**By a grouping column (`subgroup=`).** Name a column whose values mark the subgroups:

```python
qc = mfgqc.load(df, measure="width", subgroup="lot")
```

Each distinct value of `lot` is one subgroup, taken in first-appearance order. The
subgroups may be unequal in size; each one holds however many rows carry that value. Use
this when your table already records which lot, batch, or shift each part came from.

**By a fixed size (`subgroup_size=k`).** With no grouping column, mfgQC chunks
*consecutive rows* into groups of `k`:

```python
qc = mfgqc.load(df, measure="width", subgroup_size=5)   # rows 1-5, then 6-10, and so on
```

The rows must be in production or time order for this to mean anything. `subgroup_size=1`
makes each row its own subgroup, which is the individuals case that routes to an I-MR
chart. If the row count is not a multiple of `k`, the leftover rows form a smaller final
subgroup.

**If you set both, the column wins.** A `subgroup` (or `time`) role takes precedence over
`subgroup_size`.

**If you set neither:**

- A control chart raises a catchable `MissingPrerequisiteError` asking for a `subgroup` or
  `time` role, or a `subgroup_size`. It does not guess a structure.
- Capability is more forgiving. With nothing to subgroup by, it uses the overall sigma for
  the within family as well, reports `sigma_used = "overall"`, and Cp/Cpk then equal
  Pp/Ppk.

This is implemented in `QCData.subgroups()` in `mfgqc/data.py`.

## The control-chart constants

mfgQC keeps the standard SQC factor table as **plain transcribed data**, not
values computed on the fly. That is deliberate: a practitioner can check every
number against the published table they already trust (Montgomery, *Introduction
to Statistical Quality Control*, Appendix VI). They are looked up by subgroup
size $n$ via `control_constant(name, n)` for $n = 2 \ldots 25$; outside that range
the function raises a `ValueError`.

| Constant | What it is | Used by |
| --- | --- | --- |
| $d_2$ | mean of the relative range $W = R/\sigma$; converts $\bar{R}$ to a $\sigma$ estimate via $\hat\sigma = \bar{R}/d_2$ | I-MR (with $d_2$ at $n=2$); within-$\sigma$ in [capability](capability.md) |
| $d_3$ | standard deviation of the relative range $W$ | dispersion-limit width (tabulated; reported, not used directly in the R-chart formula below) |
| $A_2$ | factor for X-bar limits from $\bar{R}$ | X-bar R location limits |
| $A_3$ | factor for X-bar limits from $\bar{S}$ | X-bar S location limits |
| $D_3, D_4$ | lower/upper factors for the range limits from $\bar{R}$ | R-chart limits |
| $B_3, B_4$ | lower/upper factors for the s limits from $\bar{S}$ | S-chart limits |
| $c_4$ | bias-correction for the sample standard deviation ($E[s] = c_4\,\sigma$) | unbiased $\sigma$ from $\bar{S}$ in [capability](capability.md) |

### Table excerpt ($n = 2 \ldots 10$)

These are the exact values in `_CONTROL_TABLE`, read straight from the code:

| $n$ | $d_2$ | $d_3$ | $A_2$ | $A_3$ | $D_3$ | $D_4$ | $B_3$ | $B_4$ | $c_4$ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.128 | 0.853 | 1.880 | 2.659 | 0.000 | 3.267 | 0.000 | 3.267 | 0.7979 |
| 3 | 1.693 | 0.888 | 1.023 | 1.954 | 0.000 | 2.574 | 0.000 | 2.568 | 0.8862 |
| 4 | 2.059 | 0.880 | 0.729 | 1.628 | 0.000 | 2.282 | 0.000 | 2.266 | 0.9213 |
| 5 | 2.326 | 0.864 | 0.577 | 1.427 | 0.000 | 2.114 | 0.000 | 2.089 | 0.9400 |
| 6 | 2.534 | 0.848 | 0.483 | 1.287 | 0.000 | 2.004 | 0.030 | 1.970 | 0.9515 |
| 7 | 2.704 | 0.833 | 0.419 | 1.182 | 0.076 | 1.924 | 0.118 | 1.882 | 0.9594 |
| 8 | 2.847 | 0.820 | 0.373 | 1.099 | 0.136 | 1.864 | 0.185 | 1.815 | 0.9650 |
| 9 | 2.970 | 0.808 | 0.337 | 1.032 | 0.184 | 1.816 | 0.239 | 1.761 | 0.9693 |
| 10 | 3.078 | 0.797 | 0.308 | 0.975 | 0.223 | 1.777 | 0.284 | 1.716 | 0.9727 |

The full table runs to $n = 25$. Note that $D_3$ and $B_3$ are $0$ below $n = 7$
and $n = 6$ respectively: for small subgroups the lower dispersion limit is
clamped to zero, which is what the table encodes.

!!! note "These are tabulated factors, not live computations"
    The constants are standard published values transcribed to 3–4 decimal
    places. mfgQC does **not** recompute them from $d_2 = E[W]$ integrals at run
    time. If you need a value to more decimals, or for an $n$ outside $2 \ldots
    25$, that is a deliberate boundary, not a bug.

## Variables charts

Variables charts track a continuous measurement. mfgQC computes the location
panel (the process center) and a dispersion panel (the spread) together, and runs
the [run rules](run-rules.md) on the location series. The dispersion panel is
additionally checked for the **beyond-limits rule only** (Rule 1), so a subgroup
whose range or s blows up is flagged even when its mean stays inside the location
limits.

### I-MR (individuals + moving range)

For one measurement per time point ($n = 1$). Spread is estimated from the
**moving range** of consecutive points, $MR_i = |x_i - x_{i-1}|$, and converted to
$\sigma$ using $d_2$ at $n = 2$, because each moving range spans two observations:

$$
\bar{MR} = \frac{1}{k-1}\sum_{i=2}^{k} MR_i, \qquad
\hat\sigma = \frac{\bar{MR}}{d_2}, \quad d_2 = 1.128
$$

| Panel | CL | UCL | LCL |
| --- | --- | --- | --- |
| Individuals (I) | $\bar{x}$ | $\bar{x} + 3\hat\sigma$ | $\bar{x} - 3\hat\sigma$ |
| Moving range (MR) | $\bar{MR}$ | $D_4\,\bar{MR}$, $\ D_4 = 3.267$ | $0$ |

The MR series is aligned to points $2 \ldots k$ (the first point has no moving
range and plots as `NaN`). `kind="i"` is an accepted alias for `"i_mr"`, and an
explicit I-MR chart on a plain measure column defaults each row to its own
size-1 subgroup rather than erroring.

### X-bar R (subgroup mean and range)

For constant subgroup size $2 \le n \le 10$. Spread per subgroup is the range
$R = \max - \min$; $\bar{R}$ is the mean range, $\bar{\bar{x}}$ the grand mean.

| Panel | CL | UCL | LCL |
| --- | --- | --- | --- |
| Mean (X-bar) | $\bar{\bar{x}}$ | $\bar{\bar{x}} + A_2\bar{R}$ | $\bar{\bar{x}} - A_2\bar{R}$ |
| Range (R) | $\bar{R}$ | $D_4\,\bar{R}$ | $D_3\,\bar{R}$ |

### X-bar S (subgroup mean and standard deviation)

Same structure as X-bar R but the per-subgroup spread is the sample standard
deviation $s$ (computed with `ddof=1`), and $\bar{S}$ is the mean of those.
This is the preferred variables chart for larger subgroups.

| Panel | CL | UCL | LCL |
| --- | --- | --- | --- |
| Mean (X-bar) | $\bar{\bar{x}}$ | $\bar{\bar{x}} + A_3\bar{S}$ | $\bar{\bar{x}} - A_3\bar{S}$ |
| Std dev (S) | $\bar{S}$ | $B_4\,\bar{S}$ | $B_3\,\bar{S}$ |

For the run-rules engine, the location-panel sigma is recovered from the limit
width itself, $\sigma_{\text{loc}} = (\text{UCL} - \text{CL})/3$, so the
zone tests use the same $\pm 3\sigma$ band the chart draws.

## Attributes charts

Attributes charts track counts or proportions of defectives/defects. All four use
**3σ limits** with the lower limit clamped at zero (`np.clip(cl - 3*sd, 0,
None)`). Each must be requested explicitly with `kind=`; they are never inferred.

The sample size for p/np/u comes from `n=`: a **column name** (per-point sizes,
which produce *stepped* limits when they vary), an **int** (constant size), or
`None` (falls back to a `size` role, else 1).

| Chart | Tracks | Point | CL | $3\sigma$ half-width |
| --- | --- | --- | --- | --- |
| **p** | proportion defective, variable $n$ | $p_i = c_i / n_i$ | $\bar{p} = \dfrac{\sum c_i}{\sum n_i}$ | $3\sqrt{\dfrac{\bar{p}(1-\bar{p})}{n_i}}$ |
| **np** | count defective, **constant $n$** | $c_i$ | $n\bar{p}$ | $3\sqrt{n\bar{p}(1-\bar{p})}$ |
| **c** | defects per unit, constant area | $c_i$ | $\bar{c}$ | $3\sqrt{\bar{c}}$ |
| **u** | defects per unit, variable area | $u_i = c_i / n_i$ | $\bar{u} = \dfrac{\sum c_i}{\sum n_i}$ | $3\sqrt{\dfrac{\bar{u}}{n_i}}$ |

For **p** and **u** the half-width depends on $n_i$, so the limits step up and
down with each point's sample size; a point is flagged only if it falls outside
**its own** limits. The **np** chart enforces constant $n$ (it raises if sizes
vary) because $n\bar{p}$ has no meaning otherwise. Attributes charts have a
location panel only; there is no dispersion panel.

Each attributes chart attaches an **over/under-dispersion check** (chi-square
against the binomial for p/np, against the Poisson for c/u). It reports the
dispersion ratio and warns; it does not switch models.

## Time-weighted and short-run charts

These exist for situations a Shewhart chart handles poorly. One line each; see
the source for full parameter detail.

- **EWMA** (`qc.ewma_chart(...)`, `mfgqc/timeseries_charts.py`): exponentially
  weighted moving average for detecting **small sustained shifts**. Recursion
  $z_i = \lambda x_i + (1-\lambda)z_{i-1}$ with time-varying limits
  $\mu_0 \pm L\sigma\sqrt{\frac{\lambda}{2-\lambda}\big(1-(1-\lambda)^{2i}\big)}$.
  Key parameters: smoothing $\lambda$ (default $0.1$) and limit width $L$ in
  sigma (default $2.7$).
- **CUSUM** (`qc.cusum_chart(...)`): tabular two-sided cumulative-sum chart for
  the same small-shift regime. Accumulates $C^+$ and $C^-$ deviations and signals
  when either exceeds the decision interval $H = h\sigma$. Key parameters:
  reference value $k$ (slack, default $0.5\sigma$) and $h$ (default $5\sigma$).
- **Short-run / standardized** (`qc.short_run_chart(by=...)`): pools several
  part numbers onto one chart by plotting the standardized deviation
  $z = (x - \text{target}_{\text{part}})/\sigma_{\text{part}}$ against fixed
  $\pm 3$ limits. `target` is a scalar, a `{part: target}` map, or `None` (each
  part's own mean). It checks homogeneity of within-part variance across the
  pooled parts.

!!! warning "EWMA and CUSUM assume a known in-control baseline"
    Both require an in-control $\mu_0$ and $\sigma$. mfgQC defaults $\mu_0$ to the
    spec target if set (else the sample mean) and $\sigma$ to the I-chart estimate
    $\bar{MR}/d_2$, and **records which source it used**. For a trustworthy chart,
    supply $\mu_0$ and $\sigma$ from a stable baseline study rather than letting
    them be estimated from the data you are charting.

## Chart-kind inference

When you call `qc.control_chart()` with **no `kind=`**, mfgQC infers a *variables*
chart from the subgroup sizes. The rule is exactly (`_infer_variable_kind`):

| Subgroup size $n$ | Inferred chart |
| --- | --- |
| all $n = 1$ | `i_mr` |
| $2 \le n \le 10$ | `xbar_r` |
| $n > 10$ | `xbar_s` |

The threshold between X-bar R and X-bar S is **$n \le 10$** (X-bar R) versus **$n > 10$**
(X-bar S). The inferred choice is recorded in the provenance history (`inferred:
true` in the step params) and named in the report. Attributes charts are **never
inferred**: you must pass `kind="p"`, `"np"`, `"c"`, or `"u"` explicitly.

!!! tip "Guardrail: no silent switching"
    Inference only fills in a *missing* `kind`. If you request a kind explicitly,
    mfgQC computes that kind or raises; it never quietly substitutes a different
    chart. For how to override the inferred choice, see
    **[Choosing a control chart](../guide/choosing-a-control-chart.ipynb)**.

## Assumptions and what mfgQC checks

| Assumption | Why it matters | Checked? |
| --- | --- | --- |
| **Rational subgrouping** | Subgroups must be formed so within-subgroup variation captures only common cause; this underlies the $\bar{R}/d_2$ and $\bar{S}$ sigma estimates. | **Not** checked; it is a sampling-design decision the engineer owns. |
| **Independence** | Autocorrelated points make Shewhart limits too tight and produce false alarms. | **Checked.** A lag-1 autocorrelation test (`check_independence`) runs on the location series; it reports $r_1$, a p-value, and, if it fails, recommends a time-series chart (EWMA). It self-flags `low_power` for $n < 30$. |
| **Approximate normality of subgroup means** | The $\pm 3\sigma$ limits assume the plotted statistic is roughly normal; for X-bar charts the CLT makes this mild even for moderately skewed data. | Not tested directly on the variables charts; relied on via the CLT. (Per-point normality of individuals on the I-MR chart is the stronger assumption; judge it with a [normality check](capability.md).) |
| **Binomial / Poisson counts** | p/np assume binomial counts; c/u assume Poisson defects. | **Checked.** A chi-square over/under-dispersion test runs on every attributes chart and reports the dispersion ratio. |

True to mfgQC's guardrail design, these checks **report and recommend; they
never silently switch methods**. An autocorrelation failure does not auto-convert
your X-bar R chart to an EWMA; it tells you to consider one.

## Worked example

A 20-subgroup, size-5 dataset. With no `kind=`, $n = 5$ falls in $2 \le n \le 10$,
so mfgQC infers X-bar R:

```python
import numpy as np, pandas as pd, mfgqc

rng = np.random.default_rng(7)
df = pd.DataFrame({
    "width": np.round(rng.normal(1.52, 0.12, size=100), 3),
    "lot":   np.repeat(np.arange(1, 21), 5),     # 20 subgroups of 5
})
qc = (mfgqc.load(df, measure="width", subgroup="lot", subgroup_size=5)
           .spec(lower=1.0, upper=2.0, target=1.5))

print(qc.control_chart())
```

```text
Control Chart: xbar_r (inferred); rules=nelson
==============================================
Xbar: CL=1.4992  UCL=1.6475  LCL=1.3509
R: CL=0.25705  UCL=0.5434  LCL=0

Out-of-control signals: none (process in control)

Assumption checks:
  [PASS] independence (lag-1 autocorrelation): r=0.193, p=0.387; n=20 [low power]
```

You can read the formulas straight off the numbers. With $\bar{R} = 0.25705$ and
$n = 5$ (so $A_2 = 0.577$, $D_4 = 2.114$, $D_3 = 0$):

- $\text{UCL}_{\bar{x}} = 1.4992 + 0.577 \times 0.25705 = 1.6475$
- $\text{UCL}_R = 2.114 \times 0.25705 = 0.5434$, $\ \text{LCL}_R = 0$

An **I-MR** chart on single observations, showing the dispersion-panel
beyond-limits flag and the $\bar{MR}/d_2$ sigma:

```python
rng = np.random.default_rng(3)
df = pd.DataFrame({"visc": np.round(rng.normal(50, 2, size=24), 2)})
qc = mfgqc.load(df, measure="visc")
print(qc.control_chart(kind="i_mr"))
```

```text
Control Chart: i_mr (specified); rules=nelson
=============================================
Individual: CL=49.913  UCL=56.915  LCL=42.912
MR: CL=2.6326  UCL=8.6007  LCL=0

Out-of-control signals: 1
  point 2 (dispersion): nelson_1 - one point beyond control limits

Assumption checks:
  [PASS] independence (lag-1 autocorrelation): r=-0.182, p=0.371; n=24 [low power]
```

Here $\hat\sigma = \bar{MR}/d_2 = 2.6326 / 1.128 = 2.334$, so the individuals
limits are $49.913 \pm 3 \times 2.334$, and the MR upper limit is
$D_4 \bar{MR} = 3.267 \times 2.6326 = 8.6007$. Point 2's moving range exceeds it.

A **p-chart** with a constant sample-size column:

```python
df = pd.DataFrame({"defs": [3,5,2,4,6,1,3,4,2,5], "n": [100]*10})
qc = mfgqc.load(df, measure="defs")
print(qc.control_chart(kind="p", n="n"))
```

```text
Control Chart: p (specified); rules=nelson
==========================================
Proportion (p): CL=0.035  UCL=0.090134  LCL=0

Out-of-control signals: none (process in control)

Assumption checks:
  [PASS] dispersion (chi-square dispersion): dispersion ratio 0.74, p=0.672; n=10 [low power]
```

$\bar{p} = 35/1000 = 0.035$; the half-width is
$3\sqrt{0.035 \times 0.965 / 100} = 0.0551$, giving $\text{UCL} = 0.0901$ and a
clamped $\text{LCL} = 0$.

## Source and standard

- **Constants table and variables/attributes limit formulas:** Montgomery,
  D. C., *Introduction to Statistical Quality Control* (Wiley). This is mfgQC's
  primary source for SPC. The constants follow the conventional SQC tables
  (Appendix VI). See the [Bibliography](bibliography.md). These are the same
  standard tabulated factors published in ASTM control-chart references; mfgQC
  cites Montgomery as the transcription source.
- The run-rules engine (Nelson / Western Electric) is documented separately on
  **[Run rules](run-rules.md)** and cites Nelson (1984) and the Western Electric
  *Statistical Quality Control Handbook* (1956).

## See also

- **[Run rules](run-rules.md)**: the zone tests that flag out-of-control points.
- **[Choosing a control chart](../guide/choosing-a-control-chart.ipynb)**: the
  inference rule in practice and how to override it.
- **[Capability](capability.md)**: where $d_2$, $c_4$, and the within-σ
  estimator reappear.
- **[Bibliography](bibliography.md)**: full citations.
