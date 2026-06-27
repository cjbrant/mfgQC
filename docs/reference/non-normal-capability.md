# Non-normal capability

The ordinary capability formulas (`method="normal"`) assume the measurement is
normally distributed: they convert a $\pm 3\sigma$ spread into a fraction defective
through the normal CDF. When the data are genuinely skewed or bounded (flatness,
roundness, runout, concentration, time-to-event), that assumption fails and the
normal $C_{pk}$ can be wildly optimistic (or pessimistic). mfgQC exposes four
non-normal methods for these cases.

This page documents exactly what each method does **as implemented** in
`mfgqc/capability.py`, when to use it, and, as important, when not to. For
the normal method, the σ families, and confidence intervals, see
[Capability](capability.md).

!!! warning "First, are you sure it's non-normal, or out of control?"
    A non-normal *shape* is not the same thing as an out-of-control *process*. A
    special cause (a tool change, a bad batch, a drifting setpoint) produces data
    that look non-normal because they are a mixture of two processes, and no
    transform fixes that. **Establish statistical control first** (see
    [Choosing a control chart](../guide/choosing-a-control-chart.ipynb)), confirm
    the non-normality is inherent to the in-control process, and only then reach
    for these methods. Transforming away a special cause buries the problem; it
    does not solve it.

## How mfgQC reports non-normality

The default `capability()` (`method="normal"`) **never transforms**. It runs the
normality check (Anderson–Darling) and, crucially, reports the **estimated effect
on the index**: the relative shift in $C_{pk}$ between the normal calculation and an
auto-fit non-normal distribution (`_cpk_shift` in the code). A failed normality
test with a large `est. Cpk impact` is your signal to switch methods, but mfgQC
**warns and recommends; it never silently switches**. You opt in by passing
`method=`. See [Reading the assumption report](../guide/assumption-report.ipynb).

!!! note "Non-normal methods report no confidence interval"
    Every non-normal method sets `cp_ci`/`cpk_ci` to `None`, and the report prints
    `CI: n/a (non-normal method)`. The CIs in mfgQC are normal-theory (chi-square
    for $C_p$, the approximate-normal interval for $C_{pk}$; see
    [Capability](capability.md)). Those derivations do not hold on transformed or
    percentile-fitted scales, so mfgQC reports **n/a** rather than print an interval
    it cannot stand behind. If you need uncertainty on a non-normal index, bootstrap
    it outside the library.

## Method selector

| Method | Use it when | Key assumption |
| --- | --- | --- |
| `boxcox` | Strictly positive data with a smooth skew that a power transform can straighten. | A single power $\lambda$ normalizes the data; spec limits transform cleanly. |
| `clements` / `percentile` | You want a distribution-fit percentile method and don't want to commit to one family up front. | One of {normal, lognormal, gamma, Weibull, exponential} fits the data well by log-likelihood. |
| `johnson` | Skew *and* heavy/light tails that a single power can't capture; you want a flexible translation system. | The Johnson $S_U$ family fits the data. |

All non-normal indices are reported as a **long-term (overall)** view: the code sets
$P_p^\ast = C_p^\ast$, etc. There is no separate within-subgroup short-term index for
these methods. `sigma (within)` and `sigma (overall)` are still printed from the
raw data for reference, but the indices come from the transform/fit, not from those
σ values.

## Box-Cox

Box & Cox (1964). Find the power $\lambda$ that best normalizes the data, transform
the data **and the spec limits** onto that scale, then compute ordinary capability
indices there.

The transform is the standard Box–Cox family:

$$
y^{(\lambda)} =
\begin{cases}
\dfrac{y^{\lambda} - 1}{\lambda}, & \lambda \neq 0 \\[2ex]
\ln y, & \lambda = 0
\end{cases}
$$

What the code does, step by step:

1. **Positivity shift.** Box–Cox requires $y > 0$. If $\min(y) \le 0$, the code adds
   a shift so the minimum becomes a tiny positive number (`shift = 1e-9 - min(y)`).
   The same shift is applied to the spec limits before transforming, so the
   transform is consistent across data and limits.
2. **Fit $\lambda$.** `scipy.stats.boxcox` chooses the $\lambda$ that maximizes the
   normal log-likelihood of the transformed data.
3. **Transform the spec limits** with the *same* $\lambda$ and shift via the internal
   `_bc(...)` function. This is the part that makes the method honest: a spec limit
   is a point on the measurement scale, so it must travel through the identical
   transform the data did.
4. **Compute indices on the transformed scale** using the ordinary
   $C_p = (\text{USL}'-\text{LSL}')/6\sigma'$, $C_{pk}=\min(C_{pu},C_{pl})$ formulas,
   where USL$'$, LSL$'$, $\mu'$, $\sigma'$ are all on the transformed scale and
   $\sigma'$ is the ordinary sample SD of the transformed data.

The fitted $\lambda$ is reported in the `Cp/Cpk sigma` line, e.g.
`box-cox (lambda=0.0829)`, and recorded in provenance as part of the transform
params.

!!! warning "When Box-Cox does *not* apply"
    Box–Cox needs strictly positive, continuously skewed data that a single power
    can straighten. It cannot help with bimodal data, hard physical bounds that
    create a spike at the limit, or data whose non-normality comes from a special
    cause. If the transform doesn't normalize the data, the recomputed index is no
    more trustworthy than the normal one.

## Clements percentile method

Clements (1989). A **percentile** method: instead of transforming, it locates the
0.135 / 50 / 99.865 percentiles of a fitted distribution and forms the indices as
spread ratios. The original Clements paper fits Pearson curves from the sample
skewness and kurtosis; **mfgQC implements the same percentile idea by fitting a
parametric distribution** (see [Implementation note](#implementation-note-pearson-curves-vs-fitted-distributions)
below) and taking its $0.00135$, $0.5$, and $0.99865$ quantiles.

The percentile formulas (`_percentile_indices` in the code) are:

$$
C_p = \frac{\text{USL} - \text{LSL}}{X_{0.99865} - X_{0.00135}}
$$

$$
C_{pu} = \frac{\text{USL} - M}{X_{0.99865} - M},
\qquad
C_{pl} = \frac{M - \text{LSL}}{M - X_{0.00135}},
\qquad
C_{pk} = \min(C_{pu}, C_{pl})
$$

where $M = X_{0.5}$ is the **median** of the fitted distribution (not the mean), and
$X_{0.00135}$ / $X_{0.99865}$ are the lower / upper $\pm 3\sigma$-equivalent
percentiles. For a normal distribution these reduce to the ordinary indices, so the
method degrades gracefully. The denominator $X_{0.99865} - X_{0.00135}$ is the
distribution's natural "6σ-equivalent" spread; on a skewed fit it is wider on the
long tail and narrower on the short tail, which is precisely the correction the
normal method misses.

`clements` and `percentile` are the **same method** in mfgQC: both call
`_fit_best_distribution`. The `percentile` name is an alias.

### How the distribution is chosen, `_fit_best_distribution`

The helper fits each candidate family and keeps the one with the highest log-likelihood:

| Candidate | scipy family | Fit constraint |
| --- | --- | --- |
| normal | `stats.norm` | location free |
| lognormal | `stats.lognorm` | `floc=0` (positive support) |
| gamma | `stats.gamma` | `floc=0` |
| Weibull | `stats.weibull_min` | `floc=0` |
| exponential | `stats.expon` | `floc=0` |

The positive-support families are only considered when $\min(y) > 0$, and they are
fit with `floc=0` so they recover the data-generating percentiles instead of
drifting onto a spurious location shift (e.g. a clean 2-parameter lognormal). Each
candidate is fit, its log-likelihood $\sum \ln f(y_i)$ is computed, and the
**highest-likelihood** finite fit wins. If every candidate fails (e.g. degenerate
data), the helper falls back to a plain normal fit. The winning family is named in
the report, e.g. `clements percentile (lognormal fit)`.

## Percentile / fitted-distribution method

`method="percentile"` is the alias of `clements` described above: same
`_fit_best_distribution` selection, same percentile indices. It exists so the
generic name is available; there is no behavioral difference from `clements`.

## Johnson system

Johnson (1949). A **translation** system: find a monotone function that maps the
data to normality, drawing from three families: $S_L$ (lognormal-type, bounded
below), $S_B$ (bounded both sides), and $S_U$ (unbounded). The classic procedure
selects among the families by the data's moments.

As implemented, mfgQC commits to the **unbounded $S_U$ family** specifically:
`stats.johnsonsu.fit(values)` fits a Johnson-$S_U$ to the data, and then the same
percentile machinery as Clements is applied: the $0.00135$, $0.5$, $0.99865$
quantiles of the fitted $S_U$ go into the percentile formulas above. The report
labels it `johnson percentile (johnson-su fit)`. $S_U$ is the most flexible Johnson
family for unbounded, heavy- or light-tailed skew; mfgQC does not currently
auto-select $S_B$ or $S_L$.

!!! note "What "Johnson" means here, precisely"
    Some textbooks describe the Johnson method as transforming to a normal scale and
    computing indices there. mfgQC instead uses the fitted $S_U$ distribution's own
    percentiles in the spread-ratio formulas (the Clements-style percentile route).
    The two are equivalent in intent (both anchor on the $\pm 3\sigma$-equivalent
    quantiles), but the code path is the percentile one. Document the index off the
    fitted $S_U$ quantiles, not off a back-transformed normal interval.

## Worked example

A skewed, strictly positive measurement (lognormal-shaped flatness readings) with a
single upper spec at 4.0. First the normal method, noting that it *flags itself*:

```python
import numpy as np, pandas as pd, mfgqc

rng = np.random.default_rng(42)
y = np.round(rng.lognormal(mean=0.0, sigma=0.5, size=120), 4)
qc = mfgqc.load(pd.DataFrame({"flatness": y})).spec(upper=4.0)

print(qc.capability())                       # normal
```

```text
Process Capability (method=normal)
==================================
n = 120   mean = 1.0461
sigma (within)  =   n/a
sigma (overall) = 0.41642
Cp/Cpk sigma    = overall

Cp  =   n/a
Cpk = 2.364  95% CI (2.06, 2.67)   (Cpu=2.364, Cpl=  n/a)
Pp  =   n/a    Ppk = 2.364   (Ppu=2.364, Ppl=  n/a)
Cpm =   n/a

Assumption checks:
  [FAIL] normality (Anderson-Darling): AD=1.22, p=0.00347; est. Cpk impact 68.8%; n=120

Recommendations:
  - Data are not normal (AD=1.22, p=0.00347); for capability use a non-normal method (method='clements'/'johnson').
```

The normal method reports $C_{pk} = 2.36$, but the guardrail flags non-normality
**and** estimates that it moves $C_{pk}$ by about 69%. That is the cue to switch.
Box–Cox, Clements, and Johnson all land far lower and close to each other:

```python
print(qc.capability(method="boxcox"))
print(qc.capability(method="clements"))
print(qc.capability(method="johnson"))
```

```text
Process Capability (method=boxcox)
==================================
...
Cp/Cpk sigma    = box-cox (lambda=0.0829)
...
Cpk = 1.272  CI: n/a (non-normal method)   (Cpu=1.272, Cpl=  n/a)
```

```text
Process Capability (method=clements)
====================================
...
Cp/Cpk sigma    = clements percentile (lognormal fit)
...
Cpk = 1.4  CI: n/a (non-normal method)   (Cpu=1.4, Cpl=  n/a)
```

```text
Process Capability (method=johnson)
===================================
...
Cp/Cpk sigma    = johnson percentile (johnson-su fit)
...
Cpk = 1.445  CI: n/a (non-normal method)   (Cpu=1.445, Cpl=  n/a)
```

| Method | Reported $C_{pk}$ | What the line says |
| --- | --- | --- |
| normal | 2.364 | overstates capability, long upper tail ignored |
| boxcox | 1.272 | `box-cox (lambda=0.0829)` |
| clements / percentile | 1.400 | `clements percentile (lognormal fit)` |
| johnson | 1.445 | `johnson percentile (johnson-su fit)` |

The normal $C_{pk}$ of 2.36 is badly misleading here: the upper tail of a lognormal stretches
well past where a normal $+3\sigma$ would put it, so the normal method massively
under-counts the defect risk. The three non-normal methods agree near 1.3–1.4. Note
every non-normal line shows `CI: n/a (non-normal method)`: there is no
normal-theory interval to report.

You can confirm the Clements number by hand from the fitted lognormal's percentiles:

```python
from mfgqc.capability import _fit_best_distribution
frozen, name = _fit_best_distribution(y)
lo, med, hi = (float(v) for v in frozen.ppf([0.00135, 0.5, 0.99865]))
print(name, round(lo, 4), round(med, 4), round(hi, 4))
# lognormal 0.3004 0.9702 3.1337
```

$$
C_{pu} = \frac{\text{USL} - M}{X_{0.99865} - M}
       = \frac{4.0 - 0.9702}{3.1337 - 0.9702} = 1.400
$$

which matches the reported $C_{pk}$ exactly (single-sided spec, so
$C_{pk} = C_{pu}$).

## Assumptions and limits, summarized

- **Statistical control comes first.** None of these methods is a remedy for a
  process that is out of control. Confirm control, then characterize the in-control
  shape.
- **Box–Cox** needs strictly positive, smoothly skewed data and a $\lambda$ that
  actually normalizes it. It transforms the spec limits with the data.
- **Clements / percentile** needs one of {normal, lognormal, gamma, Weibull,
  exponential} to fit; it picks the best by log-likelihood and uses median-anchored
  percentile ratios.
- **Johnson** uses the $S_U$ family (unbounded) only, then the same percentile
  ratios. It does not auto-select $S_B$ / $S_L$.
- **No confidence intervals** are reported for any non-normal method.
- **All non-normal indices are long-term (overall)**: $P_p^\ast = C_p^\ast$,
  $P_{pk}^\ast = C_{pk}^\ast$, etc. There is no short-term/within-subgroup variant.
- The chosen $\lambda$ or fitted family is recorded in the result's provenance
  ([Provenance model](provenance.md)), so a reviewer can see exactly which
  transform produced the number.

## Implementation note: Pearson curves vs fitted distributions

Clements' 1989 paper derives the $0.135$/$50$/$99.865$ percentiles from **Pearson
curves** indexed by the sample skewness and kurtosis. mfgQC reaches the same
percentile-ratio indices by **fitting a parametric distribution** (best of normal /
lognormal / gamma / Weibull / exponential by log-likelihood) and reading its
quantiles. The *index formulas* are identical to Clements; the *source of the
percentiles* differs (a maximum-likelihood distribution fit rather than a Pearson
moment-matching). For the common lognormal- and gamma-shaped manufacturing
distributions the two agree closely, but the numbers can differ from a strict
Pearson-curve Clements implementation (e.g. Minitab). When you compare against
another tool, this is the most likely source of a small discrepancy.

## Source standards

- Box, G. E. P., & Cox, D. R. (1964). "An Analysis of Transformations." The
  Box–Cox transform.
- Clements, J. A. (1989). "Process Capability Calculations for Non-Normal
  Distributions." The percentile method.
- Johnson, N. L. (1949). "Systems of Frequency Curves Generated by Methods of
  Translation." The Johnson translation system.

Full citations: [Bibliography](bibliography.md#non-normal-capability).

## See also

- [Capability](capability.md): the normal method, σ families, and confidence intervals.
- [Reading the assumption report](../guide/assumption-report.ipynb): what the normality flag and `est. Cpk impact` mean.
- [Provenance model](provenance.md): how the chosen transform is recorded.
