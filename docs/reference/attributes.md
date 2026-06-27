---
title: Attributes & Pareto
---

# Attributes capability, process sigma, and Pareto

Not every quality characteristic is a measurement. Some are a verdict: the unit
passed or it failed, the solder joint is good or it is bad, the form had three
errors or none. This page covers the analyses mfgQC offers for that kind of data:
turning a defect rate into a sigma level (`process_sigma`), reporting capability for
pass/fail or count data (`attribute_capability`), and ranking defect categories to
find the few that matter (`pareto`).

These differ from [variables capability](capability.md) in what they consume.
Variables capability needs a column of measurements and a tolerance, and asks how
much of that tolerance the process uses. Attributes capability needs only a count of
defects and a count of units. There are no spec limits, because there is no
continuous scale to compare a limit against. You count, and you convert.

Every formula and default on this page is pinned to `mfgqc/process_sigma.py` and
`mfgqc/pareto_analysis.py`. Every number printed below comes from running the code.

## 1. Process sigma and DPMO

### What it is and when to use it

When a unit either conforms or it does not, the natural summary is a rate: how many
defects per unit, or per million opportunities. `process_sigma` takes a defect count
and a unit count and reports that rate four ways (DPU, DPMO, yield, and a sigma
level), with an exact confidence interval on each. Use it when your inspection
produces counts rather than measurements, or when you want a Six Sigma "sigma level"
for an attribute process.

Two input kinds are accepted, set by the `kind` argument:

- `kind="defectives"` (the default): each unit passes or fails. This is a **binomial**
  count. `defects` is the number of failing units; it cannot exceed `units`.
- `kind="defects"`: a unit can carry several independent defects, counted across a
  fixed number of opportunities per unit. This is a **Poisson** count. `opportunities`
  is the number of defect opportunities per unit (default 1).

### The formulas mfgQC computes

Let $d$ be the defect count, $n$ the number of units, and $o$ the opportunities per
unit.

**Defectives (binomial).** The estimated proportion defective is

$$
\hat p = \frac{d}{n}, \qquad
\text{DPU} = \hat p, \qquad
\text{DPMO} = \hat p \times 10^6, \qquad
\text{FTY} = 1 - \hat p
$$

where FTY is first-time yield. There is one opportunity per unit, so DPMO is the
proportion defective scaled to a million.

**Defects (Poisson).** With total exposure $n \cdot o$,

$$
\text{DPU} = \frac{d}{n}, \qquad
\text{DPMO} = \frac{d}{n\,o} \times 10^6, \qquad
\text{FTY} = e^{-\text{DPU}}
$$

Here yield is the Poisson probability of zero defects on a unit. For a single process
step mfgQC reports rolled throughput yield (RTY) equal to FTY; with one step there is
nothing to roll up.

**From rate to sigma.** The long-term sigma value is the standard-normal quantile that
leaves the defect rate $p$ in the upper tail (`_z_from_rate`):

$$
Z_{\text{lt}} = \Phi^{-1}(1 - p)
$$

The short-term sigma level adds the conventional shift (`SHIFT = 1.5`):

$$
Z_{\text{st}} = Z_{\text{lt}} + 1.5
$$

### The 1.5-sigma shift, stated plainly

The 1.5-sigma shift is a **convention**, not a measurement. The reasoning behind it is
that a process holding a given short-term spread will drift over the long run, and
1.5 sigma is the drift the Six Sigma tradition assumes when translating between a
short-term capability and a long-term observed rate. mfgQC computes $Z_{\text{lt}}$
directly from your observed rate and then *adds* 1.5 to report $Z_{\text{st}}$. It does
not measure the shift from your data.

Because of this, the code reports both bases and never prints a single bare "sigma
level" without saying which one it is. The `BASIS` line in every report spells it out.
The headline "sigma level" is the short-term $Z_{\text{st}}$.

### Confidence intervals

The point rate is only an estimate, and at small $n$ it is a loose one. mfgQC attaches
an exact interval on the rate, then maps its endpoints through to DPMO and to $Z$:

- **Defectives:** the Clopper-Pearson exact binomial interval (`_clopper_pearson`).
- **Defects:** the Garwood exact Poisson interval (`_poisson_exact`).

The higher rate is the worse end, so it maps to the lower $Z$. The interval is a 95%
exact interval by default (`alpha=0.05`).

### Zero defects

A sample with no defects has an estimated rate of zero, which would push $Z$ to
infinity. mfgQC does not report an infinite sigma. Instead it reports the one-sided
exact upper bound on the rate and the corresponding lower bound on $Z_{\text{st}}$,
and tells you to collect more units. The point sigma is unbounded; the bound is what
the data actually support.

### The guardrail

`process_sigma` attaches one assumption check, `rate_stability` (`_adequacy`). It fails
when the sample is too small to trust ($n < 30$) or when zero defects were observed,
and it flags low statistical power when $n < 50$. The check reports the width of the
exact DPMO interval as its magnitude. It warns and recommends; it does not change the
number. This follows mfgQC's guardrail rule: report the assumption, leave the decision
to you.

### Worked example: defectives

Inspecting 500 units and finding 12 defective:

```python
import mfgqc
r = mfgqc.process_sigma(12, 500, kind="defectives")
print(r.report())
```

```text
Attributes capability (defectives): process sigma
=================================================
units = 500   opportunities/unit = 1   defects = 12
DPU = 0.024   DPMO = 24000   (1.246e+04 to 41547.6, exact 95% CI)
first-time yield = 97.6000%

Z.lt (long-term, from the observed rate) = 1.98   [1.73, 2.24]
Z.st (short-term sigma level)            = 3.48   [3.23, 3.74]

BASIS: Z.st = Z.lt + 1.5 sigma. The 1.5 sigma shift is a CONVENTION linking long- and short-term, not a measured quantity; the "sigma level" is the short-term basis.

Assumption checks:
  [PASS] rate_stability (exact CI width / n): exact=2.91e+04; n=500
```

Read it as: 24,000 defects per million, a first-time yield of 97.6%, a long-term sigma
of 1.98, and a short-term sigma level of 3.48. The exact 95% interval on the sigma
level runs from 3.23 to 3.74, so quoting "3.5 sigma" is fair and quoting more
precision than that is not.

### Worked example: defects (Poisson)

A form with 5 opportunities per unit, 200 forms, 37 total errors:

```python
r = mfgqc.process_sigma(37, 200, opportunities=5, kind="defects")
print(r.report())
```

```text
Attributes capability (defects): process sigma
==============================================
units = 200   opportunities/unit = 5   defects = 37
DPU = 0.185   DPMO = 37000   (2.605e+04 to 50999.6, exact 95% CI)
first-time yield = 83.1104%   rolled throughput yield = 83.1104%

Z.lt (long-term, from the observed rate) = 1.79   [1.64, 1.94]
Z.st (short-term sigma level)            = 3.29   [3.14, 3.44]

BASIS: Z.st = Z.lt + 1.5 sigma. The 1.5 sigma shift is a CONVENTION linking long- and short-term, not a measured quantity; the "sigma level" is the short-term basis.

Assumption checks:
  [PASS] rate_stability (exact CI width / n): exact=2.49e+04; n=200
```

DPU is 37/200 = 0.185 errors per form, but DPMO divides by the full exposure of
$200 \times 5 = 1000$ opportunities, giving 37,000. First-time yield is
$e^{-0.185} = 0.8311$, the Poisson chance of a defect-free form.

### Worked example: zero defects

```python
r = mfgqc.process_sigma(0, 300, kind="defectives")
print(r.report())
```

```text
Attributes capability (defectives): process sigma
=================================================
units = 300   opportunities/unit = 1   defects = 0
DPU = 0   DPMO = 0   (0 to 12221, exact 95% CI)
first-time yield = 100.0000%

zero defects observed: the point sigma is unbounded, so mfgQC reports the one-sided exact upper bound on the rate (DPMO <= 1.222e+04), which gives Z.st >= 3.75. Collect more units to tighten it.

BASIS: Z.st = Z.lt + 1.5 sigma. The 1.5 sigma shift is a CONVENTION linking long- and short-term, not a measured quantity; the "sigma level" is the short-term basis.

Assumption checks:
  [FAIL] rate_stability (exact CI width / n): exact=1.22e+04; n=300 [low power]

Recommendations:
  - The rate estimate is unstable (n=300, zero defects); the exact CI spans 1.222e+04 DPMO. Treat the sigma as provisional and collect more units before quoting a single number.
```

Zero defects in 300 units does not prove a perfect process. The exact bound says the
true rate could still be as high as 12,221 DPMO, so the most you can claim is
$Z_{\text{st}} \ge 3.75$. The guardrail fails and the recommendation says to collect
more units.

## 2. Attribute capability

`QCData.attribute_capability` is the fluent entry point to the same computation, for
when your defect or pass/fail data live in a loaded table rather than a pair of
scalars.

```python
qc  = mfgqc.load(df, measure="reject")
cap = qc.attribute_capability()
```

The dispatcher (`mfgqc/data.py`) reads the column named by `defect` (defaulting to the
measure), counts the non-missing units, and sums the column for the total defect
count. If you do not pass `kind`, it infers one: a column containing only 0 and 1 is
read as `defectives` (binomial pass/fail), anything else as `defects` (Poisson
counts). It then calls the same `compute` function described above and returns the
same `ProcessSigmaResult`. Everything in section 1, including the formulas, the 1.5
shift, the exact intervals, and the guardrail, applies unchanged.

How this differs from variables $C_p$/$C_{pk}$: those indices compare a tolerance
width to a process spread and require both a continuous measure and spec limits.
Attribute capability has neither. It summarizes a rate and converts it to a sigma
level. If you have measurements and a tolerance, use [variables
capability](capability.md); use attribute capability when all you have is a count of
good versus bad.

## 3. Pareto analysis

### What it is and when to use it

When defects come from many causes, a few causes usually account for most of the
defects. A Pareto analysis ranks the categories by count and reads off that "vital
few" so you know where to spend your effort first. It is a descriptive tool. It makes
no statistical assumptions and runs no test; it sorts, accumulates, and labels.

```python
import pandas as pd
counts = pd.Series({"Scratch": 50, "Dent": 30, "Misalign": 15, "Other": 5})
r = mfgqc.pareto(counts)
```

`pareto` accepts a pandas Series of counts indexed by category, or a DataFrame plus a
`category` column (with an optional `count` column; if omitted, mfgQC counts the
frequency of each category value).

### How it ranks

The categories are sorted by count in descending order. The cumulative percentage is
the running sum of counts divided by the total, scaled to 100:

$$
\text{cum}\%_k = \frac{100}{\text{total}} \sum_{i=1}^{k} \text{count}_i
$$

The **vital few** are the categories taken in rank order up to and including the first
one whose cumulative fraction reaches the `threshold` (default 0.80). The cutoff is
inclusive: the category that crosses 80% is part of the vital few, not the first one
left out. With a total of zero, the cumulative percentages are all zero and the vital
few is empty.

### Worked example

```python
import mfgqc
import pandas as pd
counts = pd.Series({"Scratch": 50, "Dent": 30, "Misalign": 15, "Other": 5})
r = mfgqc.pareto(counts)
print(r.report())
```

```text
Pareto Analysis
===============
total count   = 100
categories    = 4
vital few (<= 80% cumulative) = 2: Scratch, Dent
top category  = Scratch (50% of total)

Ranked categories:
  * Scratch: 50  (cum 50%)
  * Dent: 30  (cum 80%)
    Misalign: 15  (cum 95%)
    Other: 5  (cum 100%)

Assumption checks:
  (none)
```

Scratch and Dent together reach exactly 80% of the 100 defects, so they are the vital
few (marked with `*`). The reading is direct: fix scratches and dents and you address
four out of five defects. Misalignment and Other are the "trivial many" by comparison.

### The chart

`r.view()` draws the canonical Pareto chart: descending bars colored to set the vital
few apart, with the cumulative-percentage line and the threshold reference on a second
axis. Like every mfgQC chart it is headless-safe.

## Sources

`process_sigma` and `attribute_capability` implement the standard Six Sigma defect-rate
and sigma-level definitions (DPU, DPMO, yield, the rate-to-$Z$ conversion, and the
1.5-sigma shift convention) found in **Montgomery**, *Introduction to Statistical
Quality Control* (see the [bibliography](bibliography.md)). The exact intervals
(Clopper-Pearson for the binomial proportion, Garwood for the Poisson rate) are
classical exact constructions; they are standard methods, not tied to a single text in the bibliography.

`pareto` is a descriptive ranking and is not pinned to a numbered formula in any of the
listed standards.

## See also

- [Variables capability](capability.md): $C_p$/$C_{pk}$ for continuous measurements
  against a tolerance.
- [API reference](api.md): the result surface (`report`, `summary`, `to_dict`,
  `view`) shared by every analysis.
- [Bibliography](bibliography.md): the standards mfgQC is pinned to.
