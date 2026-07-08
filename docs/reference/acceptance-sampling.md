# Acceptance sampling

Acceptance sampling answers a receiving-dock question: a lot of parts has arrived,
inspecting every unit is too slow or too expensive, so you inspect a sample and decide
whether to accept or reject the whole lot from what you find. The decision rule is a
plan. A plan does not measure the true quality of the lot. It trades the cost of
inspection against two risks: rejecting a good lot, and accepting a bad one. mfgQC builds
those plans, draws the curve that shows exactly how the two risks trade off, and reports
the assumptions that the plan rests on but cannot check from the sample.

This page pins every formula and default to the code in `mfgqc/sampling.py`. The
entry points are `mfgqc.sampling_plan`, `mfgqc.find_plan`, `mfgqc.z14_plan` (attributes,
ANSI/ASQ Z1.4), and `mfgqc.z19_plan` (variables, ANSI/ASQ Z1.9).

A note on two cultures of plan. An **attributes** plan classifies each inspected unit as
good or defective and counts the defectives. A **variables** plan measures a continuous
characteristic on each unit and judges the lot from the sample mean and standard
deviation. Attributes plans make no distributional assumption about the measurement.
Variables plans assume the characteristic is normal, inspect fewer units for the same
protection, and give an invalid result, without warning, when that normal assumption
fails. mfgQC covers both and states the normal assumption prominently on the variables side.

---

## 1. Single attribute sampling plans (`sampling_plan`)

### What it is and when you use it

A single attribute plan is a pair of numbers, the sample size $n$ and the acceptance
number $c$. The rule is: draw $n$ units at random from the lot, inspect them, count the
defectives $d$. Accept the lot if $d \le c$, otherwise reject it. "Single" means one
sample and one decision, with no provision for a second sample. You use a single
attributes plan for go/no-go inspection: a unit either passes a gauge or it does not, a
seal either holds or it does not, a label is either present or absent.

```python
plan = mfgqc.sampling_plan(134, 3)     # n = 134, accept if 3 or fewer defectives
```

### The probability of acceptance

Everything about a plan follows from one function: the probability that the plan accepts
a lot whose true incoming fraction defective is $p$. Call it $P_a(p)$. Acceptance means
$d \le c$, so $P_a(p)$ is the probability that a count of defectives is at most $c$.

mfgQC computes $P_a(p)$ from one of three models (`_pa` in `sampling.py`):

| model | distribution of the defective count | $P_a(p)$ |
| --- | --- | --- |
| `binomial` (default) | $X \sim \mathrm{Binomial}(n, p)$ | $P(X \le c)$ |
| `hypergeometric` | $X \sim \mathrm{Hypergeometric}(N,\, D{=}\mathrm{round}(Np),\, n)$ | $P(X \le c)$ |
| `poisson` | $X \sim \mathrm{Poisson}(np)$ | $P(X \le c)$ |

The binomial is the model for sampling from a lot large enough to treat as effectively
infinite (drawing a defective does not appreciably change the fraction defective of what
remains). The hypergeometric is the exact model for a finite lot of known size $N$, where
drawing without replacement does change the remaining fraction. The Poisson is an
approximation to the binomial that holds when $n$ is large and $p$ is small; mfgQC uses it
only when you ask for it by name.

### How the model is chosen

mfgQC does not silently switch models, in keeping with the guardrail design. The rule
(`_select_model`) is:

- If you pass `model=`, that model is used and the reason recorded is "explicitly
  requested".
- Otherwise, if you pass `lot_size=N` and the sampling fraction $n/N$ exceeds `0.1`, the
  hypergeometric model is selected, because at that fraction the binomial approximation to
  a finite lot is questionable.
- Otherwise the binomial is used (the infinite-lot default).

The choice and its reason are attached to the result as an assumption check, so the
report tells you which model produced the numbers and why. If you force the binomial on a
finite lot with $n/N > 0.1$, a second check fires recommending the hypergeometric.

The difference is usually small at the sampling fractions plans run at. For $n=134$,
$c=3$, $p=0.02$: the binomial gives $P_a = 0.719$; the hypergeometric for a lot of
$N=500$ gives $P_a = 0.734$.

### The OC curve

The operating-characteristic (OC) curve is the plot of $P_a(p)$ against the incoming
fraction defective $p$. It is the complete picture of what a plan does. A good lot (small
$p$) sits on the high-left of the curve and is almost always accepted; a bad lot (large
$p$) sits on the low-right and is almost always rejected; in between, the curve slides
from acceptance to rejection. The steeper that slide, the more sharply the plan
discriminates good lots from bad. Increasing $n$ steepens the curve; increasing $c$ shifts
it to the right.

Call the curve on a plan with `plan.oc_curve()`, or chart it directly with `plan.view()`
(the OC curve is the canonical view of a plan). The curve evaluates $P_a$ on a grid of
$p$ from near zero up to twice the LTPD.

### The four risk points

From the OC curve mfgQC reads four standard points (`_build_plan`, via `_invert_pa`,
which inverts the monotone curve by a bracketed root find):

| quantity | definition | meaning |
| --- | --- | --- |
| **AQL** | the $p$ at which $P_a = 0.95$ | the **acceptable quality level**: the worst incoming quality that the plan still accepts with high probability (95%). |
| **Indifference quality** | the $p$ at which $P_a = 0.50$ | the quality at which accept and reject are equally likely. |
| **LTPD / RQL** | the $p$ at which $P_a = 0.10$ | the **lot tolerance percent defective** (also rejectable quality level): quality so poor that the plan accepts it only 10% of the time. |
| **producer's risk $\alpha$** | $1 - P_a(\text{AQL})$ | the chance of rejecting a lot that is actually at the AQL (a good lot). |
| **consumer's risk $\beta$** | $P_a(\text{LTPD})$ | the chance of accepting a lot that is actually at the LTPD (a bad lot). |

By construction $\alpha = 0.05$ and $\beta = 0.10$ on any plan derived this way, because
the AQL and LTPD are defined as the $P_a = 0.95$ and $P_a = 0.10$ points. The producer's
risk is the supplier's exposure (good work rejected); the consumer's risk is the buyer's
exposure (bad work accepted). The AQL is what you promise the supplier; the LTPD is the
quality you are protecting yourself against.

### Worked example: derive a plan and its OC curve

The plan $n = 134$, $c = 3$ is a textbook single attributes plan.

```python
import mfgqc
plan = mfgqc.sampling_plan(134, 3)
print(plan.report())
```

Real output:

```
Sampling Plan n=134, c=3 [single]
=================================
n = 134   c = 3   model = binomial

AQL (Pa=0.95)          = 1.03%
Indifference (Pa=0.50) = 2.73%
LTPD/RQL (Pa=0.10)     = 4.92%
Producer risk alpha    = 0.05
Consumer risk beta     = 0.1

Stated assumptions (not data-checkable): random sampling of the lot, lot homogeneity, and binary good/defective classification of each unit.

Assumption checks:
  [PASS] model_choice (Pa model selection): Pa=0; n=134

Recommendations:
  - no lot_size -> binomial (infinite-lot default)
```

This plan accepts lots at 1.03% defective 95% of the time and lots at 4.92% defective only
10% of the time. Reading $P_a(p)$ along the OC curve makes the shape concrete:

| incoming $p$ | $P_a(p)$ |
| --- | --- |
| 0.50% | 0.9952 |
| 1.00% | 0.9537 |
| 1.03% (AQL) | 0.9494 |
| 2.00% | 0.7192 |
| 2.73% (indifference) | 0.5010 |
| 4.00% | 0.2123 |
| 4.92% (LTPD) | 0.0998 |
| 6.00% | 0.0371 |

(For this plan, $P_a(0.01) = 0.9537$ and $P_a(0.05) = 0.0931$.)

### Dispositioning a lot

Given a plan, `plan.inspect(d)` applies the accept/reject rule to a sample in which $d$
defectives were found:

```python
plan.inspect(2)     # 2 defectives, c = 3
```

```
Lot Disposition: ACCEPT
=======================
decision   = accept
defectives = 2  (acceptance number c = 3)
n          = 134
observed fraction defective = 1.49%
Pa at observed rate         = 0.858
```

Two defectives is at or below $c = 3$, so the lot is accepted. Five defectives would
exceed $c$ and the lot would be rejected.

### Finding a plan from the risks you want (`find_plan`)

`sampling_plan` takes $n$ and $c$ and reports the risks. `find_plan` works the other way:
you state the AQL and LTPD (and optionally $\alpha$ and $\beta$, defaulting to `0.05` and
`0.10`), and it searches for the plan with the **smallest** $n$ whose OC curve gives
$P_a(\text{AQL}) \ge 1 - \alpha$ and $P_a(\text{LTPD}) \le \beta$. Integer $(n, c)$ rarely
hits the requested points exactly, so the result records both what you asked for and what
the chosen plan actually achieves.

```python
mfgqc.find_plan(aql=0.01, ltpd=0.05)
```

```
Sampling Plan n=132, c=3 [find_plan]
====================================
n = 132   c = 3   model = binomial

AQL (Pa=0.95)          = 1.04%
Indifference (Pa=0.50) = 2.77%
LTPD/RQL (Pa=0.10)     = 4.99%
Producer risk alpha    = 0.05
Consumer risk beta     = 0.1

Requested vs achieved:
  AQL  requested 1% -> achieved 1.04%
  LTPD requested 5% -> achieved 4.99%
```

### Assumptions

A single attributes plan rests on three assumptions that the sample cannot verify, and
mfgQC states them on every plan rather than testing them:

1. **Random sampling.** The $n$ units are a random draw from the lot. If inspectors pull
   from the top of the pallet, the sample is not representative and $P_a$ does not apply.
2. **Lot homogeneity.** The lot is a single population, not a mix of a good batch and a
   bad one.
3. **Binary classification.** Each unit is cleanly good or defective.

The one thing mfgQC does check is the model choice itself, described above. The OC-curve
probabilities are exact for the chosen model; the limit is that the model is only
as right as those three assumptions.

### Source

The single-sampling plan, the OC curve, the binomial and hypergeometric acceptance
probabilities, and the AQL / LTPD / producer's-and-consumer's-risk vocabulary follow
**Montgomery**, *Introduction to Statistical Quality Control* (see the
[bibliography](bibliography.md)).

---

## 2. ANSI/ASQ Z1.4, attributes (`z14_plan`)

### What it is and when you use it

Z1.4 is the standard attributes-sampling system used across defense and commercial
supply. Instead of designing a plan from scratch, you look one up by lot size and AQL. The
standard organizes plans so that, as lots get larger, sample sizes grow and the plans
discriminate more finely, and it provides switching rules that tighten inspection after a
run of bad lots and relax it after a run of good ones. You use Z1.4 when a contract or a
customer specifies inspection "to Z1.4 at AQL such-and-such".

```python
mfgqc.z14_plan(lot_size=1000, aql=1.0)     # aql is a PERCENT here
```

### What mfgQC implements

`z14_plan` implements **single sampling, general inspection level II, normal severity**.
That is the most common configuration and the one most contracts default to. The lookup
has two steps:

1. **Lot size to sample-size code letter.** The lot size $N$ maps to a letter A through Q
   (I and O are skipped) from the level-II table `_Z14_CODE_LETTERS`, and each letter maps
   to a sample size $n$ in `_Z14_SAMPLE_SIZE` (A=2, B=3, C=5, ... J=80, ... Q=1250).
2. **Code letter and AQL to the acceptance plan.** The master table is encoded as the
   acceptance-number staircase $[0, 1, 2, 3, 5, 7, 10, 14, 21]$ that the published normal
   table runs along its diagonals (`_z14_cell`, anchored so code K at AQL 1.0 gives
   $A_c = 3$). The cell returns the acceptance number $A_c$ (which mfgQC stores as $c$) and
   the rejection number $R_e = A_c + 1$.

The tabled AQLs are the standard percent-defective columns: `0.10, 0.15, 0.25, 0.40,
0.65, 1.0, 1.5, 2.5, 4.0, 6.5, 10.0`. The AQL argument is a **percent** (`1.0` means 1%),
not a fraction, which differs from `sampling_plan` where the derived AQL is reported as a
fraction internally.

When a (letter, AQL) cell falls off the staircase, the published table prints an arrow:
down to the first plan below (a larger sample) or up to the first plan above (a smaller
sample). `_resolve_z14` follows that arrow to the adjacent code letter and uses the
resolved row for both the sample size and the $(A_c, R_e)$. The report records when an
arrow was followed.

After the plan is fixed, mfgQC computes the same OC-derived risk points as for any plan,
using the binomial (the Z1.4 convention for these lot sizes) unless the sampling fraction
forces the hypergeometric.

### Worked example

```python
mfgqc.z14_plan(1000, 1.0)
```

```
Sampling Plan n=80, c=2 [Z1.4 normal]
=====================================
n = 80   c = 2   model = binomial
lot size N = 1000   (n/N = 0.08)
code letter = J   Ac = 2   Re = 3

AQL (Pa=0.95)          = 1.03%
Indifference (Pa=0.50) = 3.33%
LTPD/RQL (Pa=0.10)     = 6.52%
Producer risk alpha    = 0.05
Consumer risk beta     = 0.1

Requested vs achieved:
  AQL  requested 1% -> achieved 1.03%

Assumption checks:
  [PASS] model_choice (Pa model selection): Pa=0.08; n=80
  [PASS] z14_lookup (ANSI/ASQ Z1.4 single-sampling normal): ANSI/ASQ=80; n=80

Recommendations:
  - lot_size given but n/N=0.08 <= 0.1 -> binomial
  - lot 1000 -> code J; n=80; AQL 1.0% -> Ac=2, Re=3.
```

A lot of 1000 falls in the J code letter, giving $n = 80$. At AQL 1.0% the cell is
$A_c = 2$, $R_e = 3$: inspect 80 units, accept on 2 or fewer defectives, reject on 3 or
more.

### What is and is not covered

mfgQC encodes only level II, normal severity, single sampling. The other severities and
the switching rules that move between them are documented limits, not silent omissions:

- `severity="tightened"` and `severity="reduced"` raise `NotImplementedError`. The
  tightened and reduced master tables are not encoded.
- Inspection levels other than II raise `NotImplementedError`.
- Double and multiple sampling are not implemented.

The normal/tightened/reduced switching procedure of Z1.4 is therefore not automated. You
get the normal-inspection plan; moving to tightened or reduced inspection after the
prescribed run of accepted or rejected lots is a manual step against your copy of the
standard.

### Source

**ANSI/ASQ Z1.4**, *Sampling Procedures and Tables for Inspection by Attributes* (the
civilian successor to MIL-STD-105). Z1.4 is named here as the source standard; it is not a
formatted entry in the [bibliography](bibliography.md), which carries Montgomery as the
general text for the underlying OC-curve mathematics.

---

## 3. ANSI/ASQ Z1.9, variables (`z19_plan`)

### What it is and when you use it

A variables plan judges a lot from measurements rather than from a good/defective count.
You measure a continuous characteristic (a diameter, a fill weight, a tensile strength) on
each of $n$ sampled units, compute the sample mean and standard deviation, and from those
estimate how much of the lot lies outside the specification limit. You use Z1.9 when the
characteristic is measured anyway and you want the protection of an attributes plan from a
smaller sample. The price is an assumption: the estimate of percent nonconforming is only
valid if the characteristic is normally distributed.

```python
mfgqc.z19_plan(lot_size=100, aql=1.0, lower=200.0, upper=210.0)   # aql is a PERCENT
```

### What mfgQC implements

`z19_plan` implements the **standard-deviation method** (sigma unknown), general
inspection level II, normal severity. The lookup mirrors Z1.4: the lot size maps to a
code letter (`_Z19_CODE_LETTERS`, anchored to the published cell lot size 100 to code F)
and each letter to a sample size (`_Z19_SAMPLE_SIZE`, F=10, G=15, ... up to P=200). The
sample sizes are smaller than Z1.4's at the same lot size, which is the point of a
variables plan.

The acceptance rule is built on the **quality index** $Q$, the distance from the sample
mean to a spec limit in sample standard deviations:

$$Q_U = \frac{\text{USL} - \bar{x}}{s}, \qquad Q_L = \frac{\bar{x} - \text{LSL}}{s}.$$

A larger $Q$ means the mean sits farther inside the limit, so less of the lot is
estimated to fall outside it. The standard offers two equivalent acceptance forms, and
mfgQC reports both:

- **Form 1, the k-method.** Accept the lot if every applicable $Q \ge k$, where $k$ is the
  acceptability constant for the sample size and AQL. mfgQC derives $k$ from the
  normal-approximation design that underlies the published table (`_z19_k`):

  $$k = z_{\text{AQL}} - z_\alpha \sqrt{\frac{1}{n} + \frac{z_{\text{AQL}}^2}{2n}},$$

  where $z_{\text{AQL}} = \Phi^{-1}(1 - p)$ is the standard-normal quantile at the AQL
  fraction $p$, $z_\alpha = \Phi^{-1}(1 - \alpha)$ with producer's risk $\alpha = 0.05$,
  and $\Phi^{-1}$ is the inverse standard-normal CDF. This reproduces the published normal-inspection $k$ to about 0.02 for mid-range $n$. Confirm the smallest and largest code letters against your Z1.9 table copy.

- **Form 2, the M-method.** Estimate the percent nonconforming from $Q$ using normal
  theory ($\widehat{p} = \Phi(-Q)$, the normal tail beyond the limit), and accept if that
  estimate is at most $M$. mfgQC sets $M$ as the boundary estimate, the percent
  nonconforming implied when $Q$ exactly equals $k$ (`M = _pct_nonconforming(k)`). This is
  the normal-theory $M$, which is not identical to the published table $M$; the two forms
  agree on the accept/reject decision at the boundary, which is the property that matters.

`z19_plan` returns a `Z19Plan` carrying $n$, the code letter, $k$, $M$, the AQL, and any
attached spec limits. It does not by itself judge a lot. Call `Z19Plan.inspect(sample,
lower=, upper=)` to disposition a sample, which computes $\bar{x}$, $s$, the quality
indices, the estimated percent nonconforming, and the Form-1 decision.

### Worked example: the plan

```python
plan = mfgqc.z19_plan(100, 1.0, lower=200.0, upper=210.0)
print(plan.report())
```

```
Z1.9 Variables Plan (code F)
============================
lot size 100 -> code letter F   (level II, normal)
n = 10   AQL = 1%
Form 1 (k-method): accept if Q >= k = 1.325
Form 2 (M-method): accept if est. % nonconforming <= M = 9.258%
method: standard deviation (sigma unknown). ASSUMES a normal characteristic.

Assumption checks:
  [PASS] normality (assumed (Z1.9)): assumed (Z1.9); n=10

Recommendations:
  - Z1.9 assumes a normal characteristic; check normality on the sample (inspect() does this) - the % nonconforming estimate is invalid otherwise.
```

A lot of 100 maps to code letter F and a sample of 10. At AQL 1% the derived
acceptability constant is $k = 1.325$.

### Worked example: dispositioning a lot

`inspect` applies the plan to a measured sample. With ten readings whose mean sits well
inside both limits, the lot is accepted:

```python
plan = mfgqc.z19_plan(100, 1.0)
sample = [206.5, 207.1, 208.4, 206.9, 207.8, 208.0, 207.2, 206.7, 208.9, 207.5]
plan.inspect(sample, lower=200.0, upper=210.0)
```

```
Z1.9 Lot Disposition: ACCEPT
============================
n = 10   xbar = 207.5   s = 0.7717   k = 1.325
QL = (xbar-LSL)/s = 9.719   est. % below LSL = 0.000%
QU = (USL-xbar)/s = 3.240   est. % above USL = 0.060%
estimated total % nonconforming = 0.060%
Decision: accept (Form 1: accept if every Q >= k = 1.325).

Assumption checks:
  [PASS] normality (Anderson-Darling): AD=0.119, ...
```

Both quality indices clear $k$, so the lot is accepted. Push the mean up toward the upper
limit and the upper quality index drops below $k$:

```python
sample = [208.2, 209.5, 210.3, 208.8, 209.9, 207.6, 209.1, 210.6, 208.4, 209.6]
plan.inspect(sample, lower=200.0, upper=210.0)
```

```
Z1.9 Lot Disposition: REJECT
============================
n = 10   xbar = 209.2   s = 0.9592   k = 1.325
QL = (xbar-LSL)/s = 9.592   est. % below LSL = 0.000%
QU = (USL-xbar)/s = 0.834   est. % above USL = 20.212%
estimated total % nonconforming = 20.212%
Decision: reject (Form 1: accept if every Q >= k = 1.325).
```

Here $Q_U = 0.834$ is below $k = 1.325$, the normal-theory estimate puts about 20% of the
lot above the upper limit, and the lot is rejected.

### Assumptions and the normality guardrail

The variables plan adds one assumption that the attributes plans do not have: the measured
characteristic is **normally distributed**. The percent-nonconforming estimate is a normal
tail area, so if the characteristic is skewed or heavy-tailed, $\Phi(-Q)$ is the wrong
number and the accept/reject decision built on it is not trustworthy.

mfgQC does not assume this silently. The plan carries a standing note that Z1.9 assumes
normality, and `inspect` runs an Anderson-Darling normality test on the sample and attaches
the result. If the sample fails the test, the recommendation is rewritten to say the
estimate and the decision are unreliable. At a Z1.9 sample size (often 10 or fewer), a
normality test has low power, so a pass is weak evidence; the report flags the low power
rather than overstating the check.

The random-sampling and lot-homogeneity assumptions of the attributes plans apply here
too.

### Source

**ANSI/ASQ Z1.9**, *Sampling Procedures and Tables for Inspection by Variables for
Percent Nonconforming* (the civilian successor to MIL-STD-414), is the source standard for
the code-letter and sample-size tables and the $k$- and $M$-method acceptance forms. Z1.9
is named here inline; it is not a formatted entry in the [bibliography](bibliography.md).
The normal-theory derivation of $k$ and the percent-nonconforming estimate follow the same
normal-distribution framework as **Montgomery**, *Introduction to Statistical Quality
Control* (see the [bibliography](bibliography.md)). As the code states, the derived $k$
approximates the published table to about 0.02 for mid-range sample sizes; confirm the
extreme code letters against a copy of the standard.

---

## Result objects and methods

`sampling_plan`, `find_plan`, `z14_plan`, and `z19_plan` return frozen result objects that
share the common mfgQC result surface: `.report()` for the full text, `.summary()` for a
flat scalar dict, `.to_dict()` for a JSON-serializable payload, and `.view()` for the
chart (the OC curve for attribute plans). The plan objects also carry the analysis methods
shown above: `oc_curve()`, `aoq_curve()`, and `inspect()` on the attribute `SamplingPlan`,
and `inspect()` on `Z19Plan`. Each method returns a new result object with its own
provenance step appended.

For the full signatures and result-object methods, see the [API reference](api.md). The
source standards are listed in the [bibliography](bibliography.md).
