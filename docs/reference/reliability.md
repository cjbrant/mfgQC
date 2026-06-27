---
title: Reliability
---

# Reliability and life-data analysis

Reliability analysis asks how long a part survives, and with what confidence. The
answer depends on the data you have. Sometimes you have failure times from a life
test, some of them censored because the test stopped before every unit failed.
Sometimes you have a count of failures over a known number of operating hours.
Sometimes you have no field data at all and you are sizing a demonstration test or
composing the reliability of a system from its components. mfgQC covers each of these
with a separate entry point, and each one reports the assumption that makes its number
trustworthy.

This page pins every formula, default, and method to the code in the
`mfgqc/reliability/` subpackage: `life.py`, `nonparametric.py`, `system.py`,
`availability.py`, and `demonstrate.py`. The maximum-likelihood life fit is regression
tested against R's `survreg`; the rest are pinned to the standards and texts cited at
the end.

A note on roles. The life-data entry points read two metadata roles off the loaded
table: `time` (the operating time or cycles at which the unit failed or was censored)
and `event` (1 for an exact failure, 0 for a right-censored suspension). If you do not
set a `time` role, the measure column is used as the time. If you do not set an `event`
role, every observation is treated as an exact failure. Set the roles with `.roles(...)`:

```python
qc = mfgqc.load(df, measure="hours").roles(time="hours", event="event")
```

---

## 1. Parametric life-data fitting (`life_fit`)

### What it is and when you use it

A life-data fit puts a probability distribution on time-to-failure. Once you have the
fitted distribution you can read off the mean time to failure, the time by which 10
percent of units have failed (B10 life), the median life (B50), and the reliability
$R(t)$ at any time. You use it when you have a sample of failure times, possibly with
some units still running at the end of the test (suspensions), and you want a smooth
model rather than the empirical step function.

```python
fit = qc.life_fit(dist="weibull")          # MLE, 95% CI, with right censoring
```

### The distributions

`life_fit` supports four distributions (the `_DISTS` tuple in `life.py`):
`exponential`, `weibull`, `lognormal`, and `normal`. The default is `weibull`. The
parameters mfgQC reports are:

| distribution | parameters | scipy form |
| --- | --- | --- |
| `exponential` | `scale` ($\theta$) | `expon(scale=θ)` |
| `weibull` | `shape` ($\beta$), `scale` ($\eta$) | `weibull_min(β, scale=η)` |
| `lognormal` | `mu`, `sigma` (of $\ln t$) | `lognorm(σ, scale=e^{μ})` |
| `normal` | `mu`, `sigma` | `norm(loc=μ, scale=σ)` |

The Weibull shape $\beta$ is the diagnostic that matters most on the plant floor.
$\beta < 1$ is infant mortality (failure rate decreasing with age), $\beta = 1$ is a
constant failure rate (the Weibull reduces to the exponential), and $\beta > 1$ is
wear-out (failure rate increasing with age).

### Maximum likelihood and how censoring is handled

The default method is maximum likelihood (`method="mle"`). Censoring is built into the
likelihood, not worked around. Each observation contributes a different term to the log
likelihood depending on what is known about it (`_negloglik` in `life.py`):

- an **exact failure** at $t$ contributes the log density $\ln f(t)$;
- a **right-censored suspension** (still running at $t$) contributes the log survival
  $\ln R(t) = \ln\big(1 - F(t)\big)$;
- a **left-censored** unit contributes $\ln F(t)$;
- an **interval-censored** unit, known to have failed between $t_{lo}$ and $t_{hi}$,
  contributes $\ln\big(F(t_{hi}) - F(t_{lo})\big)$.

The total log likelihood is the sum of these terms, and mfgQC minimizes its negative
with a Nelder-Mead search. The public `life_fit` dispatcher reads the `event` role and
splits the data into exact failures (`event == 1`) and right-censored suspensions
(`event == 0`). The left- and interval-censored terms exist in the likelihood machinery
but are not currently populated by the dispatcher, so through `qc.life_fit(...)` you get
exact failures plus right censoring. A fit needs at least 2 exact failures; with fewer,
`life_fit` raises an error and points you to the nonparametric `life_table` instead.

There is a second method, `method="rankreg"`, which fits by rank regression on the
probability plot. It uses median-rank plotting positions adjusted for suspensions by the
Johnson method, with the Benard approximation $F = (\text{rank} - 0.3)/(n + 0.4)$, then
fits a straight line to the linearized cumulative distribution. It is the recommended
cross-check when you have few failures, because the Weibull-shape MLE is biased upward
at small failure counts.

The reported MTTF (mean time to failure) is the mean of the fitted distribution, never
the sample mean of censored data. For the Weibull, $\text{MTTF} = \eta\,\Gamma(1 +
1/\beta)$, computed through scipy's frozen distribution.

### Confidence intervals

For the MLE, the parameter confidence intervals are **likelihood-ratio profile
intervals** (`_lr_ci` in `life.py`), not Wald (normal-approximation) intervals. For each
parameter, mfgQC profiles the log likelihood: it fixes the parameter at a trial value,
re-optimizes the others, and finds the two values where the profile log likelihood drops
below the maximum by $\chi^2_{1,\,1-\alpha}/2$. These intervals are not symmetric about
the estimate, which matters for the Weibull shape and scale at small samples. The
rank-regression method uses a Wald interval from the numerically estimated Hessian
instead.

### What mfgQC checks for you

`life_fit` attaches three assumption checks (`_adequacy` in `life.py`):

1. **Distribution adequacy.** The probability-plot correlation (PPCC) is reported, and
   the check passes if it is at least 0.95. mfgQC also fits all four distributions and
   reports their AIC so you can compare. It does not switch distributions for you.
2. **Failure count.** Fewer than 10 exact failures is flagged as low power, with a
   recommendation to cross-check the shape with `method="rankreg"`.
3. **Constant failure rate.** For a Weibull fit, a shape more than 0.5 away from 1 is
   flagged, because a constant-rate MTBF would mislead you when the rate is not constant.

### Worked example

Ten units on a life test. Two were still running when the test stopped (event = 0):

```python
import pandas as pd, mfgqc

df = pd.DataFrame({
    "hours": [120, 230, 410, 580, 690, 750, 880, 1010, 1200, 1450],
    "event": [1,   1,   1,   1,   0,   1,   1,   0,    1,    1],
})
qc  = mfgqc.load(df, measure="hours").roles(time="hours", event="event")
fit = qc.life_fit(dist="weibull")
print(fit.report())
```

```
Life fit (weibull, mle): R(t) and percentiles
=============================================
n = 10   failures = 8   suspensions (right-censored) = 2
method = mle (MLE primary; rank regression secondary)

shape     =        1.686   [0.86732, 2.8272]  (95% CI)
scale     =       918.99   [584.8, 1559.8]  (95% CI)

MTTF = 820.44 (from the fitted distribution, not the sample mean)
B10 (10% fail) = 241.9   B50 (median life) = 739.44
AIC = 126.59   probability-plot correlation = 0.9929

competing fits (AIC, lower is better): weibull=126.6, exponential=127.1, normal=127.6, lognormal=128.1

Assumption checks:
  [PASS] distribution_fit (prob-plot correlation): prob-plot=0.993; n=10
  [FAIL] failure_count (exact failures): exact=8; n=10 [low power]
  [FAIL] constant_failure_rate (Weibull shape vs 1): Weibull=1.69; n=10
```

The shape estimate is 1.69, so these units are in wear-out, and mfgQC flags that a
constant-rate MTBF would be the wrong summary here. The profile interval on the shape,
$[0.87,\,2.83]$, is wide because there are only 8 exact failures, and mfgQC flags that
too. The fitted result also exposes `fit.R(t)` for the reliability at any time and
`fit.hazard(t)` for the instantaneous failure rate.

---

## 2. Nonparametric reliability (`life_table`, Kaplan-Meier)

### What it is and when you use it

The Kaplan-Meier estimator gives the reliability function $R(t)$ without assuming any
distribution. It is the product-limit step function: the empirical survival curve,
corrected for censoring. Use it as the assumption-free baseline, to read the median life
straight from the data, or to judge by eye whether a parametric fit is reasonable.

```python
km = qc.life_table()
```

### The product-limit estimator

Order the distinct failure times $t_1 < t_2 < \dots$. At each failure time $t_i$, let
$d_i$ be the number of failures at $t_i$ and $n_i$ the number of units still at risk
(operating and not yet failed or censored) immediately before $t_i$. The estimator is the
running product (`kaplan_meier` in `nonparametric.py`):

$$
\hat R(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right).
$$

A suspension removes a unit from the at-risk count without contributing a failure, so it
flattens the curve rather than dropping it. The confidence bounds use **Greenwood's
formula** for the variance,

$$
\widehat{\operatorname{Var}}\big(\hat R(t)\big) = \hat R(t)^2 \sum_{t_i \le t}
\frac{d_i}{n_i\,(n_i - d_i)},
$$

with a normal $z$-interval clipped to $[0, 1]$. The median life is the first failure
time at which $\hat R(t)$ falls to or below 0.5; if the curve never reaches 0.5 (heavy
censoring), the median is reported as "not reached." The estimator carries no assumption
checks, because there is no distributional assumption to check.

### Worked example

The same ten units as above:

```python
km = qc.life_table()
print(km.report())
```

```
Kaplan-Meier R(t): 8 failures / 10 units
========================================
n = 10   failures = 8   suspensions = 2
median life = 750   [1200, 410]

         t      R(t)     lower     upper
         0    1.0000    1.0000    1.0000
       120    0.9000    0.7141    1.0000
       230    0.8000    0.5521    1.0000
       410    0.7000    0.4160    0.9840
       580    0.6000    0.2964    0.9036
       750    0.4800    0.1587    0.8013
       880    0.3600    0.0445    0.6755
      1200    0.1800    0.0000    0.4752
      1450    0.0000    0.0000    0.0000
```

The median life of 750 hours sits close to the Weibull B50 of 739 from the parametric
fit, which is one reason to trust the Weibull here.

---

## 3. MTBF (`mtbf`)

### What it is and when you use it

When the failure rate is constant (the exponential life model), the natural summary is
the mean time between failures: total operating time divided by the number of failures.
You use it for repairable systems and for components in their useful-life period, where
failures arrive at a roughly steady rate. The constant-rate assumption is the catch, and
mfgQC says so on every report.

```python
m = mfgqc.reliability.mtbf(5000.0, failures=3)          # total time, failures
m = qc.mtbf()                                            # from time + event roles
```

### The formula and bounds

The point estimate is total accumulated test time $T$ over the number of failures $r$:

$$
\widehat{\text{MTBF}} = \frac{T}{r}.
$$

The confidence bounds come from the chi-square distribution and depend on how the test
was stopped (`mtbf` in `demonstrate.py`). For a **time-terminated** test (stopped at a
fixed time, the default `kind="time_terminated"`):

$$
\text{lower} = \frac{2T}{\chi^2_{1-\alpha/2,\;2r+2}}, \qquad
\text{upper} = \frac{2T}{\chi^2_{\alpha/2,\;2r}}.
$$

For a **failure-terminated** test (stopped at the $r$-th failure, `kind="failure_terminated"`):

$$
\text{lower} = \frac{2T}{\chi^2_{1-\alpha/2,\;2r}}, \qquad
\text{upper} = \frac{2T}{\chi^2_{\alpha/2,\;2r}}.
$$

The difference is the extra two degrees of freedom in the time-terminated lower bound,
which accounts for the partial interval after the last failure. The default confidence
is 0.90. Passed a `QCData`, `mtbf` derives $T$ as the sum of the time column and $r$ as
the count of `event == 1`.

### Worked example

A test ran for 5000 total unit-hours and saw 3 failures, stopped at a fixed time:

```python
m = mfgqc.reliability.mtbf(5000.0, failures=3, kind="time_terminated", conf=0.90)
print(m.report())
```

```
MTBF (time_terminated, constant failure rate)
=============================================
total test time = 5000   failures = 3   test = time_terminated
MTBF = 1666.67   [644.857, 6114.78] (90% CI)
failure rate = 0.0006
```

The point MTBF is $5000/3 = 1666.67$ hours, and the 90 percent interval runs from 645 to
6115 hours. The interval is very wide because three failures is little information, which
is the honest picture at this sample size.

---

## 4. System reliability (`system`, `series`, `parallel`, `k_of_n`)

### What it is and when you use it

System reliability composes the reliability of a system from the reliabilities of its
components. You use it in design and planning, before you have field data, to see
whether an architecture meets a target and where redundancy helps. Every formula here
assumes the component failures are **independent**, and mfgQC attaches an independence
flag to every result, because a shared cause (a common power supply, a common
environment) breaks the assumption.

### The formulas

For component reliabilities $R_1, \dots, R_n$ (`system.py`):

**Series** (all components must survive):

$$
R_{\text{series}} = \prod_{i=1}^{n} R_i.
$$

**Parallel** (redundant, at least one must survive):

$$
R_{\text{parallel}} = 1 - \prod_{i=1}^{n} (1 - R_i).
$$

**k-of-n** (at least $k$ of $n$ identical components, each with reliability $R$,
survive) is the upper tail of a binomial:

$$
R_{k/n} = \sum_{j=k}^{n} \binom{n}{j} R^{\,j} (1 - R)^{\,n-j}.
$$

The `system(blocks)` function nests these. A `blocks` argument is a float (one
component), a list (a series of blocks), or a dict `{"series": [...]}` or
`{"parallel": [...]}` that nests the above to any depth, so you can build a block diagram
like `{"series": [0.99, {"parallel": [0.9, 0.9]}]}`.

### Worked examples

```python
print(mfgqc.reliability.series([0.99, 0.98, 0.97]).report())
print(mfgqc.reliability.parallel([0.90, 0.90]).report())
print(mfgqc.reliability.k_of_n(2, 3, 0.95).report())
```

```
System reliability (series)
===========================
structure: series   components: 3
R = prod(0.99, 0.98, 0.97)
system reliability = 0.941094
```

Three components in series multiply to $0.99 \times 0.98 \times 0.97 = 0.941$. Two
components in parallel at 0.90 each give $1 - (0.10)(0.10) = 0.99$, and a 2-of-3 system
of 0.95 components gives 0.99275. Each report repeats the independence assumption in
plain words.

---

## 5. Availability (`availability`)

### What it is and when you use it

Availability is the fraction of time a repairable system is up and able to do its job.
Reliability asks how long until failure; availability folds in how fast you repair. You
use it for equipment that is maintained and returned to service, where the steady-state
uptime matters more than a single time-to-failure.

### The formulas

mfgQC computes three availability indices (`availability` in `availability.py`), each a
steady-state ratio of uptime to total time:

**Inherent** availability (`kind="inherent"`, the default) counts corrective repair only:

$$
A_i = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}}.
$$

**Achieved** availability (`kind="achieved"`) adds preventive maintenance. With
corrective rate $\lambda = 1/\text{MTBF}$, preventive frequency $f_{pm}$ (actions per
unit time), and preventive time $t_{pm}$ per action, mfgQC forms the mean time between
maintenance and the mean maintenance time,

$$
\text{MTBM} = \frac{1}{\lambda + f_{pm}}, \qquad
\bar M = \frac{\lambda\,\text{MTTR} + f_{pm}\,t_{pm}}{\lambda + f_{pm}}, \qquad
A_a = \frac{\text{MTBM}}{\text{MTBM} + \bar M}.
$$

**Operational** availability (`kind="operational"`) adds the mean logistics delay time
(waiting for a part or a technician):

$$
A_o = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR} + \text{MLDT}}.
$$

### Worked example

Using the MTBF from the example above (1666.67 hours) and an 8-hour mean repair time:

```python
print(mfgqc.availability(mtbf=1666.67, mttr=8.0, kind="inherent").report())
```

```
Availability (inherent)
=======================
inherent availability = 0.995223 (99.5223%)
formula: MTBF / (MTBF + MTTR)

inputs:
  mtbf = 1666.67
  mttr = 8
```

The system is up 99.52 percent of the time once corrective repair is accounted for.

---

## 6. Bearing life (`bearing_life`, ISO 281)

### What it is and when you use it

For rolling-element bearings, the standard life metric is the basic rating life $L_{10}$:
the life that 90 percent of a population of identical bearings will reach under a stated
constant load and speed. It is a fleet rating, not a single-unit prediction. You use it
to size or compare bearings against a duty cycle.

### The formula

The basic rating life in millions of revolutions follows the load-life relation
(`bearing_life` in `system.py`):

$$
L_{10} = \left(\frac{C}{P}\right)^{p}
$$

where $C$ is the basic dynamic load rating, $P$ is the equivalent dynamic bearing load,
and the life exponent $p$ depends on the contact geometry: $p = 3$ for ball bearings
(`kind="ball"`) and $p = 10/3$ for roller bearings (`kind="roller"`). mfgQC converts to
hours using the shaft speed in rpm,

$$
L_{10,\,\text{hours}} = \frac{10^6}{60 \cdot \text{rpm}} \cdot L_{10}.
$$

It also reports the rated life at other reliabilities through the ISO 281 reliability
factor $a_1$, computed from a Weibull slope of 1.5:

$$
a_1(R) = \left(\frac{\ln(1/R)}{\ln(1/0.90)}\right)^{1/1.5}.
$$

### Worked example

A ball bearing with a dynamic rating of 26000 N, an equivalent load of 4000 N, running
at 1800 rpm:

```python
print(mfgqc.reliability.bearing_life(C=26000, P=4000, rpm=1800, kind="ball").report())
```

```
ISO 281 rated bearing life (ball)
=================================
C (dynamic rating) = 26000   P (equivalent load) = 4000   speed = 1800 rpm   life exponent p = 3
L10 = 274.62 million revolutions = 2542.82 hours

rated life at other reliabilities (ISO 281 a1 factor):
  L10 (R=0.90): 2542.82 hours
  L 5 (R=0.95): 1573.64 hours
  L 4 (R=0.96): 1351.43 hours
  L 3 (R=0.97): 1111.76 hours
  L 2 (R=0.98): 845.546 hours
  L 1 (R=0.99): 530.866 hours
```

Here $(C/P)^3 = 6.5^3 = 274.6$ million revolutions, which at 1800 rpm is 2543 hours at
90 percent reliability. Demanding 99 percent reliability cuts the rated life to 531
hours.

---

## 7. Demonstration test (`demonstration_test`)

### What it is and when you use it

A reliability demonstration test sizes a pass/fail test to prove a reliability target at
a stated confidence. The common case is the zero-failure (success-run) test: how many
units must run without failure to demonstrate reliability $R$ at confidence $C$? You use
it to plan acceptance tests and to trade off sample size against the strength of the
claim.

### The formula

The default model is binomial (`demonstration_test` in `demonstrate.py`). For a
**zero-failure** plan (`failures=0`), the confidence that the true reliability is at
least $R$ when $n$ units pass is

$$
C = 1 - R^{\,n},
$$

which rearranges to the required sample size

$$
n = \left\lceil \frac{\ln(1 - C)}{\ln R} \right\rceil.
$$

You leave exactly one of `reliability`, `confidence`, or `n` as `None`, and
`demonstration_test` solves for it. For a **fixed-failure** plan (`failures = f > 0`),
the confidence comes from the binomial tail, $C = 1 - \text{Binom.cdf}(f;\, n,\, 1 - R)$,
and mfgQC searches for the sample size (or solves for reliability) that meets the target.

A zero-failure pass **bounds** the reliability at the stated confidence; it does not
prove it, and it demonstrates the assumed model as much as the hardware. mfgQC prints
that caveat on every plan.

### Worked example

How many units must pass a zero-failure test to demonstrate 95 percent reliability at 90
percent confidence?

```python
print(mfgqc.reliability.demonstration_test(reliability=0.95, confidence=0.90, failures=0).report())
```

```
Reliability demonstration test (solved for n)
=============================================
demonstrate reliability >= 0.9500 at 90% confidence
plan: n = 45 units, zero-failure (success-run)
assumed model: binomial
```

The plan requires 45 units to pass with no failures. That matches
$\lceil \ln(0.10)/\ln(0.95) \rceil = \lceil 44.9 \rceil = 45$.

---

## Sources

- **ISO 281**, *Rolling bearings: dynamic load ratings and rating life.* The basis for
  the $L_{10}$ basic rating life, the $(C/P)^p$ load-life relation with $p = 3$ for ball
  bearings and $p = 10/3$ for roller bearings, and the $a_1$ reliability adjustment. See
  the [bibliography](bibliography.md).
- **Montgomery, D. C.** *Introduction to Statistical Quality Control.* Wiley. mfgQC's
  primary source for SPC and capability; the reliability material here draws on the
  same general framework for the exponential MTBF and chi-square interval. See the
  [bibliography](bibliography.md).
- The parametric maximum-likelihood fit matches R's `survreg` from the `survival` package, a standard reference for survival models. It is not a primary text in the [bibliography](bibliography.md).

The Kaplan-Meier product-limit estimator with Greenwood variance, the series/parallel/
k-of-n system formulas, the availability indices, and the binomial demonstration plan are
standard results in the reliability-engineering literature. The bibliography does not yet
carry a dedicated reliability-engineering text (for example Meeker and Escobar, or
O'Connor); the formulas above are pinned to the cited standards and to the code, and the
text notes where no primary source is listed rather than inventing one.

For the full signatures and result-object methods, see the
[API reference](api.md).
