# Advanced control charts

The Shewhart charts on the [Control charts](control-charts.md) page react to one
point at a time. A single X-bar or individual reading is compared against limits set
three sigma out, and the run-rules engine on the [Run rules](run-rules.md) page
looks for patterns across a window of points. That design is good at catching a
large, sudden shift. It is slow to catch a *small* shift that the process holds
for many readings, because each individual point still looks plausible on its own.

This page covers the charts that fill that gap and a few that answer a different
question entirely:

- **EWMA** and **CUSUM** accumulate information across observations, so a small
  sustained shift in the mean builds up into a signal.
- The **short-run chart** standardizes the measurement so several part numbers,
  each with its own target, share one chart.
- **Pre-control** is a shop-floor running check driven by the tolerance, not by
  control limits.
- The **time-series screen** characterizes drift and autocorrelation so you know
  whether the independence assumption behind a control chart holds at all.

The constants table, the variables/attributes limit formulas, and the run-rules
catalog are not repeated here. See [Control charts](control-charts.md) and
[Run rules](run-rules.md).

A note that applies to EWMA and CUSUM both: each assumes the in-control mean
`mu0` and standard deviation `sigma` are *known*, estimated from a stable
phase-I baseline. mfgQC does not silently assume that baseline is valid. Every
EWMA and CUSUM result carries an `in_control_parameters` assumption check that
states where `mu0` and `sigma` came from and recommends validating phase-I
stability (for example with an I-MR chart) before trusting the limits.

## EWMA

### What it is and when to use it

The exponentially weighted moving average smooths the series. Each plotted point
is a weighted average of the current reading and all readings before it, with the
weights decaying geometrically into the past. Because the statistic carries
memory, a shift that is too small to push any single point past a Shewhart limit
still accumulates in the EWMA until it crosses. Use an EWMA chart for individual
measurements when you care about detecting small, sustained shifts in the mean
(roughly 0.5 to 2 sigma) faster than a Shewhart chart would.

### The formula

The EWMA statistic is the recursion

$$
z_t = \lambda\, x_t + (1 - \lambda)\, z_{t-1}, \qquad z_0 = \mu_0 .
$$

The smoothing constant $\lambda$ sets how much weight the current reading gets. A
small $\lambda$ remembers more history and reacts to smaller shifts; a $\lambda$
of 1 reduces the chart to plotting the raw readings. mfgQC defaults to
$\lambda = 0.1$.

The control limits widen as history builds up. At observation $t$ the half-width is

$$
\text{half-width}_t = L\,\sigma \sqrt{\frac{\lambda}{2 - \lambda}\,\bigl(1 - (1-\lambda)^{2t}\bigr)} ,
$$

with center line $\mu_0$, so $\mathrm{UCL}_t = \mu_0 + \text{half-width}_t$ and
$\mathrm{LCL}_t = \mu_0 - \text{half-width}_t$. The factor $\bigl(1 - (1-\lambda)^{2t}\bigr)$
is the variance build-up term. The variance of $z_t$ starts small and grows
toward its steady-state value, so the early limits are tight and flare out over
the first several points before settling. $L$ is the limit width in sigma units;
mfgQC defaults to $L = 2.7$. As $t$ grows the half-width approaches its asymptote
$L\,\sigma\sqrt{\lambda/(2-\lambda)}$; for $\lambda = 0.1$, $L = 2.7$, $\sigma = 0.15$
that asymptote is about $0.0929$, which the example below approaches from below.

mfgQC resolves the parameters this way (in `mfgqc/timeseries_charts.py`):

- `mu0` defaults to the spec target if one is set, otherwise the sample mean.
- `sigma` defaults to the moving-range estimate $\bar{R}_m / d_2$ with $d_2 = 1.128$
  for moving ranges of size 2, the same short-term estimate an I-MR chart uses.
  This isolates within-process variation and is not inflated by the very shift the
  chart is meant to catch.
- A point signals when $z_t > \mathrm{UCL}_t$ or $z_t < \mathrm{LCL}_t$.

A caution on the sample-mean default: if you let `mu0` default to the sample mean
of a record that already contains a sustained shift, the mean absorbs part of that
shift and the chart can fail to signal. When you know the in-control mean, pass it
as `mu0=` (and `sigma=` if known). The example below does.

### Assumptions

EWMA assumes the in-control `mu0` and `sigma` are known from a stable phase-I
baseline. It assumes individual observations that are independent over time;
autocorrelation distorts the limits. mfgQC reports the parameter source through
the `in_control_parameters` check but does not test independence inside the EWMA
path. To screen for autocorrelation first, use the [time-series screen](#time-series-screen)
below.

### Source

Montgomery, *Introduction to Statistical Quality Control* (see the
[Bibliography](bibliography.md)), the EWMA chapter.

## CUSUM

### What it is and when to use it

The tabular cumulative sum keeps two running totals: $C^+$ accumulates how far the
process has drifted *above* target, and $C^-$ how far it has drifted *below*. As
long as the process sits at `mu0` the sums stay near zero, because a slack value is
subtracted at each step. Once the mean shifts, the relevant sum climbs steadily
until it crosses a decision interval. Like EWMA, CUSUM is built to detect small
sustained shifts faster than a Shewhart chart. The two are alternatives; CUSUM is
the choice when you want the explicit two-sided up/down accounting.

### The formula

mfgQC computes the two-sided tabular CUSUM. With reference value $K = k\sigma$ and
decision interval $H = h\sigma$:

$$
C^+_t = \max\!\bigl(0,\; x_t - (\mu_0 + K) + C^+_{t-1}\bigr), \qquad C^+_0 = 0,
$$

$$
C^-_t = \max\!\bigl(0,\; (\mu_0 - K) - x_t + C^-_{t-1}\bigr), \qquad C^-_0 = 0 .
$$

The **reference value** $k$ is the slack, in sigma units. It is the amount of
drift you are willing to ignore before the sum starts to grow; it is conventionally
set to half the shift you want to detect, so $k = 0.5$ targets a one-sigma shift.
The **decision interval** $h$ is the threshold the sum must cross to signal, also in
sigma units. mfgQC defaults to $k = 0.5$ and $h = 5$, which is a standard pairing
for a one-sigma shift.

Reading a signal: a point is out of control when $C^+_t > H$ (an upward shift) or
$C^-_t > H$ (a downward shift). The two arms are checked separately, and the result
labels the violating point with `cusum_upper` or `cusum_lower` so you know which
direction the process moved.

`mu0` and `sigma` default exactly as for EWMA (spec target or sample mean; then
$\bar{R}_m / d_2$). The chart plots $C^+$ on a positive track and $C^-$ negated on
its own track for readability, with the decision interval drawn at $\pm H$.

### Assumptions

Same as EWMA: known in-control `mu0` and `sigma` from a stable phase-I baseline,
and independent observations over time. The `in_control_parameters` check reports
the parameter source. Independence is not tested inside the CUSUM path.

### Source

Montgomery, *Introduction to Statistical Quality Control* (see the
[Bibliography](bibliography.md)), the CUSUM chapter.

## Worked examples: EWMA and CUSUM

The data below holds 15 readings around an in-control mean of 10.0, then a
sustained shift of about one unit to 11.0 starting at reading 16. The in-control
mean and standard deviation are known from phase I, so they are passed explicitly.

```python
import pandas as pd
import mfgqc

x = [10.1, 9.8, 10.0, 9.9, 10.2, 9.7, 10.1, 10.0, 9.9, 10.0,
     10.2, 9.8, 10.1, 9.9, 10.0,                       # in control at 10.0
     11.0, 10.9, 11.1, 11.0, 10.8, 11.2, 11.0, 10.9, 11.1, 11.0]  # shifted to 11.0
qc = mfgqc.load(pd.DataFrame({"width": x}), measure="width")

ewma = qc.ewma_chart(mu0=10.0, sigma=0.15)   # defaults: lam=0.1, L=2.7
print(ewma.report())
```

```text
EWMA Chart: lambda=0.1, L=2.7
=============================
EWMA: CL=10  mu0=10  sigma=0.15
UCL: 10.04 -> 10.093 (time-varying)
LCL: 9.9595 -> 9.9073 (time-varying)

Out-of-control signals: 9
  point 17: ewma - EWMA statistic beyond control limit
  point 18: ewma - EWMA statistic beyond control limit
  point 19: ewma - EWMA statistic beyond control limit
  point 20: ewma - EWMA statistic beyond control limit
  point 21: ewma - EWMA statistic beyond control limit
  point 22: ewma - EWMA statistic beyond control limit
  point 23: ewma - EWMA statistic beyond control limit
  point 24: ewma - EWMA statistic beyond control limit
  point 25: ewma - EWMA statistic beyond control limit

Assumption checks:
  [PASS] in_control_parameters (phase-I baseline (assumed)): phase-I baseline (assumed); n=25

Recommendations:
  - EWMA/CUSUM assume the in-control mu0 and sigma are known from a stable
    phase-I baseline; here mu0=10 (user-specified) and sigma=0.15
    (user-specified). Validate phase-I stability (e.g. an I-MR chart) before
    trusting these limits.
```

The EWMA statistic climbs through the shift and crosses the upper limit at point
17, two readings after the shift began. The build-up is visible in the statistic:
$z$ goes $9.986,\ 10.087,\ 10.169,\ 10.262,\ 10.336,\ 10.382$ across points 15
through 20, while the upper limit has settled near $10.092$. The notice on the
report is the `in_control_parameters` check confirming the parameters were
user-specified, not a warning that anything failed.

The same data through a CUSUM chart at the defaults:

```python
cusum = qc.cusum_chart(mu0=10.0, sigma=0.15)   # defaults: k=0.5, h=5
print(cusum.report())
```

```text
CUSUM Chart: k=0.5, h=5
=======================
CUSUM: mu0=10  sigma=0.15
K (reference)=0.075  H (decision interval)=0.75

Out-of-control signals: 10
  point 16: cusum_upper - C+ exceeds decision interval H (upward shift)
  point 17: cusum_upper - C+ exceeds decision interval H (upward shift)
  point 18: cusum_upper - C+ exceeds decision interval H (upward shift)
  point 19: cusum_upper - C+ exceeds decision interval H (upward shift)
  point 20: cusum_upper - C+ exceeds decision interval H (upward shift)
  point 21: cusum_upper - C+ exceeds decision interval H (upward shift)
  point 22: cusum_upper - C+ exceeds decision interval H (upward shift)
  point 23: cusum_upper - C+ exceeds decision interval H (upward shift)
  point 24: cusum_upper - C+ exceeds decision interval H (upward shift)
  point 25: cusum_upper - C+ exceeds decision interval H (upward shift)

Assumption checks:
  [PASS] in_control_parameters (phase-I baseline (assumed)): phase-I baseline (assumed); n=25
```

Here $K = k\sigma = 0.5 \times 0.15 = 0.075$ and $H = h\sigma = 5 \times 0.15 = 0.75$.
The upper sum $C^+$ stays at $0$ through the in-control stretch, then climbs
$0.925,\ 1.75,\ 2.775,\ 3.7$ over points 16 through 19 and crosses $H = 0.75$ at
point 16, the first shifted reading. On this dataset CUSUM signals one reading
earlier than EWMA, and the `cusum_upper` label tells you the shift was upward.

## Short-run charts

### What it is and when to use it

A standardized short-run chart lets several part numbers, each with a different
target, share one control chart. Instead of charting the raw measurement, it
charts how far each piece sits from its part's target, in that part's own standard
deviation units. That puts a 12 mm shaft and a 40 mm bore on the same vertical
scale. Use it for low-volume or mixed production where no single part runs long
enough to build a stable chart of its own.

### The formula

For each part $p$ with target $T_p$ and within-part standard deviation $\sigma_p$,
every reading is standardized:

$$
z = \frac{x - T_p}{\sigma_p} .
$$

The pooled $z$ values are plotted on one chart with center line $0$ and limits at
$\pm 3$. The target is resolved per the `target` argument to `short_run_chart`: a
scalar applies one target to every part, a `{part: target}` mapping gives each part
its own, and `None` (the default) uses each part's own sample mean as its target.
$\sigma_p$ is the sample standard deviation of that part's readings (with one degree
of freedom; a part with a single reading contributes $z = 0$). The run-rules engine
runs on the standardized series; the default rule set is Nelson.

### Assumptions

The standardized scale is only shared if the parts have comparable within-part
variability. A part whose spread departs from the pool breaks the common scale.
mfgQC checks this: when two or more parts have usable spread it runs a homogeneity-
of-variance check across them and attaches the result, so a discordant part is
flagged rather than quietly distorting the chart.

### Source

Montgomery, *Introduction to Statistical Quality Control* (see the
[Bibliography](bibliography.md)), the short-run and standardized-chart material.

## Pre-control

### What it is and when to use it

Pre-control, also called stoplight control, is a running shop-floor check tied to
the **tolerance**, not to control limits. It splits the tolerance into colored
zones and uses a simple count of greens, yellows, and reds to decide whether to
keep running, adjust, or stop. It needs no subgroups and no computed control
limits, which is why it is used at the machine for a quick go/no-go on a process
that is already known to be capable and centered.

That last condition matters. Pre-control assumes a capable, centered process. It
is not a substitute for a control chart, and it is unreliable on an incapable or
off-center process. mfgQC estimates Cpk from the data and flags when capability has
not been established.

### The zones and the rules

The tolerance $[\text{LSL}, \text{USL}]$ is divided at the **pre-control lines**

$$
\text{PC}_{\text{lo}} = \text{LSL} + \tfrac{1}{4}(\text{USL} - \text{LSL}), \qquad
\text{PC}_{\text{hi}} = \text{USL} - \tfrac{1}{4}(\text{USL} - \text{LSL}),
$$

which is the central half of the tolerance. A piece is **green** if it falls in
that central half, **yellow** if it is between a pre-control line and its spec
limit (the outer quarters), and **red** if it is outside spec. The target defaults
to the midpoint of the tolerance when no target is set.

Qualification and the running rule:

- **Qualify** the run when five consecutive pieces are green. Until then the run is
  not qualified.
- After qualification, a **two-piece** rule decides on each consecutive pair:
    - a red piece means **STOP** (a piece is out of spec);
    - two yellows on the **same** side mean **ADJUST** (the process has drifted
      off-center toward that limit);
    - two yellows on **opposite** sides mean **STOP** (variation is too wide for
      the tolerance).

`precontrol()` requires both spec limits (set with `.spec(lower=, upper=)`); it
raises a `MissingPrerequisiteError` if either is absent.

### Assumptions

The method presumes the process is capable and centered. mfgQC estimates
$\widehat{\text{Cpk}} = \min(\text{USL} - \mu,\ \mu - \text{LSL}) / (3\hat\sigma)$
from the pooled data and attaches a `capability_prerequisite` check; if the
estimate is below 1.33 the result recommends running a capability study and a
control chart first. Pre-control is a running check, not a capability study.

### Source

Pre-control is shop-floor practice; the specific zone definitions and the five-
green qualification plus two-piece running rule implemented here are not pinned to
a source listed in the [Bibliography](bibliography.md). Treat this method's source
as not yet cited there, rather than assuming one.

## Time-series screen

### What it is and when to use it

The time-series screen answers a prerequisite question for any control chart: are
the observations independent and stationary over time, or do they drift and carry
memory? Control charts (Shewhart, and to a degree EWMA and CUSUM) assume the
readings are not autocorrelated. When they are, the limits are unreliable. The
screen characterizes that structure. It is deliberately *not* a forecaster: there
is no ARIMA or machine-learning model here, and nothing predicts future values.

`timeseries()` reports two things:

- **Trend.** It regresses the measure on time order (a linear slope with a t-test)
  and also runs the nonparametric **Mann-Kendall** trend test (the $\tau$ statistic
  and its p-value). Reporting both means a monotone but nonlinear drift is still
  caught. The direction is read off Mann-Kendall: increasing, decreasing, or no
  trend at the chosen $\alpha$ (default 0.05).
- **Autocorrelation.** It computes the sample autocorrelation function ACF($k$) on
  the mean-centered series and flags every lag whose ACF falls outside the band
  $\pm z_{1-\alpha/2}/\sqrt{n}$ (with $z_{1-\alpha/2} = 1.96$ at the default
  $\alpha$). A flagged lag is evidence the observations are not independent.

mfgQC surfaces both as assumption checks. If a trend is present, the
recommendation says to detrend before control charting or to model the trend. If
autocorrelation is present, the recommendation says Shewhart limits are unreliable
and points you toward EWMA, CUSUM, or a time-series model. The result's own summary
text states plainly that it is an exploratory screen that complements, and does not
replace, the CUSUM and EWMA charts for monitoring. The screen needs at least four
observations.

There is also a richer set of characterization tools in `mfgqc/timeseries.py`
(a standalone trend result, a full ACF/PACF view via the Durbin-Levinson
recursion, and a classical additive trend-plus-seasonal-plus-residual
decomposition). Those are reached through the module functions rather than the
`timeseries()` dispatcher.

### Source

The linear-trend regression follows mfgQC's own regression machinery (ordinary
least squares). The Mann-Kendall test and the $\pm 1.96/\sqrt{n}$ ACF band are
standard time-series methods; they are not pinned to a text listed in the
[Bibliography](bibliography.md); the NIST/SEMATECH e-Handbook listed there covers these methods at the reference level.

## See also

- [Control charts](control-charts.md): the Shewhart variables/attributes charts,
  the constants table, and the limit formulas.
- [Run rules](run-rules.md): the Western Electric and Nelson out-of-control rules.
- [Bibliography](bibliography.md): the sources cited above.
- [API reference](api.md): `ewma_chart`, `cusum_chart`, `short_run_chart`,
  `precontrol`, and `timeseries`.
