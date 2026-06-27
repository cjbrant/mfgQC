# Run rules

Control limits catch a single point thrown far out, a large, abrupt shift. They
are deliberately blind to **smaller, sustained patterns**: a slow drift, a process
that has settled on the wrong side of center, a creeping increase in spread. Run
rules (also called *zone tests* or *tests for special causes*) close that gap by
examining the *pattern* of points relative to the center line and the standard-deviation
**zones**:

| Zone | Distance from center line |
| --- | --- |
| **A** | between $2\sigma$ and $3\sigma$ |
| **B** | between $1\sigma$ and $2\sigma$ |
| **C** | within $1\sigma$ |

mfgQC ships two rule sets, **Nelson** (the default) and **Western Electric**, and
reports every violation as a structured `Violation` object, not just a flag on a
plot:

| Field | Meaning |
| --- | --- |
| `point` | 1-based index of the point that *completes* the pattern |
| `value` | the plotted value at that point |
| `chart` | `"location"` (the X-bar / individuals panel) or `"dispersion"` (R / S / MR) |
| `rule` | the rule id, e.g. `nelson_2`, `western_electric_4` |
| `description` | a plain-language statement of the criterion |

Because the result object carries them structurally, a frontend or report builder
consumes `result.to_dict()` or iterates `result.violations`; it never parses the
report text. See [Control charts](control-charts.md) for the chart families these
rules run on.

## Selecting the rule set

The rule set is chosen with the `rules=` argument on `control_chart`:

```python
qc.control_chart(rules="nelson")            # default
qc.control_chart(rules="western_electric")
```

Any other value raises `ValueError`. The chosen set is recorded in the result's
provenance step (`params["rules"]`), so an archived analysis records which tests it
was judged against.

!!! note "Rule 1 is shared"
    Both sets begin with the same headline test, **one point beyond $3\sigma$**,
    emitted as `nelson_1` or `western_electric_1` depending on the set. Everything
    after Rule 1 differs.

## The Nelson rules

`rules="nelson"` applies the following. Each rule's "point" is the index that
*completes* the pattern (e.g. the 9th point of a run, not the 1st).

| Rule id | Detects | Exact criterion as implemented |
| --- | --- | --- |
| `nelson_1` | A gross, single-point excursion | One point with $\lvert z\rvert > 3$ (beyond $3\sigma$) |
| `nelson_2` | A sustained shift in the mean | Nine points in a row on the same side of the center line |
| `nelson_3` | A trend / drift | Six points in a row, each strictly greater than (or each strictly less than) the one before |
| `nelson_4` | Over-control / systematic oscillation | Fourteen points in a row strictly alternating up and down |
| `nelson_5` | A shift, caught faster than Rule 2 | Two out of three consecutive points beyond $2\sigma$ on the same side, **with the completing point itself beyond $2\sigma$** |
| `nelson_6` | A smaller sustained shift | Four out of five consecutive points beyond $1\sigma$ on the same side, **with the completing point itself beyond $1\sigma$** |
| `nelson_7` | Stratification (variance too small) | Fifteen points in a row all within $1\sigma$ (Zone C, either side) |
| `nelson_8` | A mixture / bimodal pattern | Eight points in a row all beyond $1\sigma$ (none in Zone C), on either side |

Here $z = (x - \text{CL})/\sigma$ is the standardized distance of a plotted point
from the center line.

!!! info "Implementation details worth knowing"
    - **Rules 5 and 6 require the completing point to clear the threshold itself.**
      The code (`_k_of_m_beyond`) demands `abs(z[i]) > thresh` for the point that
      closes the window, not merely "$k$ of the last $m$." This is slightly stricter
      than a literal "2 of 3 in Zone A" reading and prevents an in-control point from
      inheriting a signal.
    - **The "$k$ of $m$ same side" count is taken per side**, not netted: a window
      with two points above $2\sigma$ *and* one below does not cancel: it is the
      count on the high side ($\ge k$) or the low side ($\ge k$) that triggers.
    - **Rule 1 takes priority on shared points.** If two rules would flag the same
      index, the first one assigned wins (the engine uses `setdefault`), and Rule 1
      is assigned first. One index yields at most one `Violation`.

## The Western Electric rules

`rules="western_electric"` applies four rules. The first three are the classic
zone tests; the fourth is the run test.

| Rule id | Detects | Exact criterion as implemented |
| --- | --- | --- |
| `western_electric_1` | A gross, single-point excursion | One point beyond $3\sigma$ ($\lvert z\rvert > 3$) |
| `western_electric_2` | A shift toward one side | Two out of three consecutive points beyond $2\sigma$ on the same side, **with the completing point itself beyond $2\sigma$** |
| `western_electric_3` | A smaller sustained shift | Four out of five consecutive points beyond $1\sigma$ on the same side, **with the completing point itself beyond $1\sigma$** |
| `western_electric_4` | A sustained shift in the mean | Eight points in a row on the same side of the center line |

The same per-side counting and "completing point clears the threshold" details from
the Nelson section apply to `western_electric_2` and `western_electric_3`: they are
the **same engine** (`_k_of_m_beyond` with $(k,m,\text{thresh}) = (2,3,2.0)$ and
$(4,5,1.0)$). The only differences between the two sets are: Nelson adds the trend,
alternation, stratification and mixture tests (Rules 3, 4, 7, 8) and uses a **9-point**
run for the same-side test, whereas Western Electric uses an **8-point** run.

## How the engine detects patterns

All rules run off two derived series computed once per chart panel:

- $z = (x - \text{CL})/\sigma$, the standardized distance, used by every zone test.
- $\text{sign}(x - \text{CL})$, used by the same-side run tests.

For the location panel, $\sigma$ is recovered from the limits as
$\sigma = (\text{UCL} - \text{CL})/3$, so the zones are always exactly thirds of the
control band regardless of the chart family. The helpers are:

- `_runs_same_side(signs, length)`: Rules 2 / 4 (WE): a window of `length`
  consecutive points all with the same sign.
- `_trend(x, length)`: Rule 3 (Nelson): a window whose successive differences are
  *all* positive or *all* negative (strict monotonicity; equal consecutive values
  break the trend).
- `_alternating(x, length)`: Rule 4 (Nelson): a window with every successive
  difference non-zero and the sign of each difference flipping relative to the last.
- `_k_of_m_beyond(z, k, m, thresh)`: Rules 5 / 6 (Nelson) and 2 / 3 (WE).

!!! warning "Guardrail: rules are skipped when σ is not usable"
    If the recovered $\sigma$ is non-positive or non-finite, `_apply_rules` returns
    **no** violations rather than dividing by zero or emitting garbage. This happens,
    for example, when every subgroup range is identical to zero. The control limits
    still print; the zone tests simply have nothing meaningful to test against.

### Where each panel runs the rules

The full run-rule engine runs on the **location** panel (X-bar / individuals /
standardized). The **dispersion** panel (R, S, or moving-range) is checked with
**Rule 1 only** (a single point beyond its control limits) so that an inflated
subgroup range surfaces even when the subgroup *mean* stays inside the X-bar limits.
On the dispersion panel the violation is described as `one point beyond control
limits`.

!!! note "One-sided dispersion and attribute limits"
    For small subgroups the lower limit on the R, S, and moving-range panels is
    zero (the constants $D_3$ / $B_3$ are zero), so that panel is effectively
    one-sided: only an *over*-dispersed subgroup can violate Rule 1. The same
    clamping applies to attribute charts, where the lower control limit is clipped
    at zero (`np.clip(cl - 3*sd, 0, None)`); a count or proportion can never be
    flagged for going below zero. Attribute charts (`p`, `np`, `c`, `u`) run **only**
    the per-point beyond-limits test (their limits vary point-to-point with the
    sample size), not the zone/run pattern rules.

## Worked example

A process running on target for the first twelve subgroups, then shifting up by
about one standard deviation. The shift is too small to be an obvious $3\sigma$
excursion every time, but the run and zone rules catch it. The data are five
measurements per lot across twenty lots, so mfgQC infers an X-bar R chart.

```python
import numpy as np, pandas as pd, mfgqc

rng = np.random.default_rng(11)
parts = []
for lot in range(1, 21):
    center = 50.0 if lot <= 12 else 50.9        # sustained shift after lot 12
    parts.append(pd.DataFrame({"x": rng.normal(center, 0.5, 5), "lot": lot}))
df = pd.concat(parts, ignore_index=True)
df["x"] = df["x"].round(3)

qc = mfgqc.load(df, measure="x", subgroup="lot", subgroup_size=5)
print(qc.control_chart(rules="nelson").report())
```

```text
Control Chart: xbar_r (inferred); rules=nelson
==============================================
Xbar: CL=50.372  UCL=50.963  LCL=49.782
R: CL=1.0233  UCL=2.1634  LCL=0

Out-of-control signals: 14
  point 5 (location): nelson_1 - one point beyond 3 sigma
  point 6 (location): nelson_1 - one point beyond 3 sigma
  point 7 (location): nelson_6 - four of five points beyond 1 sigma (same side)
  point 8 (location): nelson_6 - four of five points beyond 1 sigma (same side)
  point 9 (location): nelson_2 - nine points in a row on one side of CL
  point 10 (location): nelson_1 - one point beyond 3 sigma
  point 12 (location): nelson_5 - two of three points beyond 2 sigma (same side)
  point 14 (location): nelson_5 - two of three points beyond 2 sigma (same side)
  point 15 (location): nelson_1 - one point beyond 3 sigma
  point 16 (location): nelson_1 - one point beyond 3 sigma
  point 17 (location): nelson_1 - one point beyond 3 sigma
  point 18 (location): nelson_6 - four of five points beyond 1 sigma (same side)
  point 19 (location): nelson_1 - one point beyond 3 sigma
  point 20 (location): nelson_5 - two of three points beyond 2 sigma (same side)

Assumption checks:
  [FAIL] independence (lag-1 autocorrelation): r=0.69, p=0.00203; n=20 [low power]

Recommendations:
  - Observations are autocorrelated (lag-1 r=0.69); control limits are unreliable - consider a time-series chart (e.g. EWMA).
```

The same data under Western Electric flags the *same* points; only the rule labels
change (and `western_electric_4` replaces `nelson_2`, firing one point sooner, an
8-point run instead of 9):

```python
print(qc.control_chart(rules="western_electric").report())
```

```text
Control Chart: xbar_r (inferred); rules=western_electric
========================================================
Xbar: CL=50.372  UCL=50.963  LCL=49.782
R: CL=1.0233  UCL=2.1634  LCL=0

Out-of-control signals: 14
  point 5 (location): western_electric_1 - one point beyond 3 sigma
  point 6 (location): western_electric_1 - one point beyond 3 sigma
  point 7 (location): western_electric_3 - four of five beyond 1 sigma (same side)
  point 8 (location): western_electric_3 - four of five beyond 1 sigma (same side)
  point 9 (location): western_electric_3 - four of five beyond 1 sigma (same side)
  point 10 (location): western_electric_1 - one point beyond 3 sigma
  point 12 (location): western_electric_2 - two of three beyond 2 sigma (same side)
  point 14 (location): western_electric_2 - two of three beyond 2 sigma (same side)
  point 15 (location): western_electric_1 - one point beyond 3 sigma
  point 16 (location): western_electric_1 - one point beyond 3 sigma
  point 17 (location): western_electric_1 - one point beyond 3 sigma
  point 18 (location): western_electric_3 - four of five beyond 1 sigma (same side)
  point 19 (location): western_electric_1 - one point beyond 3 sigma
  point 20 (location): western_electric_2 - two of three beyond 2 sigma (same side)
```

!!! tip "Read the assumption block alongside the signals"
    The independence check failed here because a sustained shift *is* a form of
    serial dependence: successive points stay on the same side. That is the special
    cause the rules are flagging, not an artifact. For genuinely autocorrelated
    in-control processes, the recommendation to use a time-series chart (EWMA) is the
    point: the standard $3\sigma$ limits and these run rules both assume independent
    points.

## Choosing between the sets

- **Western Electric** is the older, leaner set: Rule 1 plus the three zone tests
  plus an 8-point run. It is the historical baseline and what many quality manuals
  and audits expect.
- **Nelson** extends it with explicit tests for trend (Rule 3), over-control
  oscillation (Rule 4), stratification (Rule 7) and mixtures (Rule 8), and lengthens
  the same-side run to 9.

More tests means a higher chance of a false alarm on a genuinely in-control process;
fewer tests means slower detection of subtle special causes. mfgQC does not choose
for you and does not blend the sets; it applies exactly the set you name and tells
you which rule fired. See [Choosing a control chart](../guide/choosing-a-control-chart.ipynb)
for picking the chart the rules run on.

## Source standards

- The four zone/run tests follow **Western Electric Co.**, *Statistical Quality
  Control Handbook* (1956), the original "Western Electric rules."
- The eight-rule set follows **Nelson, L. S.** (1984), "The Shewhart Control Chart,   Tests for Special Causes," *Journal of Quality Technology* 16(4).

Both are listed in the [Bibliography](bibliography.md). The control-chart constants
that set the limits the zones are measured against come from Montgomery; see
[Control charts](control-charts.md).
