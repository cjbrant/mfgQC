---
title: Power & sample size
---

# Power and sample size

Before you run a test you can ask how big a sample it needs. A test that is too small
fails to detect a difference that is really there. A test that is too large spends
parts and time you did not need to spend. Power and sample-size planning answers the
question quantitatively: given the size of difference you care about and the risks you
are willing to take, how many measurements should you collect.

This is pre-data planning. It happens before a single part is measured, which is why
the planning engine sits parallel to [design of experiments](doe.md) rather than being
a method on a loaded table. The functions live in the `mfgqc/power/` subpackage
(`solve.py`); this page pins every formula and default to that code.

A note on what the numbers rest on. A power calculation needs an effect size you have
not yet observed and a variance you are estimating. mfgQC treats both as planning
inputs, not data-checked facts. Every `PowerResult` carries two assumption records that
say so plainly:

```text
Recommendations:
  - the effect size is a planning assumption, not measured from data.
  - the variance / baseline rate is an estimate; revisit the plan when phase-I data arrive.
```

mfgQC reports these. It does not pick an effect size for you.

---

## 1. The core idea: solve for the missing one

Four quantities sit in a fixed relationship for any one test:

- the **effect size**, the difference worth detecting, expressed in standardized units,
- **alpha**, the false-alarm risk (the chance of declaring a difference when there is
  none), default $0.05$,
- **power**, the chance of detecting the difference when it is real (one minus the
  miss rate, beta),
- the **sample size** $n$.

Fix any three and the fourth is determined. mfgQC fixes `alpha` separately (it is
always a keyword you set, never the unknown) and solves for whichever of `effect`,
`n`, or `power` you leave as `None`. Exactly one of the three must be left out, or the
call raises a `ValueError` telling you so.

```python
import mfgqc
plan = mfgqc.power.t_test(effect=0.5, power=0.80)   # n is None, so solve for n
```

The relationship is not closed-form. Under the null hypothesis the test statistic
follows a central $t$, $F$, or normal distribution, and the critical value is read off
that. Under the alternative the same statistic follows a *noncentral* distribution: the
true effect shifts it away from zero by a noncentrality parameter that grows with both
the effect size and $\sqrt{n}$. Power is the area of that shifted distribution beyond
the critical value. mfgQC computes power directly from scipy's noncentral
distributions (`scipy.stats.nct`, `ncf`, and the normal `norm`), and when the unknown
is `n` or `effect` it finds the value by root-finding.

The root-finder (`_solve` in `solve.py`) uses the fact that power increases
monotonically in $n$ and in the effect size. It brackets the target by doubling the
variable up from a small starting value until power crosses the target, then calls
Brent's method (`scipy.optimize.brentq`) on that bracket. If the target power is not
reachable over the search range, it raises a `ValueError` rather than returning a
misleading number. A target power must lie strictly between `alpha` and $1$; otherwise
the call is rejected.

The solved $n$ is reported as a real number, and the report adds the smallest integer
that reaches or exceeds the target (`math.ceil`), since you cannot measure a fractional
part:

```text
recommended n = 64 per group (smallest integer reaching the target power)
```

When `effect` is the solved quantity, it is reported as the **minimum detectable
effect** at the given $n$ and power, with a note that mfgQC does not choose an effect
size for you.

---

## 2. The t-test (`t_test`)

### What it is and when you use it

Use a $t$-test plan when the response is a measured value (a width, a torque, a
temperature) and you are comparing means: one mean against a target (`one-sample`),
two means against each other (`two-sample`, the default), or before-and-after pairs
(`paired`).

### The effect size

The effect is **Cohen's d**, the standardized mean difference: the difference in means
divided by the standard deviation. The `effect` argument is `d` directly. Standardizing
this way is what lets the plan run before you know the raw spread in your units. As a
rough convention $d = 0.2$ is a small shift, $0.5$ medium, $0.8$ large, but the right
value is the smallest shift that matters for your process, which only you can set.

### The noncentral t

For a two-sample test with $n$ per group, the degrees of freedom are $2n - 2$ and the
noncentrality parameter is

$$\delta = d \sqrt{n/2}.$$

For one-sample and paired tests the degrees of freedom are $n - 1$ and
$\delta = d\sqrt{n}$. Power is the noncentral-$t$ tail beyond the central-$t$ critical
value. For the two-sided case (the default) mfgQC sums both tails:

$$\text{power} = P\!\left(T'_{\delta} > t_{1-\alpha/2,\,df}\right)
              + P\!\left(T'_{\delta} < -t_{1-\alpha/2,\,df}\right),$$

where $T'_{\delta}$ is noncentral $t$ with that $df$ and noncentrality $\delta$
(`_t_power` in `solve.py`, via `stats.nct`). For `alternative="one-sided"` it uses the
single upper tail at $t_{1-\alpha,\,df}$.

### Worked example: solve for n

You want to detect a half-standard-deviation shift in means ($d = 0.5$) between two
groups, with 80 percent power at the default $\alpha = 0.05$, two-sided. How many parts
per group.

```python
import mfgqc
plan = mfgqc.power.t_test(effect=0.5, power=0.80)
print(plan.report())
```

```text
Power / sample size (t_test): solved for n
==========================================
solved for: n
Cohen's d = 0.5   n = 63.76561019095242 (per group)   total n = 127.531
power = 0.8   alpha = 0.05   test = two-sample   alternative = two-sided
recommended n = 64 per group (smallest integer reaching the target power)

Assumption checks:
  [PASS] effect_size (planning input): planning input; n=0 [low power]
  [PASS] variance (planning input): planning input; n=0 [low power]

Recommendations:
  - the effect size is a planning assumption, not measured from data.
  - the variance / baseline rate is an estimate; revisit the plan when phase-I data arrive.
```

The answer is 64 per group (128 total). The fractional $63.77$ is the exact crossing
point; rounding up to 64 is what reaches the target. Checking both integers confirms
the boundary:

```python
mfgqc.power.t_test(effect=0.5, n=63).power   # 0.7951683381233381
mfgqc.power.t_test(effect=0.5, n=64).power   # 0.8014595579222545
```

At $n = 63$ power is just under the target; at $n = 64$ it just clears it.

### Assumptions

The $t$-test plan assumes the two groups are independent (for `two-sample`), the
measurements are approximately normal, and the two groups share a common variance. The
variance you feed in (through the standardized $d$) is an estimate; the plan is only as
good as that estimate. mfgQC records these as planning inputs rather than checking them,
because there is no data yet to check.

---

## 3. One-way ANOVA (`anova`)

### What it is and when you use it

Use an ANOVA plan when you are comparing the means of more than two groups at once: $k$
machines, $k$ suppliers, $k$ settings of a factor. You pass the number of groups as
`groups`, and the plan returns the per-group sample size.

### The effect size

The effect is **Cohen's f**, which measures how spread out the group means are relative
to the within-group standard deviation. Larger $f$ means the group means are farther
apart and easier to detect. As a rough convention $f = 0.10$ is small, $0.25$ medium,
$0.40$ large. (Cohen's $f$ relates to eta-squared by
$f = \sqrt{\eta^2 / (1 - \eta^2)}$, but the `effect` argument is $f$.)

### The noncentral F

With $k$ groups and $n$ per group the numerator degrees of freedom are $k - 1$ and the
denominator degrees of freedom are $k(n - 1)$. The noncentrality parameter is

$$\lambda = f^2 \, k \, n,$$

and power is the noncentral-$F$ tail beyond the central-$F$ critical value at $1-\alpha$
(`_anova_power` in `solve.py`, via `stats.ncf`):

$$\text{power} = P\!\left(F'_{\lambda} > F_{1-\alpha,\,k-1,\,k(n-1)}\right).$$

The ANOVA $F$-test is one-sided in $F$ by construction, so there is no two-sided
variant here.

### Example

Four groups, a medium effect $f = 0.25$, 80 percent power at $\alpha = 0.05$:

```python
mfgqc.power.anova(groups=4, effect=0.25, power=0.80)
```

```text
solved for: n
Cohen's f = 0.25   n = 44.59927430609987 (per group)   total n = 178.397
power = 0.8   alpha = 0.05   test = one-way   alternative = two-sided
recommended n = 45 per group (smallest integer reaching the target power)
```

That is 45 per group, 180 total across the four groups.

### Assumptions

The ANOVA plan assumes independent groups, approximately normal responses, and equal
variances across groups (homoscedasticity). It assumes equal per-group $n$, since the
solver returns one number for every group.

---

## 4. Proportions (`proportion`)

### What it is and when you use it

Use a proportion plan when the response is pass/fail and you are comparing rates: a
defect rate against a target (`kind="one-sample"`), or two defect rates against each
other (`kind="two-sample"`, the default). You pass the two proportions directly, `p1`
and `p2`, and the effect is fixed by their difference. Because both proportions are
given, the proportion solver only solves for `n` or `power`, not for the effect.

### Normal approximation

mfgQC uses the normal approximation to the binomial here, not an exact test. Every
`proportion` result is flagged `approximate=True` and the report carries a note:

```text
NOTE: normal approximation; it degrades for small n*p (few expected successes).
```

For the two-sample case, with pooled proportion $\bar p = (p_1 + p_2)/2$ and
$z = z_{1-\alpha/2}$, power is

$$\text{power} = \Phi\!\left(
  \frac{|p_1 - p_2|\sqrt{n} - z\sqrt{2\,\bar p (1-\bar p)}}
       {\sqrt{p_1(1-p_1) + p_2(1-p_2)}}\right),$$

where $\Phi$ is the standard normal CDF (`_prop_power` in `solve.py`). The one-sample
case compares `p1` (the alternative) against `p2` (the null) and uses
$\sqrt{p_2(1-p_2)}$ for the null variance and $\sqrt{p_1(1-p_1)}$ for the denominator.
The test is two-sided in the critical value ($z_{1-\alpha/2}$).

The approximation is trustworthy when the expected number of successes and failures in
each group is large enough (a common rule of thumb is at least about 5 to 10 of each).
With a small rate and a small $n$ it can be off, which is what the note warns about.

### Worked example: solve for power

You have a baseline defect rate of 10 percent and a process change you hope brings it to
5 percent. You can afford 400 parts per group. What power does that buy you, two-sided
at $\alpha = 0.05$.

```python
import mfgqc
plan = mfgqc.power.proportion(p1=0.10, p2=0.05, n=400)
print(plan.report())
```

```text
Power / sample size (proportion): solved for power
==================================================
solved for: power
|p1 - p2| = 0.05   n = 400 (per group)   total n = 800
power = 0.7667   alpha = 0.05   test = two-sample   alternative = two-sided
NOTE: normal approximation; it degrades for small n*p (few expected successes).
NOTE: p1=0.1, p2=0.05 (difference 0.05).

Assumption checks:
  [PASS] effect_size (planning input): planning input; n=0 [low power]
  [PASS] variance (planning input): planning input; n=0 [low power]

Recommendations:
  - the effect size is a planning assumption, not measured from data.
  - the variance / baseline rate is an estimate; revisit the plan when phase-I data arrive.
```

400 per group buys about 77 percent power for this difference, short of the usual 80
percent target. To reach 80 percent you would leave `n=None` and solve for it instead.

### Assumptions

The proportion plan assumes independent observations, a fixed sample size, and a normal
approximation to the binomial that holds only when the expected counts are not too
small. mfgQC always flags this result as approximate.

---

## 5. Two-variance F-test (`variance`)

There is a fourth solver for comparing two variances. Use it when the question is about
spread, not the mean: can a test of $n$ per group detect that one process variance is a
given multiple of another. You pass `ratio`, the true variance ratio
$\sigma_1^2 / \sigma_2^2$ you want to detect, and solve for `ratio`, `n`, or `power`.

Power is built on the central $F$ with $n - 1$ and $n - 1$ degrees of freedom and a
two-sided rejection region at $\alpha/2$ in each tail (`_var_power` in `solve.py`). For
example, detecting a doubling of variance ($\text{ratio} = 2$) at 80 percent power and
$\alpha = 0.05$:

```python
mfgqc.power.variance(ratio=2.0, power=0.80).n   # 67.32302105880645
```

That is 68 per group after rounding up. Variance tests need large samples to detect even
a sizable ratio, which the number reflects.

---

## 6. The result object

Every solver returns a `PowerResult`, a frozen dataclass that behaves like the other
mfgQC result objects:

- `.report()` prints the full text shown above (the solved quantity, the operating
  point, the assumption records).
- `.summary()` returns a flat dict: `test`, `solved_for`, `effect`, `n`, `n_total`,
  `power`, `alpha`, `kind`, `alternative`, `approximate`.
- `.view(kind="power_curve")` draws the power curve over $n$ (or over the effect when
  the effect was the solved quantity), with the operating point marked.
- `.to_dict()` gives the full JSON-serializable payload, including the provenance step.

`n_total` accounts for the group count: it is $n$ times 2 for a two-sample test, times
`groups` for ANOVA, and times 1 for a one-sample test.

---

## Source

The noncentral $t$, $F$, and normal-approximation formulas for power and sample size
are standard and are pinned to **Montgomery, _Introduction to Statistical Quality
Control_** (see the [bibliography](bibliography.md)), which covers hypothesis testing,
operating-characteristic curves, and sample-size determination for means, variances,
and proportions. The standardized effect sizes Cohen's $d$ and Cohen's $f$ and the
small/medium/large conventions follow Cohen, _Statistical Power Analysis for the
Behavioral Sciences_ (1988); that text is not in the bibliography.

## See also

- [Hypothesis testing](hypothesis-testing.md) for the tests these plans size.
- [Design of experiments](doe.md) for the parallel pre-data design engine.
- [API reference](api.md) for the full function signatures.
- [Bibliography](bibliography.md) for the cited sources.
</content>
</invoke>
