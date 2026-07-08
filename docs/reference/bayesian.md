# Bayesian analytics

Classical capability hands you one number and a wide confidence interval that is easy to
misread. The Bayesian layer answers the question a quality engineer actually asks:
**given the parts I measured, what is the probability this process meets the
requirement?** It returns a posterior distribution over the quantity you care about
($P_{pk}$, a defect rate, a mean shift), so every headline comes with a direct
probability statement and a credible interval, and small samples and prior knowledge
enter the calculation explicitly rather than being assumed away.

Everything on this page is pinned to the `mfgqc/bayes/` modules. The engines are
**conjugate and deterministic**: closed-form posterior updates, with a *seeded* Monte
Carlo step used only to push posterior draws through a nonlinear function such as
$P_{pk}$. There is no MCMC, so results are bit-reproducible from the seed.

!!! note "This layer is opt-in and additive"
    The Bayesian functions live under `mfgqc.bayes` and never change what the classical
    analyses do. Reach for them when the sample is small, when you have credible prior
    information, or when you need a probability of conformance rather than a point index.

## 1. The one idiom

The Bayesian layer follows the same shape as the rest of mfgQC: a function produces a
result object, and the object has methods. The one addition is that a *posterior* result
also answers probability and interval questions directly.

```python
import numpy as np
from mfgqc.bayes import capability_from_values

y = [1.51, 1.546, 1.477, 1.403, 1.455, 1.391, 1.517, 1.671, 1.451, 1.436, 1.569, 1.553]

cap = capability_from_values(y, lower=1.0, upper=2.0, target=1.5, seed=1)

cap.report()             # full text: posterior summary + assumption checks
cap.prob("ppk", 1.33)    # P(Ppk >= 1.33) as (probability, MC std error)
cap.interval("ppk", 0.90)# 90% credible interval for Ppk
cap.summary()            # flat dict of the posterior scalars
cap.to_dict()            # full JSON-serializable payload (consume this from code)
```

`prob(quantity, threshold, direction=">=")` and `interval(quantity, level)` are the two
methods you will use most: they read directly off the draws, so
you never re-derive a probability from a point estimate and a standard error.

!!! important "Every Monte Carlo result takes a `seed`"
    Any function that draws (capability, comparison, assurance, monitoring) requires
    `seed=`. Same seed and same data give byte-identical draws, which is what makes a
    Bayesian result auditable. The reported `+/- ...(MC)` on a probability is the Monte
    Carlo standard error, not posterior uncertainty; shrink it with more `draws=`.

## 2. Bayesian capability

`capability_from_values` (`mfgqc/bayes/capability.py`) fits a Normal process with a
conjugate **Normal-Inverse-$\chi^2$** model and reports the posterior of the performance
indices. It is the Bayesian counterpart of [`capability()`](capability.md), built for the
small-$n$ case where the classical CI is too wide to act on.

**The conjugate update.** With prior parameters $(\mu_0, \kappa_0, \nu_0, \sigma_0^2)$ and
data of size $n$ with sample mean $\bar y$ and variance $s^2$, the posterior parameters
are (`mfgqc/bayes/conjugate.py`, `update`):

$$
\kappa_n = \kappa_0 + n, \qquad
\mu_n = \frac{\kappa_0\,\mu_0 + n\,\bar y}{\kappa_n}, \qquad
\nu_n = \nu_0 + n
$$

$$
\nu_n\,\sigma_n^2 = \nu_0\,\sigma_0^2 + (n-1)\,s^2
                    + \frac{\kappa_0\,n}{\kappa_n}\,(\bar y - \mu_0)^2
$$

The marginal posteriors are then $\sigma^2 \mid y \sim \text{Inv-}\chi^2(\nu_n, \sigma_n^2)$
and $\mu \mid y \sim t_{\nu_n}\!\big(\mu_n,\ \sigma_n^2/\kappa_n\big)$. A noninformative
analysis uses the limiting reference prior $\kappa_0=\nu_0=0$, which reproduces the
classical estimates as posterior points while adding interval width from the finite sample.

**From draws to $P_{pk}$.** The indices are nonlinear in $(\mu, \sigma)$, so mfgQC draws
from the posterior in a fixed order (bit-reproducible) and evaluates the index per draw:

$$
\sigma^2_{(i)} = \frac{\nu_n\,\sigma_n^2}{X_i},\quad X_i \sim \chi^2_{\nu_n};
\qquad
\mu_{(i)} \sim \mathcal N\!\Big(\mu_n,\ \sigma^2_{(i)}/\kappa_n\Big)
$$

$$
P_{pk(i)} = \min\!\left(\frac{\text{USL} - \mu_{(i)}}{3\,\sigma_{(i)}},\;
                        \frac{\mu_{(i)} - \text{LSL}}{3\,\sigma_{(i)}}\right)
$$

The credible interval is the empirical quantile of $\{P_{pk(i)}\}$, and
`prob("ppk", 1.33)` is the fraction of draws at or above 1.33.

**Worked example — 12 parts.** A short run of twelve measurements, spec $[1.0, 2.0]$,
target $1.5$:

```python
import numpy as np
from mfgqc.bayes import capability_from_values

y = [1.51, 1.546, 1.477, 1.403, 1.455, 1.391, 1.517, 1.671, 1.451, 1.436, 1.569, 1.553]
cap = capability_from_values(y, lower=1.0, upper=2.0, target=1.5, seed=1)
print(cap.report())
```

```text
Bayesian Capability (noninformative)
====================================
n = 12   mean = 1.4982   s (overall) = 0.07956
mu 95% credible interval = (1.4477, 1.5488)
Pp  = 2.095    Ppk point = 2.088
Ppk 95% credible interval = (1.15, 2.88)
P(Ppk >= 1.33) = 0.931 +/- 0.0008 (MC)
ppm point = 0

Assumption checks:
  [PASS] small_sample (n >= 8): n=12; n=12 [low power]
  [PASS] normality (Anderson-Darling): AD=0.225, p=0.768; skew 0.614; n=12 [low power]
```

Reading it:

- **The headline is a probability.** $P(P_{pk} \ge 1.33) = 0.931$. This is the quantity a
  decision needs, not the point index alone.
- **The interval reflects the small $n$.** The point $P_{pk}$ is 2.09, but the 95%
  credible interval runs $(1.15,\ 2.88)$. Twelve parts do not pin the index down,
  and the result reports this rather than a single decimal.
- **The guardrails still fire.** Normality and small-sample checks run exactly as they do
  in the classical layer, flagged `[low power]` because twelve points cannot test much.

```python
cap.prob("ppk", 1.33)     # (0.931, 0.0008)   probability, MC std error
cap.interval("ppk", 0.90) # (1.267, 2.723)    90% credible interval
```

## 3. Priors and the prior-data conflict guardrail

A prior is how you fold in what you already know: a validated process history, a
qualified material, a supplier's data. Priors are built in `mfgqc/bayes/priors.py` and
passed as `prior=`.

| Prior | Model | Build it from |
| --- | --- | --- |
| `NormalPrior(mu0, k0, nu0, s20)` | Normal capability | direct parameters, or `NormalPrior.from_interval(...)` / `from_expectations(...)` |
| `BetaPrior(a, b)` | proportion | pseudo-counts of fails/passes; default Jeffreys $\text{Beta}(0.5, 0.5)$ |
| `GammaPrior(a, b)` | rate | pseudo-counts and pseudo-exposure; default Jeffreys $\text{Gamma}(0.5, 0)$ |

`k0` and `nu0` are the *strength* of a `NormalPrior`, expressed as an equivalent prior
sample size: `k0=20` says "trust the prior mean as much as twenty parts." When you supply
a prior, mfgQC reports how much of the posterior it drove and tests it against the data:

```text
Bayesian Capability (informative)
=================================
...
Assumption checks:
  [FAIL] prior_weight (prior weight w = k0/(k0+n)): prior=0.625; n=12
  [PASS] prior_data_conflict (prior predictive t on ybar): prior=0.0436, p=0.966; n=12

Recommendations:
  - The prior contributes 62% of the posterior; the report is prior dominated. Widen the
    prior or collect more data.
```

Two checks on an informative prior:

- **`prior_weight`** flags when the prior drives more than half the posterior
  ($w = \kappa_0/(\kappa_0+n)$). Here twelve parts against a `k0=20` prior means the prior
  is doing 62% of the work; mfgQC says so rather than letting you present a
  prior-dominated number as data.
- **`prior_data_conflict`** evaluates the prior-predictive $t$ density at the observed
  mean. A small $p$ means the data landed in the tail of what the prior expected, i.e.
  the prior and the parts disagree; here $p=0.97$, so they are consistent.

!!! warning "A prior is a recorded input, not a hidden adjustment"
    The prior, its strength, and both conflict checks are recorded in the result's
    provenance. State where a prior came from. If `prior_data_conflict` fails, do not
    proceed on the informative result; reconcile the prior against the process first.

## 4. Attributes: proportion and rate

For pass/fail and defect-count data, `mfgqc/bayes/attributes.py` gives the two standard
conjugate models. Both are closed-form (no seed), and both answer *"is the process inside
the requirement?"* as a probability.

**Proportion** — Beta-Binomial. With a $\text{Beta}(a, b)$ prior and $y$ failures in $n$
trials, the posterior is $\text{Beta}(a+y,\ b+n-y)$, and `prob_within_spec` reports
$P(p \le p_{\max})$, the posterior probability that the true defect rate is within your
limit `max_proportion`, from the Beta CDF:

```python
from mfgqc.bayes import proportion_capability
proportion_capability(n_fail=3, n_trials=200, max_proportion=0.05).report()
```

```text
Bayesian Proportion Capability
==============================
n = 200   failures = 3   posterior mean p = 0.01741
p 95% credible interval = (0.004242, 0.03948)
prior family = beta
P(p <= 0.05) = 0.995
```

Three failures in two hundred gives a 99.5% posterior probability that the true defect
rate is under 5% — a defensible release statement, where a raw $3/200 = 1.5\%$ point
estimate says nothing about the uncertainty. `proportion_capability` also accepts a raw
0/1 (or labelled) vector via `data=` and a `mapping=`.

**Rate** — Gamma-Poisson. With a $\text{Gamma}(a, b)$ prior and total count $\sum k$ over
total exposure $\sum t$, the posterior is $\text{Gamma}(a + \sum k,\ b + \sum t)$:

```python
from mfgqc.bayes import rate_capability
rate_capability(counts=[2, 1, 0, 3, 1], exposures=[100]*5, max_rate=0.05).report()
```

```text
Bayesian Rate Capability
========================
k = 5   total count = 7   total exposure = 500
posterior mean rate = 0.015
rate 95% credible interval = (0.006262, 0.02749)
prior family = gamma
P(rate <= 0.05) = 1
```

The rate model additionally runs a **dispersion** check (`check_dispersion`, Poisson
family): if the counts are over-dispersed relative to a Poisson, the single-rate model is
too simple and the check flags it.

## 5. Comparing two processes

`compare` (`mfgqc/bayes/comparison.py`) takes two fitted capability results and reports
the probability that one is better than the other, drawing both from their posteriors
with complementary seeds so `compare(a, b)` and `compare(b, a)` agree exactly.

```python
from mfgqc.bayes import capability_from_values, compare
cap_a = capability_from_values(a, lower=6, upper=14, target=10, seed=1)
cap_b = capability_from_values(b, lower=6, upper=14, target=10, seed=2)
compare(cap_a, cap_b, seed=1, labels=("line A", "line B")).report()
```

```text
Bayesian Comparison (line B vs line A)
======================================
P(line B mean > line A mean) = 1
P(line B sd < line A sd)     = 0.218
P(line B Ppk > line A Ppk)   = 0.188
delta mean (B - A) = 1.066   95% (0.584, 1.549)
delta Ppk  (B - A) = -0.223   95% (-0.722, 0.268)
```

This answers "is B better?" without a null-hypothesis test: the mean is almost certainly
higher on B (probability 1.00), but B is *not* more capable (only a 19% chance its
$P_{pk}$ is higher), and the $P_{pk}$ difference credibly spans zero. This reports more
than a single $p$-value: the mean difference is nearly certain, the capability difference
is not.

## 6. Decisions: assurance and guardband

Two functions in `mfgqc/bayes/decisions.py` turn a posterior into a decision.

**Assurance — how many parts to collect.** `assurance` answers *"how large a sample do I
need to have a good chance of demonstrating capability?"* It is the Bayesian analogue of a
power calculation: for each candidate $n$ it simulates future datasets from the current
posterior, refits, and reports the probability the study will clear your decision rule.

```python
from mfgqc.bayes import assurance
assurance(cap, target=("ppk", 1.33), decide=(0.9, 0.1),
          n_grid=(20, 50, 100, 200), sims=300, inner_draws=1000, seed=1).report()
```

```text
Bayesian Assurance (sample size)
================================
target: P(ppk >= 1.33) exceeds 0.90
  n =   20: assurance 0.677
  n =   50: assurance 0.793
  n =  100: assurance 0.867
  n =  200: assurance 0.877
recommended n = None
Note: predictive-simulation machinery is textbook standard; the assurance framing is
from the clinical-trials literature; the validation is self-consistency (monotonicity).
```

Here even 200 parts only reaches 88% assurance, so no $n$ in the grid clears the 90% bar
and `recommended n` is `None`. This is the useful outcome: the calculation identifies when
a process is too marginal for any feasible sample to demonstrate capability, rather than
returning a number that implies otherwise.

**Guardband — acceptance limits under gauge error.** When the gauge itself has scatter,
accepting exactly to the spec limits passes some bad parts and scraps some good ones.
`guardband` finds the acceptance limits that minimize expected cost, trading the cost of
scrapping a good part (`c_scrap`) against the cost of shipping a bad one (`c_escape`):

```python
from mfgqc.bayes import guardband
guardband(cap, sigma_gauge=0.02, c_scrap=1.0, c_escape=20.0, seed=1).report()
```

```text
Bayesian Guardband (acceptance limits)
======================================
gauge sd = 0.02   costs: scrap 1, escape 20
optimal accept limits = (1.026, 1.974)
  expected cost = 7.6e-05   scrap = 0.00546%   escape = 1 ppm
naive limits (= spec) = (1, 2)
  expected cost = 0.0001843   scrap = 0.00142%   escape = 9 ppm
```

With escapes twenty times costlier than scrap, the optimizer pulls the acceptance limits
*inward* to $(1.026,\ 1.974)$, cutting expected cost by more than half versus inspecting
to the raw spec. The guardband is derived from the *posterior* predictive distribution of
the part, so the gauge error and the process uncertainty are both accounted for.

## 7. Baseline and monitoring

The monitoring functions (`mfgqc/bayes/monitoring.py`) implement a two-step Bayesian SPC
workflow: freeze a baseline from a clean Phase-1 dataset, then test each new subgroup
against it.

```python
from mfgqc.bayes import phase1, monitor
reference = phase1(baseline_values)               # frozen, hash-verified baseline
monitor(reference, new_subgroups, seed=1).report()
```

```text
Bayesian Monitoring
===================
reference digest = 31e5eb89af73...
tests = mean, sd, min, max, lag1_autocorr   alpha = 0.005   R = 10000
subgroups = 4   flagged = 1
  sub-000: mean=0.633  sd=0.501  min=0.9   max=0.431  lag1_autocorr=0.947
  sub-001: mean=0.818  sd=0.0908 min=0.253 max=0.313  lag1_autocorr=0.643
  sub-002: mean=0.302  sd=0.502  min=0.122 max=0.597  lag1_autocorr=0.669
  sub-003: mean=0      sd=0.637  min=0     max=0.0018 lag1_autocorr=0.826  [FLAG]
family-wise alpha per subgroup (k=5): 1-(1-alpha)^k = 0.0248
```

Each cell is a **posterior-predictive $p$-value**: the probability, under the frozen
baseline, of seeing a subgroup statistic as extreme as this one. A cell near 0 or 1 is a
signal. Subgroup 3 (a genuine mean shift) shows `mean=0` and `min=0` and is flagged; the
in-control subgroups sit mid-range. The `alpha` is per-test and the report shows the
family-wise rate across the five tests so you can read the effective false-alarm budget.

`predictive_check` (`mfgqc/bayes/predictive_check`) is the one-off cousin: it checks
whether a single dataset is consistent with the fitted model on a chosen statistic
(`min`, `max`, `mean`, ...), following the posterior-predictive-check recipe of BDA3 §6.3.

```text
Posterior Predictive Check
==========================
statistic T = min   T(y_obs) = -2.11
Bayesian p-value P(T_rep >= T_obs) = 0.376
two-sided p = 0.752   (n = 40, R = 10000)
```

## 8. Harder data: censored, pooled, short-run

Three functions cover cases the closed-form engine cannot.

**Censored / truncated data** — `capability_censored` (`mfgqc/bayes/censored.py`). When a
gauge saturates (readings pile up at a limit) or the population is truncated (parts
outside a window never reach you), ignoring it biases capability. This function fits
$(\mu, \sigma)$ on a **grid posterior** with a likelihood that handles observed points,
censored tails, and truncation exactly, then reports $P_{pk}$ from the grid. Pass the
censoring with a `Censoring(lower, upper, flag)` record:

```text
Bayesian Capability (censored/truncated, grid)
==============================================
n_total = 40   n_observed = 39   n_censored = 1 (left 0, right 1)
mu 95% credible interval = (1.5266, 1.6016)
Pp  = 1.413    Ppk (posterior median) = 1.232
Ppk 95% credible interval = (0.943, 1.54)
grid: shape (401, 401), refinements 1, converged True
censoring limits = (lower None, upper 1.75)
```

The grid auto-refines until the posterior is resolved (`converged True`); the report tells
you the resolution and whether it settled.

**Pooled / hierarchical capability** — `pooled_capability` (`mfgqc/bayes/pooled.py`).
Several small groups (cavities in a mold, heads on a filler, positions on a fixture) each
have too few parts to trust alone. A hierarchical model (BDA3 §5.4) **borrows strength**
across them: each position's estimate is pulled toward the family mean by an amount the
data decide, so a cavity with six parts is estimated far more stably than in isolation.

```text
Bayesian Pooled Capability (hierarchical)
=========================================
positions = 5   pooled within sd = 0.1088
P(min_j Cpk_j >= 1.33) = 0.444 +/- 0.0016 (MC)
  position 0: n=6 mean=1.535 Cpk~1.42 mean 95% CI=(1.443,1.59)
  ...
  position 3: n=6 mean=1.425 Cpk~1.3  mean 95% CI=(1.372,1.52)
```

The headline `P(min_j Cpk_j >= 1.33)` is the probability that the *worst* position clears
the requirement — the number that matters when any bad cavity fails the whole part.

**Short-run charts** — `shortrun` (`mfgqc/bayes/shortrun.py`). For runs too short to build
a classical chart, this chains a conjugate posterior stage to stage (each subgroup's
posterior becomes the next one's prior) and flags a stage when
$P(|\mu - \text{target}| > d)$ exceeds `p_star`. A proper prior is required so the first
short run has a scale; a noninformative start is allowed only with `allow_vague=True`.

```text
Bayesian Short-Run Chart
========================
stages = 5   target = 25   d = 0.025   p* = 0.90
prior = normal
first flag = none (in control)
  stage 4: mean=25.054  P(|mu-target|>d)=0.00013
```

!!! warning "Short-run charts absorb slow drift"
    Because each posterior becomes the next prior, a slow sustained drift is gradually
    absorbed into the reference and may signal late or not at all. The report carries this
    caveat verbatim; pair the chart with a fixed-target rule when catching slow drift
    matters.

## 9. Reproducibility and provenance

Every Bayesian result is a [`QCResult`](provenance.md): it carries an immutable,
hash-chained history, and its Monte Carlo draws are fully determined by `seed` and the
data. Two guarantees follow:

- **Bit-reproducible.** Re-running with the same `seed` and data gives an identical
  `provenance_digest()`. Change one measurement and the digest changes.
- **Auditable priors.** A prior's parameters, its strength, and the prior-data-conflict
  verdict are recorded in the history, so a reviewer can see exactly what prior knowledge
  entered the number.

```python
saved = cap.provenance_digest()
cap.verify_provenance(saved)   # True; tamper with the data and it is False
```

Consume results through `summary()` / `to_dict()`, never by parsing the report text; the
report is for humans, the dict is the API.

## 10. Assumptions the Bayesian layer checks

| Check | Where | What it means |
| --- | --- | --- |
| **normality** | capability | Anderson-Darling on the data; the Normal model must fit |
| **small_sample** | capability, pooled | flags low $n$ where posteriors are wide and prior-sensitive |
| **prior_weight** | informative capability | flags when the prior drives more than half the posterior |
| **prior_data_conflict** | informative capability | prior-predictive $t$ at $\bar y$; small $p$ = prior disagrees with data |
| **dispersion** | rate | over-dispersion relative to Poisson breaks the single-rate model |
| **censoring_fraction** | censored | flags when more than half the data are censored |

As everywhere in mfgQC, a failed check is **reported, never silently corrected**. The
number is still computed; the guardrail tells you how far to trust it.

## 11. Reproducing the published examples

The conjugate engines are validated by re-running worked examples from the source texts
and matching the printed results. Each block below runs the published problem and shows
mfgQC's result; the comment names the source and the value that source publishes. Every
number shown was produced by the code above it.

**Proportion — placenta previa (source: BDA3 §2.4).** 437 female of 980 births, uniform
prior.

```python
from mfgqc.bayes import proportion_capability
from mfgqc.bayes.priors import BetaPrior

# BDA3 §2.4 publishes: posterior mean 0.446, 95% interval [0.415, 0.477].
r = proportion_capability(n_fail=437, n_trials=980, prior=BetaPrior(1.0, 1.0))
print(round(r.mean, 3), tuple(round(v, 3) for v in r.interval(0.95)))
```

```text
0.446 (0.415, 0.477)
```

**Rate — asthma mortality (source: BDA3 §2.6).** 3 deaths at exposure 2.0, Gamma(3, 5)
prior.

```python
from mfgqc.bayes import rate_capability
from mfgqc.bayes.priors import GammaPrior

# BDA3 §2.6 publishes: posterior Gamma(6, 7), mean 0.857, P(rate > 1) = 0.30.
r = rate_capability([3.0], exposures=[2.0], prior=GammaPrior(3.0, 5.0))
print((int(r.a_post), int(r.b_post)), round(r.mean, 3), round(r.prob(1.0, ">="), 2))
```

```text
(6, 7) 0.857 0.3
```

**Rate engine — birth rates, two groups (source: Hoff §3.2.2).** Gamma(2, 1) prior; group
sums 217 over 111 people and 66 over 44.

```python
# Hoff §3.2.2 publishes: Gamma(219, 112) mean 1.955, Gamma(68, 45) mean 1.511.
g1 = rate_capability([217.0], exposures=[111.0], prior=GammaPrior(2.0, 1.0))
g2 = rate_capability([66.0],  exposures=[44.0],  prior=GammaPrior(2.0, 1.0))
print((int(g1.a_post), int(g1.b_post)), round(g1.mean, 3),
      (int(g2.a_post), int(g2.b_post)), round(g2.mean, 3))
```

```text
(219, 112) 1.955 (68, 45) 1.511
```

**Hierarchical pooling — eight schools (source: BDA3 §5.5).** The eight
(effect, standard error) pairs from BDA3 Table 5.2.

```python
from mfgqc.bayes import hierarchical_normal

# BDA3 §5.5 publishes the complete-pooling estimate: 7.7, standard error 4.1.
y = [28, 8, -3, 7, -1, 1, 18, 12]
s = [15, 10, 16, 11,  9, 11, 10, 18]
mean, se = hierarchical_normal(y, s, seed=1).pooled_estimate()
print(round(mean, 1), round(se, 1))
```

```text
7.7 4.1
```

Two more source examples are matched without a full reproduction here, because the source
does not tabulate the raw data. The Normal capability engine reproduces BDA3 §3.2
(Newcomb's speed of light): its posterior for the mean is the Student-$t$ marginal
$t_{65}$ with a 0.975 multiplier of 1.997, giving a 95% interval of $[23.5, 28.9]$ against
BDA3's published $[23.6, 28.8]$ (the 66 measurements appear only as a histogram, so this
uses the published moments $\bar y = 26.2$, $s = 10.8$). The posterior-predictive check
reproduces the qualitative result of BDA3 §6.3: Newcomb's minimum, $-44$, sits far below
every replicated minimum, so the min statistic flags the misfit while the mean statistic
does not.

The remaining functions follow the cited methods, but no source prints a manufacturing
worked example to reproduce, so each is checked by an internal identity or simulation
rather than against a published number:

- `compare` — posterior draws (BDA3 §3.3); complementary probabilities agree exactly, and
  the direction matches the Hoff §8.1 two-group example.
- `assurance` — predictive simulation (framing: O'Hagan et al., 2005); assurance rises
  monotonically with $n$.
- `guardband` — decision-theoretic expected cost (BDA3 ch. 9-10); the optimal expected
  cost never exceeds the cost at the naive spec limits.
- `capability_censored`, `fit_normal_grid` — grid posterior with censored/truncated
  likelihoods (BDA3 §8.7); reproduces the closed-form engine on uncensored data and
  recovers parameters on simulated censored data.
- `shortrun` — sequential conjugate updating (Colosimo & del Castillo); the chained
  posterior equals a single batch update, and the chart statistic equals a direct
  $t$-CDF evaluation.
- Prior elicitation (`from_expectations`) follows Hoff §5.5: with $n_0 = 1$ the
  expectation-based prior gives $E[\sigma^2]$ equal to the elicited variance.

## 12. Source standards

The Bayesian engines are pinned to three texts (full citations in the
[Bibliography](bibliography.md)):

- **Gelman et al., *Bayesian Data Analysis*, 3rd ed. (BDA3)** — the Normal-Inverse-$\chi^2$
  conjugate model and its marginals (§3.3), Beta-Binomial (§2.4) and Gamma-Poisson (§2.6)
  attributes, hierarchical Normal pooling (§5.4), censored/truncated likelihoods (§8.7),
  and posterior-predictive checks (§6.3).
- **Hoff, *A First Course in Bayesian Statistical Methods*** — prior elicitation and the
  interpretation of $(\kappa_0, \nu_0)$ as prior sample size (§5.5).
- **Colosimo & del Castillo, *Bayesian Process Monitoring, Control and Optimization*** —
  the process-monitoring and short-run framing. The assurance sample-size framing follows
  the clinical-trials assurance literature (O'Hagan et al.).

## See also

- [Capability](capability.md): the classical $C_p$/$C_{pk}$/$P_{pk}$ layer this mirrors.
- [Provenance model](provenance.md): the immutability and hash-chain guarantees every
  Bayesian result inherits.
- [Bayesian capability walkthrough](../guide/bayesian-capability.ipynb): a runnable,
  narrative version of §2-§3 for a small-sample release decision.
- [API reference](api.md): result-object fields, `summary()`, `to_dict()`, `prob()`,
  `interval()`.
- [Bibliography](bibliography.md): the cited Bayesian sources.
