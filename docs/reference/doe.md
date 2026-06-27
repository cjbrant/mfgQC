---
title: Design of experiments
---

# Design of experiments and multi-vari

A designed experiment changes several factors on purpose, in a planned pattern, so
that the effect of each factor can be separated from the others and from noise. The
alternative, changing one factor at a time, costs more runs and cannot see how two
factors act together. This page covers the two-level factorial designs mfgQC builds
and analyzes, and the multi-vari study that decomposes observed variation into the
families a process engineer recognizes on the floor: within a piece, piece to piece,
and across time.

Two-level designs hold every factor at exactly two settings, coded $-1$ (low) and
$+1$ (high). That coding is the heart of the arithmetic: an effect is the average
change in the response as a factor moves from $-1$ to $+1$, and every interaction is
the elementwise product of factor columns.

The DOE pieces live in `mfgqc/doe/` (`design.py`, `generate.py`, `analysis.py`,
`alias.py`, `significance.py`). The multi-vari study lives in `mfgqc/multivari.py`.
Design generation happens before any data is collected; analysis is a method on a
loaded `QCData`. This page pins every formula and default to that code.

## 1. Generating a design

Generation is pre-data. You name the factors, mfgQC returns a `Design`: a coded
matrix, a run order, and, for a fraction, its full confounding structure. No run has
been spent yet. A `Design` is immutable; `center_points`, `randomize`, and
`replicate` each return a new `Design` with a provenance step appended.

### 1.1 Full factorial

A full $2^k$ factorial runs every combination of the $k$ factors at their two
levels, which is $2^k$ runs. With $k$ factors you can estimate every main effect and
every interaction up to the $k$-way interaction, all of them clear of one another.

The coded matrix is built directly in numpy as the Cartesian product of the coded
levels in standard (Yates) order, factor A varying fastest (`coded_full_matrix` in
`generate.py`): for run $i$ and factor $j$ (zero-based, A is $j=0$) the level is $+1$
when bit $j$ of $i$ is set, else $-1$.

```python
import mfgqc
from mfgqc import design

d = design.full_factorial(["A", "B", "C", "D"])
print(d.report())
```

```text
Design (full 2-level): A, B, C, D
=================================
factors = 4   base runs = 16   replicates = 1   center points = 0   total runs = 16
```

`full_factorial(factors, replicates=1, seed=None)` takes:

- **`factors`** as a list of names (`["A", "B", "C"]`) or a dict mapping each name to
  its actual `(low, high)` levels (`{"temp": (180, 220)}`). When actual levels are
  given, `run_sheet()` adds a decoded `<name>_actual` column alongside the coded one.
- **`replicates`** to run the whole design more than once. Replication is what gives a
  pure-error estimate, and therefore $t$ and $F$ tests (Section 2.2).
- **`seed`** to randomize the run order. `seed=None` keeps standard order, so
  generation is reproducible by default; passing a seed produces a fixed permutation.
  Randomizing run order guards against a lurking time trend being read as a factor
  effect.

The run sheet is one row per run in run order, with standard order kept as a column:

```python
d.run_sheet().head()
```

```text
 run  std_order    A    B    C    D
   1          1 -1.0 -1.0 -1.0 -1.0
   2          2  1.0 -1.0 -1.0 -1.0
   3          3 -1.0  1.0 -1.0 -1.0
   4          4  1.0  1.0 -1.0 -1.0
   5          5 -1.0 -1.0  1.0 -1.0
```

### 1.2 Center points

`d.center_points(n)` adds $n$ runs with every factor at coded $0$, the midpoint of
its range. Center points do two things. Their spread is a pure-error estimate that
does not depend on any model being correct. And the gap between the factorial-point
mean and the center-point mean is a test for curvature: a two-level design fits a
plane, and if the response bends, the center runs reveal it (Section 2.5). Center
points only have a defined midpoint when factor levels are numeric.

### 1.3 Fractional factorial

When the number of factors grows, a full factorial costs runs you may not be able to
afford: seven factors is 128 runs. A fractional factorial runs a carefully chosen
*subset*, a fraction $1/2^p$ of the full design, so a $2^{k-p}$ design has $2^{k-p}$
runs. The cost is **aliasing**: with fewer runs than the full set, some effects can
no longer be told apart (Section 3).

A **generator** is the rule that defines the fraction. To add factor E to a base
$2^4$ in A, B, C, D, you set its column equal to a product of existing columns, for
example `E=ABCD`. The added factor is then perfectly confounded with that product.
The **defining relation** is the set of all such confoundings implied by the
generators; the **resolution** is the length of the shortest word in it. Higher
resolution means main effects are aliased only with higher-order interactions, which
are usually negligible.

`fractional_factorial(factors, generators=None, fraction=None, replicates=1,
seed=None)` accepts either of two ways to specify the fraction:

- **By generators.** Pass `generators=["E=ABCD"]`. The fraction is inferred as
  $p$ equal to the number of generators.
- **By fraction.** Pass `fraction="1/2"` (or `0.5`, or the integer $p$). mfgQC then
  selects a minimum-aberration generator set from a curated table (`_MIN_ABERRATION`
  in `design.py`, after Box-Hunter-Hunter and Montgomery's design tables) and reports
  it. The table covers $(k, p)$ for $k$ up to 7. If a combination is not in the
  table, mfgQC raises and asks you to pass generators explicitly rather than guess.

The selection is surfaced, never silent. If you pass both a `fraction` and
`generators` that disagree on $p$, mfgQC raises rather than picking one.

```python
d = design.fractional_factorial(["A", "B", "C", "D", "E"], fraction="1/2")
print(d.report())
```

```text
Design (fractional 2-level): A, B, C, D, E
==========================================
factors = 5   base runs = 16   replicates = 1   center points = 0   total runs = 16

generators: E=ABCD
defining relation: I = ABCDE
resolution: V

alias structure:
  A = BCDE
  B = ACDE
  C = ABDE
  D = ABCE
  E = ABCD
  AB = CDE
  AC = BDE
  AD = BCE
  AE = BCD
  BC = ADE
  BD = ACE
  BE = ACD
  CD = ABE
  CE = ABD
  DE = ABC
```

The matrix is built by taking the base full factorial in the first $k-p$ factors and
forming each added factor's column as the product of the columns named in its
generator (`product_column` in `generate.py`). The defining relation, resolution, and
alias list come from the word algebra in `alias.py`, described in Section 3.

!!! note "What mfgQC builds and what it does not"
    These are *regular* two-level designs: every effect is either clear or perfectly
    aliased with another, never partially correlated. mfgQC does not generate
    Plackett-Burman, response-surface (central composite, Box-Behnken), or mixture
    designs; the source notes pyDOE3 is installed as the intended target for those
    higher-level designs in a later version. The over-parameterization and curvature
    messages point you toward a response-surface design when a two-level fit is no
    longer enough.

### 1.4 Blocking

There is no blocking factor in the current `Design`. If your runs split across
batches, days, or machines, that structure is not represented in the generated
design. (Center points and replication are represented; blocking is not.)

## 2. Analyzing the responses

Once the runs are done and a response column is in the frame, `qc.doe(...)` estimates
the effects. The analysis sits on top of mfgQC's regression engine: it codes each
factor to $-1/+1$ from its two observed levels, builds the model matrix of main
effects and interactions up to the chosen order, fits by least squares, and reports
each effect as twice its coefficient.

```python
res = qc.doe(design=d)            # factor names and alias structure come from the Design
res = qc.doe(factors=["A", "B", "C"], order=2)   # or name columns in an external matrix
```

Pass exactly one of `design=` or `factors=`. The response is the loaded measure.

### 2.1 Effects and coefficients

For a coded model the fitted coefficient $\hat\beta_j$ is the change in the response
per one coded unit. Because a factor spans two coded units (from $-1$ to $+1$), the
**effect** is

$$
\text{effect}_j = 2\,\hat\beta_j .
$$

mfgQC reports both: `effect` (the $2\hat\beta$ form practitioners read) and `coef`
(the half-effect, the regression coefficient). The intercept is the grand mean of the
response.

Each factor column must hold exactly two levels, optionally plus a center point at
their exact midpoint. A column with three or more genuine levels, or an off-midpoint
third value, is **refused**, not silently coded, because min/max coding would
distort the legitimate levels. Center-point rows (all factors at coded $0$) are split
off before the effect estimates; they feed curvature and pure error only.

### 2.2 When the design is replicated: t and F

If the model leaves residual degrees of freedom (because the design is replicated,
has center points, or has more runs than model terms), there is a pure-error estimate
of the noise. mfgQC then reports the regression engine's standard error, $t$
statistic, and $p$ value for each term, and flags as **significant** every term with
$p < 0.05$. This is the analysis-of-variance reading of a designed experiment: each
effect is judged against an honest estimate of run-to-run error.

A replicated $2^3$ with two replicates (16 runs, 7 model terms, 8 residual df):

```python
d = design.full_factorial(["A", "B", "C"]).replicate(2)
# ... attach the response column `yield_pct` to the run sheet ...
res = mfgqc.load(df, measure="yield_pct").doe(design=d)
print(res.report())
```

```text
DOE (full 2-level): yield_pct ~ A*B*C
=====================================
n = 16   model terms = 7   df(resid) = 8
R^2 = 0.9928

term            effect   half effect     std err         t         p
A               11.459        5.7294      0.2005      28.6  2.44e-09 *
B             -0.26125      -0.13062      0.2005    -0.651     0.533
C              -5.4563       -2.7281      0.2005     -13.6   8.2e-07 *
A:B             3.9888        1.9944      0.2005      9.95  8.84e-06 *
A:C            0.42875       0.21438      0.2005      1.07     0.316
B:C            0.05375      0.026875      0.2005     0.134     0.897
A:B:C          0.34375       0.17187      0.2005     0.857     0.416

significant (p < 0.05): A, C, A:B

Adequacy flags:
  [PASS] aliasing
  [WARN] center_points
         No center points: curvature cannot be checked - add center runs to test for a quadratic response (the trigger to consider a response-surface design).
  [PASS] pure_error
  [PASS] orthogonality

Assumption checks:
  [PASS] normality (Anderson-Darling): AD=0.303, p=0.533; skew 2.32e-14; n=16 [low power]
  [PASS] constant_variance (Breusch-Pagan): Breusch-Pagan=0.119, p=0.731; n=16
  [PASS] dispersion_effect (Brown-Forsythe (across factor levels)): variance ratio 1.49, p=0.662; n=16
  [PASS] independence (Durbin-Watson): Durbin-Watson=1.93; n=16 [low power]
```

The residual diagnostics on this path are DOE-specific (`_doe_diagnostics`):
normality of the residuals (Anderson-Darling), a constant-variance test
(Breusch-Pagan, regressing squared residuals on the fitted values), a separate
dispersion-effect test (Brown-Forsythe across each factor's high/low groups,
Bonferroni-corrected), and run-order independence (Durbin-Watson). Each
recommendation is phrased for a designed experiment (transform the response, add a
term, check randomization), not as capability or hypothesis-test boilerplate. Below
about eight residual degrees of freedom these checks are marked low power, because
they are not dependable on so few residuals.

### 2.3 When the design is unreplicated: Lenth's method

A single replicate of a $2^k$ design has exactly as many runs as model terms. The fit
is saturated: zero residual degrees of freedom, no pure-error estimate, so no $t$ or
$F$ test is possible. mfgQC does not fabricate an error term. Instead it uses
**Lenth's method** (Lenth, 1989), which estimates the noise from the effects
themselves under the assumption that most effects are inactive (effect sparsity).

The procedure, pinned to `lenth` in `significance.py`, operates on the $m$ effects
(intercept excluded):

1. **Initial scale.** $s_0 = 1.5 \cdot \operatorname{median}_j |\text{effect}_j|$.
2. **Trim and re-estimate.** Drop any effect with $|\text{effect}_j| \ge 2.5\,s_0$
   (the presumed-active ones), then the **pseudo standard error** is
   $$
   \text{PSE} = 1.5 \cdot \operatorname{median}\{\,|\text{effect}_j| : |\text{effect}_j| < 2.5\,s_0\,\}.
   $$
   The factor $1.5$ makes the median absolute effect a consistent estimate of the
   standard deviation when the effect is truly zero.
3. **Reference distribution.** The PSE is treated as a $t$ variate with
   $d = m/3$ pseudo degrees of freedom.
4. **Two thresholds.**
   $$
   \text{ME} = t_{0.975,\,d}\cdot\text{PSE}, \qquad
   \text{SME} = t_{\gamma,\,d}\cdot\text{PSE}, \quad
   \gamma = \tfrac{1}{2}\!\left(1 + 0.975^{1/m}\right).
   $$
   ME is the individual margin of error (the per-effect threshold). SME is the
   simultaneous margin of error, widened by the Bonferroni-style $\gamma$ so it
   controls the family-wise error rate across all $m$ effects.

The verdict is keyed to SME. An effect is **active** when $|\text{effect}| >
\text{SME}$, **possibly active** when $\text{ME} < |\text{effect}| \le \text{SME}$,
and inactive otherwise. The per-effect pseudo-$t$ is $\text{effect}_j / \text{PSE}$.
mfgQC treats the SME-active set as the headline finding and surfaces the
possibly-active middle band rather than dropping or promoting it silently. The
numeric ME and SME are cross-checked against `BsMD::LenthPlot` in R as a secondary
correctness check; they are labeled Tier 2 in the report.

An unreplicated $2^4$ (16 runs, 15 effects, so $d = 5$):

```python
d = design.full_factorial(["A", "B", "C", "D"])
# ... attach response `rate` ...
res = mfgqc.load(df, measure="rate").doe(design=d)
print(res.report())
```

```text
DOE (full 2-level): rate ~ A*B*C*D
==================================
n = 16   model terms = 15   df(resid) = 0
no pure-error estimate; significance from Lenth PSE / half-normal

term            effect   half effect   |t_Lenth|         verdict
A               21.016        10.508    1.02e+03          active
B              0.03375      0.016875        1.64        inactive
C               8.9962        4.4981         436          active
D              -9.0638       -4.5319         439          active
A:B           -0.01375     -0.006875       0.667        inactive
A:C             8.0238        4.0119         389          active
A:D            0.01875      0.009375       0.909        inactive
B:C            0.00625      0.003125       0.303        inactive
B:D           -0.00875     -0.004375       0.424        inactive
C:D           -0.02125     -0.010625        1.03        inactive
A:B:C         -0.02625     -0.013125        1.27        inactive
A:B:D          0.00375      0.001875       0.182        inactive
A:C:D          0.00625      0.003125       0.303        inactive
B:C:D         -0.01625     -0.008125       0.788        inactive
A:B:C:D       -0.00875     -0.004375       0.424        inactive

Lenth PSE = 0.02062   ME = 0.05302   SME = 0.1263   (ME/SME are Tier-2 secondary)
active (|effect| > SME): A, C, D, A:C
```

Here the four large effects (A, C, D, and the A:C interaction) clear the SME and
everything else sits in the noise, which is exactly the effect-sparsity pattern
Lenth's method is built to read.

### 2.4 Model order and over-parameterization

`order` caps the interaction order in the model. The default is `"full"` (every
interaction) for a full factorial passed as a `Design`, `2` for a fractional
`Design`, and otherwise the largest order whose model still fits in the available
runs. Pass an integer to fix it, or `"full"`.

A model is estimable only when its parameter count does not exceed the number of
**distinct** design points. mfgQC checks distinct runs rather than total rows, so a
replicated design cannot slip a rank-deficient model past the count. If the model is
too large, it raises with a DOE-aware message that names the alternatives (lower the
order, replicate, or pass a `Design` so the alias-aware reduced model can be fit)
rather than returning a silent minimum-norm fit.

For a fractional design this matters: a $2^{4-1}$ has 8 runs, so a full order-2 model
(intercept plus four main effects plus six two-factor interactions, 11 parameters) is
not estimable, because the interactions are aliased in pairs. Fit it at `order=1`, or
use a higher-resolution design.

### 2.5 Curvature from center points

When center points are present, mfgQC compares the factorial-point mean
$\bar y_f$ to the center-point mean $\bar y_c$ (`_curvature_check`):

$$
\text{SS}_{\text{curv}} = \frac{n_f\,n_c}{n_f + n_c}\,(\bar y_f - \bar y_c)^2 .
$$

With two or more center points there is a pure-error denominator from their spread,
so this is an $F$ test on 1 and $n_c - 1$ degrees of freedom; a significant result
says the response is not planar and points toward a response-surface design. A single
center point gives the contrast but cannot test it, and mfgQC says so.

### 2.6 Adequacy flags

Every analysis attaches a set of adequacy flags (`_adequacy_flags`). They warn; they
never change the analysis. The set covers: aliasing (a fractional design is flagged so
you read the alias list before attributing an effect); resolution against the model
order (whether the model's terms are clear of each other); center points present
(curvature checkable or not); pure error present ($t$/$F$ available or Lenth only);
and orthogonality of the fitted model matrix (the maximum off-diagonal correlation).

## 3. Alias structure

Aliasing is the price of a fraction. Two effects are aliased when they share the same
column in the reduced design, so their estimates are added together and cannot be
separated. The defining relation tells you exactly which effects are tied to which.

mfgQC computes this with word algebra (`alias.py`), not by reading it from a library.
A two-level effect is a **word**: a set of factor letters, with the grand mean as the
empty word $I$. Multiplying two words is the symmetric difference of their letters,
because a squared letter is the identity ($A \cdot A = I$):

$$
A \cdot ABCD = BCD, \qquad AB \cdot ABCD = CD .
$$

From the generator words, `defining_group` forms the group closure (every product of
every subset of generators); for $p$ independent generators this is $2^p$ words
including $I$. `resolution` is the length of the shortest non-identity word.
`alias_classes` partitions all effects into alias classes by multiplying each effect
by every defining word; the class equal to the defining words themselves is confounded
with the grand mean and dropped. `alias_list` renders one line per class, lowest-order
member first.

For the $2^{5-1}$ above, the single defining word is $ABCDE$ (resolution V), so each
main effect is aliased with a four-factor interaction and each two-factor interaction
with a three-factor interaction. At resolution V, main effects and two-factor
interactions are all clear of each other, which is why the order-2 model is the useful
read.

When you analyze through `factors=` instead of a `Design`, mfgQC detects the regular
fraction directly from the coded matrix (`_detect_fraction`): any set of columns whose
elementwise product is constant ($\pm I$) is a defining word, and the same generators,
resolution, and alias list are recovered. The general representation, used so that
non-regular designs reuse the same machinery, is `correlation_map`: aliased columns
have correlation $\pm 1$.

In the analyzed result, `alias_of` maps each model term to the effects it is aliased
with, so you can read confounding term by term:

```text
A: aliased with BCDE
B: aliased with ACDE
C: aliased with ABDE
D: aliased with ABCE
E: aliased with ABCD
A:B: aliased with CDE
```

## 4. Multi-vari analysis

A multi-vari study answers a different question than a designed experiment. Before you
choose factors to change, you want to know *where the variation already lives*. A
multi-vari chart groups the data so that three families of variation are visible at
once:

- **Within-piece (positional).** Variation across a single part: thick on one end,
  thin on the other; out of round; taper. This is the spread inside the innermost
  group.
- **Piece-to-piece (cyclical).** Variation from one part to the next within a short
  time window.
- **Time-to-time (temporal).** Variation across longer spans: shift to shift, day to
  day, batch to batch.

Seeing which family dominates tells you where to aim. A large positional family points
at the fixture or the tool; a large temporal family points at drift, setup, or
material lots.

### 4.1 What `multivari` computes

`qc.multivari(factors)` takes 2 or 3 nested factor names, outermost first, for example
`["shift", "part"]` or `["shift", "part", "position"]`. The response is the loaded
measure. The decomposition (`compute` in `multivari.py`) estimates a variance
component per level:

- The **outermost** factor's component is the sample variance (ddof = 1) of its level
  means.
- Each **inner** factor's component is the mean, over its parent groups, of the
  variance of its level means within each parent.
- The **within** component is the mean, over the innermost groups, of the spread
  inside each group (ddof = 1).

These are then mapped to the three standard families. The mapping is fixed: the
outermost factor is **temporal**, the next is **cyclical**, and the **positional**
family is the third factor when three are given, or the within-group spread when two
are given. With three factors, a non-zero within-group spread is reported as a fourth
**residual** family; a negligible one is dropped. Each component is also reported as a
percent of the total.

!!! note "This is the practical decomposition, not a full nested ANOVA"
    The component for a level is the variance of that level's means within its parent,
    averaged across parents. It is the standard multi-vari reading and matches how the
    chart is drawn, but it is not the expected-mean-squares estimator of a formal
    nested random-effects ANOVA, and the components are not bias-corrected variance
    components. Read the percents as where the variation is concentrated, not as
    certified variance-component estimates.

Two-factor study (shift over part), three replicate measurements per part:

```python
res = mfgqc.load(df, measure="x").multivari(["hour", "part"])
print(res.report())
```

```text
Multi-vari study: x across hour, part
=====================================
n = 45   total variance = 0.2247

family          source              variance  % of total
temporal        hour                  0.1351       60.1%
cyclical        part                 0.04454       19.8%
positional      within               0.04507       20.1%

largest family: hour (temporal, 60% of total) - focus improvement there.
```

Three-factor study (shift, then part within shift, then position within part):

```python
res = mfgqc.load(df, measure="dia").multivari(["shift", "part", "position"])
print(res.report())
```

```text
Multi-vari study: dia across shift, part, position
==================================================
n = 36   total variance = 0.2306

family          source              variance  % of total
temporal        shift                0.06182       26.8%
cyclical        part                 0.02105        9.1%
positional      position              0.1477       64.1%

largest family: position (positional, 64% of total) - focus improvement there.
```

In the second study the positional family dominates: most of the variation is across
position within a part, which points at the within-part geometry rather than at the
shift or the part. `res.summary()` returns the same content as a flat dict (`response`,
`n`, `total_variance`, and `var[...]`/`pct[...]` per source) for programmatic use.

The multi-vari chart from `res.view()` draws, for each outermost group, a vertical
line at each innermost group spanning that group's range, with the group means
connected, so the three families are visible as the spread of the points, the spread
within a block, and the slope across blocks.

## 5. Assumptions, and which ones mfgQC checks

| Assumption | Why it matters | What mfgQC does |
|---|---|---|
| Factors are clean two-level (optionally a midpoint center) | Coding to $-1/+1$ is only valid for two corner levels | Refuses to code a 3+-level or off-midpoint factor; raises and tells you to screen the data |
| Effect sparsity (unreplicated designs) | Lenth's PSE assumes most effects are noise | Applies Lenth's method on the saturated path and reports PSE, ME, SME |
| Adequate resolution for the model order (fractional) | Aliased terms in the model are sums of effects | Flags resolution against the fit order; refuses an inestimable over-parameterized model |
| Normal, constant-variance, independent residuals (replicated) | $t$ and $F$ tests and their $p$ values rest on these | Runs Anderson-Darling, Breusch-Pagan, Brown-Forsythe, and Durbin-Watson; marks them low power below ~8 residual df |
| Planar response (no curvature) | A two-level model fits a plane | Tests curvature from center points when present; cannot test it without them |
| Randomized run order | Guards against a time trend being read as a factor | Available via `randomize(seed)`; not enforced |
| Nested factor structure (multi-vari) | The decomposition assumes the named factors nest, outermost first | Assumed from the order you pass; not verified |

Everything in this table is reported, not acted on. The one opt-in is explicit model
reduction through `order=`.

## 6. Source standards

- **Lawson, J.** *Design and Analysis of Experiments with R.* CRC Press. mfgQC's DOE
  source for design generation, effect analysis, and alias structure. See the
  [Bibliography](bibliography.md).
- **Montgomery, D. C.** *Introduction to Statistical Quality Control.* Wiley. Source
  for the two-level factorial fundamentals and the fractional-design tables.
- **Lenth, R. V.** (1989). "Quick and Easy Analysis of Unreplicated Factorials."
  *Technometrics,* 31(4), 469-473. The pseudo-standard-error method of Section 2.3.
  This paper is the primary source for the unreplicated-analysis method and is not
  yet listed in `bibliography.md`.
- The Lenth ME/SME values are cross-checked against `BsMD::LenthPlot` in R as a
  secondary correctness check.

Box, Hunter, and Hunter (*Statistics for Experimenters*) inform the curated
minimum-aberration generator table; that text is not in `bibliography.md` either.

## See also

- [Capability](capability.md) and [Non-normal capability](non-normal-capability.md)
  for what to do once a designed experiment has centered and tightened a process.
- [Gage R&R](gage-rr.md) for the variance-component decomposition of a measurement
  system, a close cousin of the multi-vari decomposition.
- [Provenance model](provenance.md) for the immutability and hash-chain guarantees the
  `Design` and result objects carry.
- [API reference](api.md) for the generated docstrings of `full_factorial`,
  `fractional_factorial`, `Design`, `doe`, `lenth`, and `multivari`.
- [Reference index](index.md) for the other method-family pages.
- [Bibliography](bibliography.md) for full citations.
