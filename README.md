# mfgQC

[![PyPI version](https://img.shields.io/pypi/v/mfgqc?cacheSeconds=3600)](https://pypi.org/project/mfgqc/)
[![Python versions](https://img.shields.io/pypi/pyversions/mfgqc?cacheSeconds=3600)](https://pypi.org/project/mfgqc/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![tests](https://github.com/cjbrant/mfgQC/actions/workflows/tests.yml/badge.svg)](https://github.com/cjbrant/mfgQC/actions/workflows/tests.yml)
[![docs](https://img.shields.io/badge/docs-mfgqc.brantnersolutions.com-0a7d5a.svg)](https://mfgqc.brantnersolutions.com)

**Auditable SPC, capability, and gage R&R for manufacturing**, by
[Brantner Solutions](https://brantnersolutions.com).
Full documentation: **[mfgqc.brantnersolutions.com](https://mfgqc.brantnersolutions.com)**.

Quality-control analysis for manufacturing practitioners, not statisticians or
programmers. Three pillars:

1. **Statistical guardrails.** Every analysis checks its own assumptions and
   reports the outcome. It warns and recommends; it never silently switches
   methods. Auto-correction is opt-in.
2. **Practitioner-oriented.** You bring domain knowledge; mfgQC brings the
   statistics, data handling, and canonical charts. Errors say what's missing
   and why.
3. **Auditable by construction.** `QCData` and result objects are immutable and
   carry a structured, propagating provenance history, so the full lineage from
   raw data to final number can be reconstructed.

## Install

```bash
pip install mfgqc
```

For development (editable install with test extras):

```bash
pip install -e ".[test]"
```

Requires Python 3.10+. Core dependencies are NumPy, pandas, SciPy, Matplotlib,
statsmodels, and scikit-learn.

## Use

The idiom is always the same: `load` a frame, attach metadata with `.spec()` /
`.roles()`, then call an analysis. Every result has `.report()` (text),
`.summary()` (a flat dict), `.to_dict()` (full structured payload), and `.view()`
(the canonical chart).

```python
import pandas as pd, mfgqc

qc = (mfgqc.load(df, measure="width", subgroup="lot", subgroup_size=5)
           .spec(lower=1.0, upper=2.0, target=1.5))

print(qc.capability())        # Cp/Cpk (within-sigma) + Pp/Ppk (overall) + assumption report
print(qc.control_chart())     # inferred chart, run-rules violations
print(qc.gage_rr())           # ANOVA gage R&R (needs part/operator/replicate roles)

fig = qc.capability().view(save="capability.png")   # canonical chart (matplotlib Figure)
```

To load a CSV, read it with pandas first: `mfgqc.load(pd.read_csv(path), measure=...)`.

## Provenance & auditability

The lineage from raw data to the final number is recorded, immutable, and
verifiable, not asserted in a doc.

**What is recorded.** Every step that derives a number appends a structured
provenance entry: ingest, spec/role binding, each transform (e.g. Box-Cox),
subgroup aggregation, the sigma method chosen, and each assumption check. The
chain is reconstructable end to end:

```python
qc  = mfgqc.load(df, measure="y").spec(lower=0.1, upper=8)
cap = qc.transform("boxcox").capability()

[s["operation"] for s in cap.lineage()]
# ['load', 'spec', 'transform', 'capability', 'assumption:normality']
```

**Immutability guarantee (append-only by construction).** `QCData` and every
result are frozen dataclasses; the history is an immutable tuple of frozen steps,
so it cannot be reordered, inserted into, or edited in place. The ingest boundary
defensive-copies the input frame, `.frame` hands back a copy, and `.values()` is
read-only, so nothing a caller does to what mfgQC returns can reach back and
change a recorded result.

**Tamper-evidence (hash-chained and verifiable).** Each step folds into a
running SHA-256, exposed as `provenance_digest()` and stamped into `to_dict()`.
Capture the digest when you record a number, and re-check it later:

```python
digest = cap.provenance_digest()      # store alongside the reported Cpk
cap.verify_provenance(digest)         # True; flips to False if any step was edited
```

**The limit, stated honestly.** The digest is a content hash, not a cryptographic
signature: code running in the same process could edit a step *and* recompute the
digest. It defends against accidental corruption and post-hoc edits to a stored
result, not against an adversary who controls the interpreter. One boundary is
explicit: once you extract the matplotlib `Figure` from `.view()`, edits to that
Figure are outside the lineage.

## Modules

- **Capability**: Cp/Cpk (within-subgroup sigma) vs Pp/Ppk (overall sigma), Cpm,
  each with confidence intervals (small-n point estimates are overconfident; the
  interval is reported so you see it); normal plus Box-Cox / Clements / Johnson
  non-normal methods.
- **Control charts**: I-MR, Xbar-R, Xbar-S, p/np/c/u, EWMA, CUSUM, short-run;
  Nelson and Western Electric run rules.
- **Measurement systems analysis**: ANOVA gage R&R, bias, linearity, stability,
  attribute agreement (Cohen/Fleiss/weighted kappa).
- **Hypothesis testing**: assumption-routing two-sample / one-sample / variance /
  proportion tests, one-way ANOVA, post-hoc (Tukey/Games-Howell/Dunn/Dunnett),
  non-parametrics (Mood's median, repeated measures).
- **Regression & DOE**: OLS, model selection, logistic, non-linear least squares,
  Box-Cox; full and fractional factorial design generation and effect analysis
  (Lenth for unreplicated designs) with alias structure.
- **Sample size & power**: t-test, ANOVA, proportion, and variance, via the
  noncentral distributions.
- **Attributes & reliability**: DPMO/sigma level, life-distribution fitting with
  censoring, Kaplan-Meier, system reliability, bearing life (ISO 281), MTBF,
  availability, demonstration tests.
- **Acceptance sampling**: single attribute plans, OC curves, ANSI/ASQ Z1.4 and
  Z1.9.

## Method choices

The choices a practitioner needs to trust the number are explicit, not hidden:

- **Control-chart inference.** `control_chart()` with no `kind=` picks the
  variables chart from the subgroup size: size 1 → I-MR (individuals + moving
  range), size 2–10 → X-bar R, size > 10 → X-bar S. Pass `kind=` to override; attribute
  charts (`p`/`np`/`c`/`u`) are always explicit.
- **Within-subgroup sigma (Cp/Cpk).** Equal subgroups use R-bar/d₂; individuals use
  MR-bar/d₂; unequal subgroup sizes use the pooled within-subgroup standard
  deviation. Pp/Ppk always use the overall standard deviation. The estimator
  actually used is reported in the capability result (`sigma_used`, e.g.
  `"within (R-bar/d2)"`).

## Building on mfgQC

mfgQC is built to be driven programmatically (e.g. by a report builder or UI):

- `mfgqc.list_analyses()` / `mfgqc.ANALYSES`: machine-readable catalog of every
  analysis and its required inputs.
- `result.to_dict()`: full JSON-serializable payload (fields, assumption checks,
  provenance); `result.summary()` is the flat form.
- `result.view(save="chart.png")`: headless chart rendering to PNG/SVG.
- `mfgqc.MissingPrerequisiteError`: specific, catchable error naming what an
  analysis still needs.

## Documentation

Full user guide and reference (with the formula, assumptions, and source standard
each method is pinned to) live at
**[mfgqc.brantnersolutions.com](https://mfgqc.brantnersolutions.com)**.

## Tests

The suite has two layers. Regression tests pin mfgQC to its build oracles
(Montgomery; AIAG MSA 4th ed.; Lawson, *Design and Analysis of Experiments with
R*). A separate `tests/correctness/` suite pins each analysis to an **independent**
source it was not built against (the NIST/SEMATECH e-Handbook and StRD certified
datasets, the R `qcc`/`SixSigma` packages, and scipy/statsmodels computed in-test).

```bash
pytest
```

---

© Brantner Solutions · [brantnersolutions.com](https://brantnersolutions.com) · MIT License
