# User Guide

Task-oriented guidance for using mfgQC day to day. If you want the formula and the
source standard behind a method, see the [Reference](../reference/index.md) instead.

- **[Install](install.md)**: `pip install mfgqc`, supported Python versions.
- **[Quickstart](../guide/quickstart.ipynb)**: the `load → spec → analysis` flow on a worked dataset.
- **[Reading the assumption report](../guide/assumption-report.ipynb)**: what each guardrail
  checks, what a warning means, and when to opt into auto-correction.
- **[Choosing a control chart](../guide/choosing-a-control-chart.ipynb)**: the inference rule
  and how to override it.
- **[Gage R&R workflow](../guide/gage-rr.ipynb)**: roles, variance components,
  %study vs %tolerance, ndc, and the AIAG verdict.
- **[The audit workflow](../guide/audit-workflow.ipynb)**: record, export, and verify a
  result's lineage. This page shows the provenance model end to end.

## The one idiom

Everything in mfgQC follows the same shape: a verb produces an object, and the
object has methods.

```python
import pandas as pd, mfgqc

qc  = mfgqc.load(df, measure="width", subgroup="lot", subgroup_size=5)
qc  = qc.spec(lower=1.0, upper=2.0, target=1.5)   # attach metadata fluently
cap = qc.capability()                              # run an analysis

cap.report()              # full text: numbers + assumption checks + recommendations
cap.summary()             # a flat dict of the headline scalars
cap.to_dict()             # full JSON-serializable payload (consume this from code)
cap.view(save="cap.png")  # the canonical chart
```

Once you know this shape, every analysis in the library works the same way.
