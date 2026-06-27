# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Quickstart
#
# This walks the whole flow on a small dataset: load a table, attach the spec
# limits, run an analysis, read the result. Every cell runs, so open it in Colab
# and change the numbers to your own.
#
# <div class="nb-buttons"><a class="nb-btn" target="_blank" href="https://colab.research.google.com/github/cjbrant/mfgQC/blob/main/docs/guide/quickstart.ipynb">Run in Colab</a><a class="nb-btn" target="_blank" href="https://github.com/cjbrant/mfgQC/blob/main/docs/guide/quickstart.ipynb">View on GitHub</a><a class="nb-btn" target="_blank" href="https://raw.githubusercontent.com/cjbrant/mfgQC/main/docs/guide/quickstart.ipynb" download>Download notebook</a></div>

# %% [markdown]
# Install it first (skip this if mfgQC is already in your environment):

# %% tags=["skip-execution"]
!pip install mfgqc

# %% [markdown]
# ## 1. Load a table and attach the spec
#
# mfgQC analyses take a `QCData` object. You build one by loading a tidy
# DataFrame, naming the measurement column, and attaching the engineering
# tolerances. Here is a made-up run of 100 widths in 20 subgroups of 5.

# %%
import numpy as np, pandas as pd, mfgqc

rng = np.random.default_rng(7)
df = pd.DataFrame({
    "width": np.round(rng.normal(1.50, 0.11, 100), 3),
    "lot":   np.repeat(np.arange(1, 21), 5),
})

qc = (mfgqc.load(df, measure="width", subgroup="lot", subgroup_size=5)
           .spec(lower=1.0, upper=2.0, target=1.5))
qc

# %% [markdown]
# `measure` is the value column, `subgroup` groups the rows into rational
# subgroups, and `.spec(...)` attaches the limits. Everything is immutable, so
# each call hands back a new object.

# %% [markdown]
# ## 2. Capability
#
# Run it and print the report.

# %%
print(qc.capability())

# %% [markdown]
# Three things to notice. Both sigma families are reported: within-subgroup
# (Cp, Cpk) and overall (Pp, Ppk), with the estimator named. Confidence
# intervals are shown because small-sample point estimates run optimistic. And
# the assumptions are checked and printed. Here normality passes but the
# subgroup count is flagged as low. mfgQC tells you and recommends a fix. It does
# not quietly change the math.
#
# Every result also draws its own chart.

# %%
qc.capability().view()

# %% [markdown]
# ## 3. Control chart
#
# With no `kind`, mfgQC picks the chart from the subgroup size. Five per
# subgroup gives an X-bar and R chart.

# %%
print(qc.control_chart())

# %%
qc.control_chart().view()

# %% [markdown]
# ## 4. One shape for everything
#
# Every analysis returns the same surface, so once you know one you know them all.

# %%
cap = qc.capability()

cap.summary()        # a flat dict of the headline numbers

# %% [markdown]
# - `cap.report()` is the text above.
# - `cap.summary()` is the flat dict you just saw.
# - `cap.to_dict()` is the full JSON payload (numbers, assumption checks, provenance). Read this from code, never the text.
# - `cap.view()` is the chart.
#
# ## Next
#
# - [Reading the assumption report](/guide/assumption-report/) explains the guardrails.
# - [Gage R&R study](/guide/gage-rr/) walks a measurement-system study end to end.
# - The [Reference](/reference/) gives the formula, assumptions, and source standard behind every method.
