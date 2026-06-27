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
# # The audit workflow
#
# This is the provenance and audit workflow end to end. You run an analysis,
# record its provenance digest next to the number you report, export the full
# result as JSON, then verify and trace that number back to the raw data later.
# Every cell below runs, so the digests, the lineage, and the True/False from
# verification are printed as real output, not pasted in.
#
# <div class="nb-buttons"><a class="nb-btn" target="_blank" href="https://colab.research.google.com/github/cjbrant/mfgQC/blob/main/docs/guide/audit-workflow.ipynb">Run in Colab</a><a class="nb-btn" target="_blank" href="https://github.com/cjbrant/mfgQC/blob/main/docs/guide/audit-workflow.ipynb">View on GitHub</a><a class="nb-btn" target="_blank" href="https://raw.githubusercontent.com/cjbrant/mfgQC/main/docs/guide/audit-workflow.ipynb" download>Download notebook</a></div>

# %% [markdown]
# Install it first (skip this if mfgQC is already in your environment):

# %% tags=["skip-execution"]
!pip install mfgqc

# %% [markdown]
# ## The example
#
# We use a small, strictly-positive dataset and apply a Box-Cox transform, so the
# lineage has something interesting in it. The seed is fixed, so the digests this
# notebook prints are reproducible run to run.

# %%
import json
import dataclasses as dc
import numpy as np, pandas as pd, mfgqc

rng = np.random.default_rng(11)
df = pd.DataFrame({
    "cycles": np.round(rng.lognormal(mean=1.2, sigma=0.35, size=80), 3),
})

qc  = mfgqc.load(df, measure="cycles").spec(lower=0.5, upper=12.0)
cap = qc.transform("boxcox").capability()
cap

# %% [markdown]
# ## 1. Run the analysis and read its lineage
#
# Every result carries the full chain of operations that produced it.
# `lineage()` returns one dict per step. Pull the operation names to see the shape
# of the computation:

# %%
[s["operation"] for s in cap.lineage()]

# %% [markdown]
# That is the whole derivation: the frame was loaded, spec limits were attached,
# the measure was Box-Cox transformed, capability was computed, and a normality
# assumption check ran. Nothing happened that is not on this list.

# %% [markdown]
# ## 2. Record the digest when you report the number
#
# When you write the reported value down (into a report, a LIMS, a Certificate of
# Analysis), capture the provenance digest next to it:

# %%
digest = cap.provenance_digest()
print(digest)

# %% [markdown]
# That SHA-256 string pins the *computation* that produced the number: the
# operations, their parameters (including the fitted Box-Cox lambda), and how many
# rows each step touched. The timestamp is deliberately not in the digest, so it
# is reproducible run to run.
#
# Store the digest as a sibling field of the reported value, not instead of it.
# The digest is a fingerprint, not the data. Keeping it next to the reported Cpk
# gives anyone re-deriving the number later something to check against.

# %% [markdown]
# ## 3. Export the full result as JSON
#
# `to_dict()` is the canonical payload. It carries the fields, the flat summary,
# the assumption checks, and the lineage plus the digest: everything a downstream
# report builder needs, with no `report()` text to parse.

# %%
d = cap.to_dict()
list(d.keys())

# %% [markdown]
# The two provenance keys are `history` (the lineage, each step carrying its
# running digest) and `provenance_digest` (the head digest from step 2). They are
# the same digest you recorded above, stamped into the export by construction:

# %%
print("provenance_digest:", d["provenance_digest"])
print("matches step 2:   ", d["provenance_digest"] == digest)
print("history step keys:", list(d["history"][0].keys()))

# %% [markdown]
# The transform step in `history` shows that the fitted lambda and its confidence
# interval are recorded in the provenance, not buried in a log:

# %%
transform_step = next(s for s in d["history"] if s["operation"] == "transform")
print(json.dumps(transform_step, indent=2))

# %% [markdown]
# The assumption checks ride along too. Here is the normality check that justifies
# the normal-method capability:

# %%
print(json.dumps(d["assumptions"][0], indent=2))

# %% [markdown]
# Write it to a file and you have a self-describing, archivable record. The
# `provenance_digest` stamped into the file equals the digest you reported, so the
# export and the reported number agree by construction.

# %%
import pathlib
payload = json.dumps(cap.to_dict(), indent=2)
pathlib.Path("result.json").write_text(payload)
print("wrote result.json,", len(payload), "bytes")

# %% [markdown]
# Frontends and report builders should consume `to_dict()` (or the flat
# `summary()`), never parse `report()` text. The JSON is the stable contract; the
# text report is for humans.

# %% [markdown]
# ## 4. Verify later
#
# Months later, someone reopens the archived result (or recomputes it from the
# same inputs) and checks it against the digest you recorded:

# %%
cap.verify_provenance(digest)

# %% [markdown]
# `verify_provenance(expected)` recomputes the digest over the current history and
# compares it to the one you pass in. True means the recorded computation is
# intact.

# %% [markdown]
# ### Tamper-evidence, demonstrated honestly
#
# The chain is tamper-evident: changing the `operation`, `params`, or `n_affected`
# of any recorded step changes the head digest, so verification fails.
#
# The result and its history are frozen, so there is no in-place edit to make. To
# show this we construct an *altered copy* with `dataclasses.replace`. We are not
# mutating the original `cap`; we build a new object whose recorded transform step
# has its fitted lambda bumped by 1.0, then verify that copy against the original
# digest.

# %%
hist = list(cap.history)
for i, s in enumerate(hist):
    if s.operation == "transform":
        bad = dict(s.params)
        bad["lambda"] = bad["lambda"] + 1.0          # alter a recorded parameter
        hist[i] = dc.replace(s, params=bad)

tampered = dc.replace(cap, history=tuple(hist))      # a new, altered copy

print("original digest: ", digest)
print("tampered digest: ", tampered.provenance_digest())
print("verify tampered: ", tampered.verify_provenance(digest))
print("original intact: ", cap.verify_provenance(digest))

# %% [markdown]
# One altered parameter, in one step, three steps deep, and the head digest moves
# and verification returns `False`. The original `cap` is untouched and still
# verifies `True`: we built a new object rather than editing it, because the
# history is append-only by construction.

# %% [markdown]
# ## 5. Trace a number back to raw data
#
# `lineage()` is the audit trail. Each step gives you its `operation`, its
# `params`, its `n_affected`, and the running digest folded in up to and including
# that step:

# %%
for s in cap.lineage():
    print(s["operation"], "| n_affected:", s["n_affected"], "| digest:", s["digest"][:16], "...")

# %% [markdown]
# Read it bottom-up to walk the reported number back to the raw frame. Each step's
# `params` records exactly what it did:

# %%
for s in cap.lineage():
    print(s["operation"])
    print("   ", s["params"])

# %% [markdown]
# So the reported capability was computed after a Box-Cox transform with the fitted
# lambda above, against spec limits [0.5, 12.0], on 80 loaded rows, and the
# normality check that justifies the normal-method capability is right there in the
# chain. No step is hidden, and each step's `digest` lets you confirm where in the
# chain a difference first appears.
#
# The running digest also lets you cross-check intermediate state. The `QCData`
# after the transform exposes the same provenance surface, and its digest equals
# the transform step's running digest in the result's lineage:

# %%
qct = qc.transform("boxcox")
transform_running = next(s["digest"] for s in cap.lineage() if s["operation"] == "transform")

print("QCData-after-transform digest:", qct.provenance_digest())
print("transform step running digest:", transform_running)
print("equal:                        ", qct.provenance_digest() == transform_running)

# %% [markdown]
# `lineage()`, `provenance_digest()`, and `verify_provenance()` exist on both
# `QCData` and every result object. The trail is continuous from the loaded frame
# through to the final number.

# %% [markdown]
# ## What passing and failing verify actually mean
#
# A passing `verify_provenance()` means the recorded result is intact: the
# archived analysis has not been edited since the digest was captured. A failing
# one means the history no longer matches, so something in the recorded chain
# changed.
#
# What it does not do, on its own: it does not stop an actor who controls the
# Python interpreter at runtime from recomputing the whole analysis over
# fabricated inputs and stamping a fresh, self-consistent digest. The digest is a
# content hash, not a cryptographic signature. It defends against accidental
# corruption and post-hoc tampering with a stored result, not against an adversary
# who controls the process that produces it.
#
# Closing that gap requires anchoring the head digest outside the process: signing
# it with a key the operator does not hold, or writing it to an append-only
# external log. That is out of scope for the core library and left to the
# deployment. The full scope statement is in
# [Provenance model](/reference/provenance/).

# %% [markdown]
# ## Next
#
# - [Provenance model](/reference/provenance/): the data model, the hash-chain
#   algorithm, and the honest scope of the guarantee.
# - [Reference](/reference/): the formula, assumptions, and source standard behind
#   every method, plus the full result surface.
