# Provenance model

When you report a Cpk, someone may later ask how you arrived at it: which data,
which transform, which sigma estimator. Provenance is mfgQC's answer. Every result
carries a record of how it was computed, and that record can be checked later. This
page describes how the record is built, what keeps it from being edited, and exactly
what checking it does and does not prove.

## What gets recorded

Every analysis takes a `QCData` and returns a frozen result. Both carry a
**history**: an immutable `tuple[Step, ...]`. A `Step` is one operation that derived
or transformed data.

| Field | Meaning |
| --- | --- |
| `operation` | what happened: `load`, `spec`, `transform`, `capability`, `assumption:normality`, and so on |
| `params` | the parameters of that operation, such as the Box-Cox λ or the spec limits |
| `n_affected` | how many rows the step touched |
| `timestamp` | wall-clock time the step ran |

You can read the chain end to end:

```python
qc  = mfgqc.load(df, measure="y").spec(lower=0.1, upper=8)
cap = qc.transform("boxcox").capability()

[s["operation"] for s in cap.lineage()]
# ['load', 'spec', 'transform', 'capability', 'assumption:normality']
```

`lineage()` returns each step as a dict, with the running digest folded in up to and
including that step.

## Why the record cannot be edited

`QCData` and every result are **frozen dataclasses**, and the history is an immutable
tuple of frozen `Step`s. So the history cannot be reordered, inserted into, or
changed in place. Three boundaries close the obvious ways around that:

- **Ingest copies the input frame.** Changing the original DataFrame later cannot
  reach a recorded result.
- **`.frame` returns a copy.** A caller cannot mutate the stored frame through it.
- **`.values()` is read-only.**

Every transform returns a *new* `QCData`. It appends a step to a copy of the history.
Nothing changes in place.

## How the record is checked

Each step folds into a running SHA-256. The timestamp is left out of the hashed
content on purpose, so the digest is the same from one run to the next. It pins the
computation, not the wall clock.

The part of a step the digest covers (the canonical step) is exactly:

```python
{
    "operation": step.operation,
    "params":    step.params,      # JSON-normalized
    "n_affected": step.n_affected,
}
```

The chain folds each canonical step into the previous digest:

$$
d_0 = \text{""}, \qquad
d_i = \mathrm{SHA256}\big(d_{i-1} \,\Vert\, \mathrm{json}(\text{canon}_i)\big)
$$

where the JSON uses sorted keys and compact separators. The final digest
$d_{\text{final}}$ is what `provenance_digest()` returns and what `to_dict()` stores.
Change the `operation`, `params`, or `n_affected` of any recorded step and
$d_{\text{final}}$ changes.

```python
digest = cap.provenance_digest()   # store this alongside the reported Cpk
cap.verify_provenance(digest)      # True now; False if a step is edited later
```

`verify_provenance(expected)` recomputes the digest over the current history and
compares it to `expected`.

## What checking proves, and what it does not

!!! warning "Scope"
    The digest is a content hash, not a cryptographic signature. Code running in the
    same process can edit a step and recompute the digest. So the digest catches
    accidental corruption and after-the-fact edits to a stored result. It does not
    stop someone who controls the interpreter.

    Stated plainly: `verify_provenance()` proves the integrity of a *recorded* result.
    It detects tampering with an archived analysis. On its own it does not stop someone
    from recomputing the whole analysis over fabricated inputs. Closing that gap needs
    an external anchor: sign $d_{\text{final}}$ with a key the operator does not hold,
    or write it to an append-only log elsewhere. That part is left to the deployment
    and is out of scope for the core library.

    One more boundary. Once you pull the matplotlib `Figure` out of `.view()`, edits to
    that figure are outside the lineage.

## Using it in an audit

The full workflow, capturing a digest, exporting a result with its lineage, and
tracing a reported number back to the raw data, is walked through with a worked
example in [The audit workflow](../guide/audit-workflow.ipynb).

*Source:* the algorithm is implemented in `mfgqc/_result.py` (`history_digest`,
`history_lineage`, `_canonical_step`, `_chain`) and exposed on both `QCData` and every
result object.
