# Capability

Normal-theory process capability answers one question: **given the spec limits, how
much of the tolerance does the process consume?** mfgQC reports the full index family
($C_p$, $C_{pk}$ with $C_{pu}$/$C_{pl}$, $P_p$, $P_{pk}$, and $C_{pm}$), names the
sigma estimator it used, attaches confidence intervals because small-sample point
estimates are overconfident, and checks the assumptions that make the numbers
meaningful in the first place.

This page pins every formula and estimator to `mfgqc/capability.py`. The default
`method="normal"` never transforms your data; for skewed processes see
[Non-normal capability](non-normal-capability.md).

```python
qc  = mfgqc.load(df, measure="width", subgroup="lot", subgroup_size=5).spec(lower=1.0, upper=2.0, target=1.5)
cap = qc.capability()
```

## 1. The formulas mfgQC computes

Let $\mu$ be the sample mean, $\hat\sigma_{\text{within}}$ the short-term
(within-subgroup) sigma, $\hat\sigma_{\text{overall}}$ the long-term (ordinary sample)
sigma, and $T$ the target. With lower/upper spec limits $\text{LSL}$/$\text{USL}$:

**Within-subgroup indices** (`_indices`, computed at $\hat\sigma_{\text{within}}$):

$$
C_p = \frac{\text{USL} - \text{LSL}}{6\,\hat\sigma_{\text{within}}}
$$

$$
C_{pu} = \frac{\text{USL} - \mu}{3\,\hat\sigma_{\text{within}}}, \qquad
C_{pl} = \frac{\mu - \text{LSL}}{3\,\hat\sigma_{\text{within}}}, \qquad
C_{pk} = \min\!\big(C_{pu},\, C_{pl}\big)
$$

**Overall (performance) indices**, the same formulas evaluated at
$\hat\sigma_{\text{overall}}$:

$$
P_p = \frac{\text{USL} - \text{LSL}}{6\,\hat\sigma_{\text{overall}}}, \qquad
P_{pk} = \min\!\left(\frac{\text{USL} - \mu}{3\,\hat\sigma_{\text{overall}}},\;
                     \frac{\mu - \text{LSL}}{3\,\hat\sigma_{\text{overall}}}\right)
$$

**Taguchi index** $C_{pm}$, a two-sided index that penalizes being off-target,
built on the **overall** sigma about the target:

$$
C_{pm} = \frac{\text{USL} - \text{LSL}}{6\,\tau}, \qquad
\tau = \sqrt{\hat\sigma_{\text{overall}}^{\,2} + (\mu - T)^2}
$$

!!! note "Which indices appear depends on the spec you set"
    $C_p$, $P_p$, and $C_{pm}$ require **both** limits. With only one limit set, mfgQC
    reports the relevant one-sided index ($C_{pu}$ *or* $C_{pl}$) and folds it straight
    into $C_{pk}$; the two-sided indices come back `n/a`. $C_{pm}$ additionally requires
    `target=` in `.spec(...)`. This mirrors `_indices` returning `None` for any index
    whose limit is absent.

## 2. The sigma estimator mfgQC selects

A **subgroup** is a small set of parts measured together under the same conditions, for
example five consecutive parts off one machine. You declare how rows group into subgroups
when you load the data; see [How subgrouping works](control-charts.md#how-subgrouping-works).
The within/overall distinction is the heart of capability and the reason mfgQC reports
**both** families instead of one number:

- **Within-subgroup ($C_p$/$C_{pk}$)** estimates the process's *inherent* short-term
  spread: the variation between consecutive parts, with shift and drift between
  subgroups filtered out. It answers *"what is this process capable of?"*
- **Overall ($P_p$/$P_{pk}$)** uses the ordinary sample standard deviation across all
  data, so it captures *everything*: short-term noise plus every shift, drift, and
  tool change over the study. It answers *"what did the customer actually receive?"*

When $C_{pk} \gg P_{pk}$, your process is stable in the short run but wandering over
time, a hunt-for-special-causes signal, not a capable process.

`_within_sigma` picks the within estimator from the subgroup structure and records the
choice in `sigma_used`:

| Subgroup structure | Estimator | `sigma_used` label |
| --- | --- | --- |
| Equal subgroups, size $n \ge 2$ | $\hat\sigma_{\text{within}} = \dfrac{\bar R}{d_2(n)}$ | `within (R-bar/d2)` |
| Individuals ($n = 1$) | $\hat\sigma_{\text{within}} = \dfrac{\overline{MR}}{d_2(2)}$ | `within (MR-bar/d2)` |
| Unequal subgroup sizes | pooled within-subgroup SD (below) | `within (pooled)` |
| No usable subgroups | falls back to overall | `overall` |

where $\bar R$ is the mean subgroup range, $\overline{MR}$ the mean moving range of the
individuals series, and $d_2$ is the bias-correction constant from
`mfgqc/constants.py` (e.g. $d_2(2)=1.128$, $d_2(5)=2.326$; Montgomery, Appendix VI). The
pooled estimator is the usual degrees-of-freedom-weighted root mean square of the
per-subgroup variances:

$$
\hat\sigma_{\text{pooled}} = \sqrt{\frac{\sum_{j}\,(n_j - 1)\,s_j^2}{\sum_{j}\,(n_j - 1)}}
$$

over subgroups $j$ with $n_j \ge 2$.

!!! important "$P_p$/$P_{pk}$ always use the overall sample SD"
    Regardless of subgroup structure, the performance indices are computed at
    $\hat\sigma_{\text{overall}} = s$ (`values.std(ddof=1)`). They never use a
    within estimator. The within and overall families are reported side by side and
    **never conflated**.

The result object carries `sigma_within`, `sigma_overall`, and the string `sigma_used`
so a report builder can show exactly which estimator drove $C_{pk}$. Consume these from
[`summary()`/`to_dict()`](api.md), never by parsing the report text.

## 3. Confidence intervals

Small-$n$ capability point estimates are biased toward looking *better* than the
process is. mfgQC reports normal-theory CIs (default 95%, `alpha=0.05`) so you read a
$C_{pk}$ as a range, not a false-precision decimal. From `_capability_cis`:

**$C_p$, exact $\chi^2$ interval** (Montgomery eq. 8.19):

$$
\widehat{C_p}\sqrt{\frac{\chi^2_{\alpha/2,\,n-1}}{n-1}}
\;\le\; C_p \;\le\;
\widehat{C_p}\sqrt{\frac{\chi^2_{1-\alpha/2,\,n-1}}{n-1}}
$$

**$C_{pk}$, large-sample normal approximation** (Montgomery eq. 8.21): the half-width
factor is

$$
m = z_{1-\alpha/2}\,\sqrt{\frac{1}{9 n\,\widehat{C_{pk}}^{\,2}} + \frac{1}{2(n-1)}},
\qquad
\big(\widehat{C_{pk}}(1 - m),\;\; \widehat{C_{pk}}(1 + m)\big)
$$

!!! note "When CIs are omitted"
    CIs require $n \ge 2$, and the $C_{pk}$ interval additionally requires
    $\widehat{C_{pk}} \neq 0$. **Confidence intervals are computed for the normal method
    only.** Non-normal methods (`boxcox`, `clements`, `johnson`) return `None` and the
    report prints `CI: n/a (non-normal method)` rather than fabricating a normal-theory
    interval. See [Non-normal capability](non-normal-capability.md).

## 4. Assumptions, and which ones mfgQC checks

| Assumption | Why it matters | Checked by mfgQC? |
| --- | --- | --- |
| **Normality** | $C_p$/$C_{pk}$ map to fraction-defective only if the data are normal | **Yes**, Anderson-Darling, reported |
| **Process in statistical control** | a capability index on an unstable process is meaningless | **No**, you must establish it first with a control chart |
| **Adequate subgroup count** | $\bar R / d_2$ needs enough subgroups to be stable | **Yes**, flags fewer than 25 |

**Normality (checked, reported).** mfgQC runs an Anderson-Darling test
(`check_normality`) and reports the verdict at $\alpha = 0.05$. The binary
`passed`/`failed` comes from the *direct* AD test; alongside it mfgQC reports an
**est. Cpk impact** magnitude, the relative shift in $C_{pk}$ between the normal method
and an auto-fit non-normal method (`_cpk_shift`). That context tells you whether the
non-normality actually moves the number you care about, but it **never flips the
verdict**: grossly non-normal data fail even if the coincidental $C_{pk}$ shift is
small. If it fails, mfgQC recommends a non-normal method. It does not silently switch.
See [Reading the assumption report](../guide/assumption-report.ipynb).

!!! warning "Establish statistical control *before* capability"
    mfgQC does **not** test for statistical control inside `capability()`. A capability
    index computed on an out-of-control process is not interpretable: the within-sigma
    estimate is contaminated and the fraction-defective projection cannot be trusted. Run a
    [control chart](control-charts.md) first, confirm there are no out-of-control
    signals, and only then compute capability.

**Subgroup sufficiency (checked, reported).** The within estimator pools the spread inside
each subgroup, so it needs enough subgroups to be stable. This is the *number of subgroups*,
not the size of each one and not the total sample size: twenty subgroups of five is twenty
subgroups, even though that is a hundred measurements. When a within estimator is in use,
mfgQC counts the subgroups and **fails the check below 25**, with the recommendation *"Only
N subgroups; >=25 recommended for a stable within-sigma estimate."* It still computes the
index. It warns, it does not refuse.

## 5. Worked example

Twenty subgroups of five, spec $[1.0, 2.0]$ with target $1.5$:

```python
import numpy as np, pandas as pd, mfgqc

rng = np.random.default_rng(7)
df = pd.DataFrame({
    "width": np.round(rng.normal(1.52, 0.12, size=100), 3),
    "lot":   np.repeat(np.arange(1, 21), 5),   # 20 subgroups of 5
})

qc  = (mfgqc.load(df, measure="width", subgroup="lot", subgroup_size=5)
            .spec(lower=1.0, upper=2.0, target=1.5))
cap = qc.capability()
print(cap.report())
```

```text
Process Capability (method=normal)
==================================
n = 100   mean = 1.4992
sigma (within)  = 0.1105
sigma (overall) = 0.10556
Cp/Cpk sigma    = within (R-bar/d2)

Cp  = 1.508  95% CI (1.3, 1.72)
Cpk = 1.506  95% CI (1.29, 1.73)   (Cpu=1.51, Cpl=1.506)
Pp  = 1.579    Ppk = 1.576   (Ppu=1.581, Ppl=1.576)
Cpm = 1.579

Assumption checks:
  [PASS] normality (Anderson-Darling): AD=0.301, p=0.572; est. Cpk impact 5.0%; n=100
  [FAIL] subgroup_sufficiency (subgroup count >= 25): subgroup count 20; n=20

Recommendations:
  - Only 20 subgroups; >=25 recommended for a stable within-sigma estimate.
```

Reading it:

- **The estimator is named.** $C_p$/$C_{pk}$ used `within (R-bar/d2)`: equal subgroups
  of 5, so $\hat\sigma_{\text{within}} = \bar R / d_2(5)$ with $d_2(5)=2.326$.
- **Both families show.** $C_{pk}=1.506$ (short-term) vs $P_{pk}=1.576$ (long-term).
  They are close here because the process is stable.
- **CIs are wide.** $C_{pk}=1.506$ with a 95% CI of $(1.29,\,1.73)$. Even with 100
  measurements the index is only pinned to one decimal.
- **The guardrails fired.** Normality passes; subgroup count (20) is flagged below 25.
  mfgQC tells you and recommends a fix; it does not alter the calculation.

The flat dashboard dict is available without parsing text:

```python
cap.summary()
# {'method': 'normal', 'n': 100, 'mean': 1.4992..., 'sigma_within': 0.11051...,
#  'sigma_overall': 0.10556..., 'Cp': 1.50813..., 'Cp_CI_low': 1.29824...,
#  'Cp_CI_high': 1.71768..., 'Cpk': 1.50581..., 'Cpk_CI_low': 1.28613...,
#  'Cpk_CI_high': 1.72549..., 'Pp': 1.57883..., 'Ppk': 1.57640...,
#  'Cpm': 1.57879..., 'confidence': 95, 'normality_passed': True}
```

### Other subgroup structures

The same call adapts the estimator to the data. Verified output:

```text
INDIVIDUALS (subgroup_size=1)  sigma_used: within (MR-bar/d2)
  sigma_within=0.55286  sigma_overall=0.57708   Cpk=1.1884  Ppk=1.1385

UNEQUAL subgroups               sigma_used: within (pooled)
  sigma_within=0.32545  sigma_overall=0.30076
```

Note that in the individuals case the within and overall sigmas differ even though both
summarize the same series: `within (MR-bar/d2)` uses the moving range
($\overline{MR}/d_2(2)$, $d_2(2)=1.128$), filtering point-to-point variation only.

## 6. Source standard

mfgQC's capability indices, the $\bar R/d_2$ and $\overline{MR}/d_2$ within-sigma
estimators, the $d_2$ constants, and both confidence-interval formulas (eq. 8.19 for
$C_p$, eq. 8.21 for $C_{pk}$) are pinned to **Montgomery, *Introduction to Statistical
Quality Control***, mfgQC's primary source for SPC and capability. See the
[Bibliography](bibliography.md).

## See also

- [Non-normal capability](non-normal-capability.md): Box-Cox, Clements/percentile, and
  Johnson methods for skewed processes (where CIs read `n/a`).
- [Control charts](control-charts.md): establish statistical control *before* you
  compute capability.
- [Reading the assumption report](../guide/assumption-report.ipynb): what the
  normality and subgroup-sufficiency checks mean and how to act on them.
- [API reference](api.md): `CapabilityResult` fields, `summary()`, `to_dict()`,
  `view()`.
- [Bibliography](bibliography.md): the cited sources.
