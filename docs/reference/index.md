# Reference

This is the implementation guide. There is one page per method family, and each
states three things explicitly:

1. **The formula**, what mfgQC actually computes.
2. **The assumptions**, what has to hold for the number to mean what it says, and
   which of these mfgQC checks for you.
3. **The source standard**, the published reference the method is pinned to.

That structure is deliberate: it makes the documentation itself auditable, which is
the same property mfgQC gives your analyses.

## Pages

- **[Capability](capability.md)**, Cp/Cpk, Pp/Ppk, Cpm, and the σ estimator used.
- **[Non-normal capability](non-normal-capability.md)**, Box-Cox, Clements
  percentile, Johnson-system; when each applies.
- **[Control charts](control-charts.md)**, the constants table and the
  variables/attributes chart families.
- **[Run rules](run-rules.md)**, the Western Electric and Nelson rule sets.
- **[Gage R&R](gage-rr.md)**, the ANOVA decomposition, the AIAG pooling rule, ndc.
- **[Bayesian analytics](bayesian.md)**, posterior capability, attributes, comparison,
  assurance, guardband, monitoring, and the conjugate engines behind them.
- **[Provenance model](provenance.md)**, the data model, immutability guarantees,
  the digest/hash-chain spec, and exactly what `verify_provenance()` proves.
- **[API reference](api.md)**, auto-generated from the in-code docstrings.
- **[Bibliography](bibliography.md)**, full citations for every source above.

!!! note "Notation"
    Throughout, $n$ is the subgroup size, $m$ the number of subgroups, $\sigma$ a
    process standard deviation (with the estimator named explicitly per page), and
    $\bar{R}$, $\bar{S}$ the mean subgroup range and standard deviation.
