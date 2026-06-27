# About

**mfgQC** is built and maintained by [Brantner Solutions](https://brantnersolutions.com).

It exists to give manufacturing practitioners quality-control analysis they can
*trust and defend* — not just compute. The three design pillars are invariants,
not preferences:

1. **Statistical guardrails.** Analyses check their own assumptions and report the
   outcome; they never silently switch methods.
2. **Practitioner-oriented.** The public surface assumes domain knowledge, not a
   statistics or programming background.
3. **Auditable by construction.** Results are immutable and carry a hash-chained
   provenance history, so a reported number can be traced back to raw data and
   verified against tampering.

## Provenance of the methods themselves

mfgQC is validated in two independent layers. *Regression* tests pin it to its
build oracles (Montgomery; AIAG MSA 4th ed.; Lawson, *Design and Analysis of
Experiments with R*). A separate *correctness* suite pins each analysis to an
independent source it was **not** built against — the NIST/SEMATECH e-Handbook and
StRD certified datasets, the R `qcc`/`SixSigma` packages, and scipy/statsmodels
computed in-test. No expected value in that suite is ever taken from a prior mfgQC
run. See the [Bibliography](reference/bibliography.md) for the full source list.

## Links

- **Documentation:** [mfgqc.brantnersolutions.com](https://mfgqc.brantnersolutions.com)
- **PyPI:** [pypi.org/project/mfgqc](https://pypi.org/project/mfgqc/)
- **Source:** [github.com/cjbrant/mfgQC](https://github.com/cjbrant/mfgQC)
- **Brantner Solutions:** [brantnersolutions.com](https://brantnersolutions.com)

## License

MIT. See [LICENSE](https://github.com/cjbrant/mfgQC/blob/main/LICENSE).
