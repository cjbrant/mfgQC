# mfgqc.bayes — Spec Erratum 001 (re: your T4.2 deviation question)

To the build agent, from the spec author.

## Verdict

Your reading is correct and the spec was wrong. "P(Ppk ≥ p̂pk) ≈ 0.5 under the
noninformative prior" does not hold: the plug-in index sits above the posterior
median because the posterior for σ sits above s (the median of χ²_ν is below ν),
so ≈0.45 at the tested n is the *correct* behavior of a *correct*
implementation. I did not intend p̂pk as the posterior median; that variant is
0.5 by construction and tests nothing. Note the irony for the docstrings: this
bias is the module's own sales pitch seen from the other side — the user
guide's worked example (point Ppk 1.17, P(Ppk ≥ 1.33) = 0.074) is the same
phenomenon. The original T4.2 contradicted the feature's reason to exist.

Your interim implementation (location calibration + assert-bias-exists) is
directionally right. Replace it with the following, which upgrades both halves
from assertion to pinned identity.

## Replacement: T4.2a–c (supersedes T4.2 in spec v1.0)

**T4.2a — location, exact (keep yours).** Noninformative prior:
P(μ ≥ x̄ | y) = 0.5 exactly, by symmetry of the t marginal. rtol 1e-12.

**T4.2b — scale, exact (replaces the bias assertion).** Noninformative prior,
ν = n−1: P(σ² ≥ s² | y) = F_{χ²_ν}(ν) exactly, where F is the χ² CDF
(evaluates ≈ 0.52 at n = 60; above 0.5 — the σ side of the same phenomenon).
Pin to the CDF value computed at the test's n. rtol 1e-12.

**T4.2c — capability, quadrature-pinned (new).** One-sided spec only (Ppu
against USL, so the min() side-switch cannot muddy the identity). With
c = 3·p̂pu and X ~ χ²_ν:

    P(Ppu ≥ p̂pu | y) = E_X[ Φ( √n · c · ( √(X/ν) − 1 ) ) ]

Evaluate the expectation by deterministic 1-D quadrature over the χ²_ν density
(Gauss–Legendre on a transformed variable or scipy quad, abs tol ≤ 1e-8), and
require the module's Monte Carlo answer to match at ±3·MCSE. Dataset: reuse the
T2.5 fixture data (seed 20260703), USL = 25.05 only. Docstring must state: the
value is below 0.5 and n-dependent; plug-in optimism is expected model
behavior; cross-reference the user guide worked example and this erratum.

## Bookkeeping

1. File this memo in `tests/bayes/fixtures/` as `erratum_001.md`; reference it
   from the three test docstrings.
2. Phase gate P1 now reads T4.1, T4.2a–c, T4.3 (T4.4 unchanged).
3. No other tests are affected; T2.5 and the user guide are consistent with
   the corrected understanding and need no changes.
4. Derivation sketch for the docstring, so it never has to be reconstructed:
   under the noninformative posterior μ = ȳ + σZ/√n (Z ~ N(0,1) independent of
   σ), σ² = νs²/X. Then Ppu ≥ p̂pu ⇔ (USL−μ)/3σ ≥ (USL−ȳ)/3s ⇔
   Z ≤ √n·c·(σ/s − 1)·(s/σ)·… reduce with σ/s = √(ν/X) to the stated form.
   Verify the reduction symbolically before trusting the docstring copy.

Process note: this is the deviation protocol working as designed — a spec error
surfaced as a written question and the suite got stronger. Logged as erratum
001. Good catch; thank you. Continue with P1 under the amended gate.
