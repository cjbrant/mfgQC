"""Prior distributions and elicitation.

Typed conjugate priors for the three engines, each carrying a ``to_params()``
that feeds the provenance digest. NormalPrior.from_interval implements the
interval-and-worth elicitation of Hoff sec 5.5 (spec Algorithm E).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from scipy import stats


@dataclass(frozen=True)
class NormalPrior:
    """Normal-Inverse-chi2 prior for the Normal model (mu0, k0, nu0, s20).

    parent_digest is set only when the prior was built from a previous analysis
    result (see :meth:`from_result`); it pins the lineage into the digest.
    """

    mu0: float
    k0: float
    nu0: float
    s20: float
    parent_digest: str | None = None
    _parent_history: tuple = field(default=(), repr=False, compare=False)

    def to_params(self) -> dict:
        params: dict = {
            "family": "normal",
            "hyperparams": {
                "mu0": float(self.mu0), "k0": float(self.k0),
                "nu0": float(self.nu0), "s20": float(self.s20),
            },
        }
        if self.parent_digest is not None:
            params["parent_digest"] = self.parent_digest
        return params

    @classmethod
    def from_interval(cls, mean_between: tuple, *, confidence: float = 0.95,
                      worth: float, sd: float | None = None,
                      sd_worth: float | None = None) -> "NormalPrior":
        """Build a prior from a plausible interval for the mean.

        ``mean_between=(lo, hi)`` at ``confidence`` becomes the central credible
        interval of the prior marginal for mu, t_{nu0}(mu0, s20/k0). ``worth`` is
        the prior sample size k0; ``sd_worth`` sets nu0 (defaults to worth). If
        ``sd`` is given, s20 is sd**2 (the interval then only sets the location).
        """
        lo, hi = mean_between
        mu0 = (lo + hi) / 2.0
        k0 = worth
        nu0 = sd_worth if sd_worth is not None else worth
        if sd is not None:
            s20 = float(sd) ** 2
        else:
            half = (hi - lo) / 2.0
            tcrit = float(stats.t.ppf((1.0 + confidence) / 2.0, nu0))
            # tcrit * sqrt(s20 / k0) = half  ->  s20 = (half / tcrit)**2 * k0
            s20 = (half / tcrit) ** 2 * k0
        return cls(mu0=float(mu0), k0=float(k0), nu0=float(nu0), s20=float(s20))

    @classmethod
    def from_expectations(cls, *, mu0: float, prior_var: float, n0: float) -> "NormalPrior":
        """Elicit a prior from prior expectations (Hoff sec 5.5).

        Given a prior mean ``mu0`` for the population mean, a prior expected
        variance ``prior_var`` (E[sigma^2]), and ``n0`` prior observations, the
        conjugate prior is theta|sigma^2 ~ N(mu0, sigma^2/n0) and
        sigma^2 ~ inverse-gamma((n0+3)/2, (n0+1)*prior_var/2). In N-Inv-chi2 form
        that is k0=n0, nu0=n0+3, s20=(n0+1)*prior_var/(n0+3); then E[sigma^2] is
        exactly prior_var, and the n0=1 case reduces to sigma^2 ~ inv-gamma(2, prior_var).
        """
        nu0 = float(n0) + 3.0
        s20 = (float(n0) + 1.0) * float(prior_var) / nu0
        return cls(mu0=float(mu0), k0=float(n0), nu0=nu0, s20=s20)

    @classmethod
    def from_result(cls, result) -> "NormalPrior":
        """Use a previous fit's posterior as this prior (Bayesian updating across
        analyses). Carries the parent history forward and embeds the parent digest
        so later tampering with any parent step breaks the child's verify (T5.3).
        """
        return cls(
            mu0=float(result.mun), k0=float(result.kn),
            nu0=float(result.nun), s20=float(result.sn2),
            parent_digest=result.provenance_digest(),
            _parent_history=tuple(result.history),
        )


@dataclass(frozen=True)
class BetaPrior:
    """Beta prior for a proportion, Beta(a, b)."""

    a: float
    b: float

    def to_params(self) -> dict:
        return {"family": "beta", "hyperparams": {"a": float(self.a), "b": float(self.b)}}


@dataclass(frozen=True)
class GammaPrior:
    """Gamma prior for a rate, Gamma(a, rate=b)."""

    a: float
    b: float

    def to_params(self) -> dict:
        return {"family": "gamma", "hyperparams": {"a": float(self.a), "b": float(self.b)}}
