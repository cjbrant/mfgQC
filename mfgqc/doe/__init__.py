"""Design of experiments (two-level factorial and regular fractional).

Generation is pre-data and module-level; analysis is post-data and a method on
``QCData``::

    import mfgqc
    from mfgqc import design

    d = design.fractional_factorial(["A", "B", "C", "D", "E"], fraction="1/2")
    sheet = d.run_sheet()                      # collect responses against this

    qc = mfgqc.load(filled_df, measure="y")
    res = qc.doe(design=d)                      # or qc.doe(factors=[...], order=2)
    print(res.report())
    res.view(kind="halfnormal")

The module owns the generators, defining relation, resolution and alias
structure; it surfaces confounding and design-adequacy flags and never silently
alters the design or the response.
"""

from __future__ import annotations

from .analysis import DOEResult, doe
from .design import Design, fractional_factorial, full_factorial
from .significance import LenthResult, lenth

__all__ = [
    "Design", "full_factorial", "fractional_factorial",
    "DOEResult", "doe", "LenthResult", "lenth",
]
