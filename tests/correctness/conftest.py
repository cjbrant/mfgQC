"""Shared infrastructure for the correctness suite.

The correctness suite pins mfgQC to sources the code was NOT built against:

* NIST/SEMATECH e-Handbook of Engineering Statistics worked examples
  (https://www.itl.nist.gov/div898/handbook/) and the NIST StRD certified
  linear-regression datasets (https://www.itl.nist.gov/div898/strd/).
* R packages ``qcc`` and ``SixSigma`` run live as an independent engine.
* scipy / statsmodels computed in-test on freshly seeded data.

Rule: no expected value in this suite is ever sourced from a prior mfgQC run.
Either the published authority states input -> output, or an independent engine
computes the answer at test time.

The R-backed tests run ``Rscript`` as a subprocess. The R script emits BOTH the
input data and the oracle answer as JSON, so mfgQC is fed R's data and compared to
R's answer; mfgQC never supplies its own expected value. If Rscript or a required
package is missing the test SKIPS rather than fails.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import mfgqc  # noqa: F401  (imported so matplotlib Agg backend / package init runs)

# --------------------------------------------------------------------------- #
# R bridge
# --------------------------------------------------------------------------- #
_RSCRIPT = shutil.which("Rscript") or "/opt/homebrew/bin/Rscript"


def _have_r() -> bool:
    return bool(_RSCRIPT) and os.path.exists(_RSCRIPT)


def run_r(body: str, packages: tuple[str, ...] = ()) -> dict:
    """Run an R snippet and return the JSON object it prints on its last line.

    ``body`` must end by printing a single JSON object via
    ``cat(jsonlite::toJSON(...))``. Skips the test if Rscript or any required
    package is unavailable, so the suite stays green on machines without R.
    """
    if not _have_r():
        pytest.skip("Rscript not available")
    libs = "\n".join(f'if (!requireNamespace("{p}", quietly=TRUE)) quit(status=2)'
                     for p in (("jsonlite",) + packages))
    script = libs + "\nsuppressMessages({\n" + \
        "\n".join(f'library({p})' for p in (("jsonlite",) + packages)) + "\n})\n" + body
    proc = subprocess.run([_RSCRIPT, "-e", script],
                          capture_output=True, text=True, timeout=180)
    if proc.returncode == 2:
        pytest.skip(f"R package(s) {packages} not installed")
    if proc.returncode != 0:
        pytest.skip(f"R failed (rc={proc.returncode}): {proc.stderr.strip()[-400:]}")
    last = proc.stdout.strip().splitlines()[-1]
    return json.loads(last)


# --------------------------------------------------------------------------- #
# Helper: build a sample with an EXACT mean and sample sd (ddof=1).
# Used where a published source states summary statistics rather than raw data.
# --------------------------------------------------------------------------- #
def exact_sample(mean: float, sd: float, n: int) -> np.ndarray:
    """Return n values with sample mean == mean and sample sd (ddof=1) == sd."""
    z = np.linspace(-1.0, 1.0, n)
    z = (z - z.mean()) / z.std(ddof=1)
    return z * sd + mean


# --------------------------------------------------------------------------- #
# NIST StRD "Norris" certified simple-linear-regression dataset.
# Source: https://www.itl.nist.gov/div898/strd/lls/data/Norris.shtml
# Columns are (y, x). 36 observations.
# --------------------------------------------------------------------------- #
_NORRIS = [
    (0.1, 0.2), (338.8, 337.4), (118.1, 118.2), (888.0, 884.6), (9.2, 10.1),
    (228.1, 226.5), (668.5, 666.3), (998.5, 996.3), (449.1, 448.6), (778.9, 777.0),
    (559.2, 558.2), (0.3, 0.4), (0.1, 0.6), (778.1, 775.5), (668.8, 666.9),
    (339.3, 338.0), (448.9, 447.5), (10.8, 11.6), (557.7, 556.0), (228.3, 228.1),
    (998.0, 995.8), (888.8, 887.6), (119.6, 120.2), (0.3, 0.3), (0.6, 0.3),
    (557.6, 556.8), (339.3, 339.1), (888.0, 887.2), (998.5, 999.0), (778.9, 779.0),
    (10.2, 11.1), (117.6, 118.3), (228.9, 229.2), (668.4, 669.1), (449.2, 448.9),
    (0.2, 0.5),
]

# Certified values from https://www.itl.nist.gov/div898/strd/lls/data/LINKS/v-Norris.shtml
NORRIS_CERT = {
    "intercept": -0.262323073774029,
    "intercept_sd": 0.232818234301152,
    "slope": 1.00211681802045,
    "slope_sd": 0.429796848199937e-03,
    "resid_sd": 0.884796396144373,
    "r_squared": 0.999993745883712,
    "df_resid": 34,
}


@pytest.fixture
def norris_df() -> pd.DataFrame:
    return pd.DataFrame(_NORRIS, columns=["y", "x"])
