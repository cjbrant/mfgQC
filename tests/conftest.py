"""Shared fixtures with reference data transcribed from published sources.

Sources:
- Montgomery, *Introduction to Statistical Quality Control*, Example 6.1
  (hard-bake / flow width) and Sec. 8.2.2 (glass bursting strength).
- AIAG *MSA Reference Manual*, 4th ed., Fig. III-B 15 (gage R&R data).
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless rendering for view() tests

import mfgqc

# --------------------------------------------------------------------------- #
# Montgomery hard-bake / flow-width: 25 subgroups x n=5. Spec 1.50 +/- 0.50.
# --------------------------------------------------------------------------- #
_MONTGOMERY = [
    [1.3235, 1.4128, 1.6744, 1.4573, 1.6914],
    [1.4314, 1.3592, 1.6075, 1.4666, 1.6109],
    [1.4284, 1.4871, 1.4932, 1.4324, 1.5674],
    [1.5028, 1.6352, 1.3841, 1.2831, 1.5507],
    [1.5604, 1.2735, 1.5265, 1.4363, 1.6441],
    [1.5955, 1.5451, 1.3574, 1.3281, 1.4198],
    [1.6274, 1.5064, 1.8366, 1.4177, 1.5144],
    [1.4190, 1.4303, 1.6637, 1.6067, 1.5519],
    [1.3884, 1.7277, 1.5355, 1.5176, 1.3688],
    [1.4039, 1.6697, 1.5089, 1.4627, 1.5220],
    [1.4158, 1.7667, 1.4278, 1.5928, 1.4181],
    [1.5821, 1.3355, 1.5777, 1.3908, 1.7559],
    [1.2856, 1.4106, 1.4447, 1.6398, 1.1928],
    [1.4951, 1.4036, 1.5893, 1.6458, 1.4969],
    [1.3589, 1.2863, 1.5996, 1.2497, 1.5471],
    [1.5747, 1.5301, 1.5171, 1.1839, 1.8662],
    [1.3680, 1.7269, 1.3957, 1.5014, 1.4449],
    [1.4163, 1.3864, 1.3057, 1.6210, 1.5573],
    [1.5796, 1.4185, 1.6541, 1.5116, 1.7247],
    [1.7106, 1.4412, 1.2361, 1.3820, 1.7601],
    [1.4371, 1.5051, 1.3485, 1.5670, 1.4880],
    [1.4738, 1.5936, 1.6583, 1.4973, 1.4720],
    [1.5917, 1.4333, 1.5551, 1.5295, 1.6866],
    [1.6399, 1.5243, 1.5705, 1.5563, 1.5530],
    [1.5797, 1.3663, 1.6240, 1.3732, 1.6887],
]


@pytest.fixture
def montgomery_qc() -> mfgqc.QCData:
    rows = []
    for sg, vals in enumerate(_MONTGOMERY, start=1):
        for v in vals:
            rows.append({"subgroup": sg, "width": v})
    df = pd.DataFrame(rows)
    return mfgqc.load(
        df, measure="width",
        roles={"subgroup": "subgroup"}, units="microns", subgroup_size=5,
    ).spec(lower=1.0, upper=2.0, target=1.5)


# --------------------------------------------------------------------------- #
# AIAG gage R&R: 10 parts x 3 appraisers x 3 trials.
# Rows below are (appraiser, trial) -> measurement for parts 1..10.
# --------------------------------------------------------------------------- #
_AIAG = {
    ("A", 1): [0.29, -0.56, 1.34, 0.47, -0.80, 0.02, 0.59, -0.31, 2.26, -1.36],
    ("A", 2): [0.41, -0.68, 1.17, 0.50, -0.92, -0.11, 0.75, -0.20, 1.99, -1.25],
    ("A", 3): [0.64, -0.58, 1.27, 0.64, -0.84, -0.21, 0.66, -0.17, 2.01, -1.31],
    ("B", 1): [0.08, -0.47, 1.19, 0.01, -0.56, -0.20, 0.47, -0.63, 1.80, -1.68],
    ("B", 2): [0.25, -1.22, 0.94, 1.03, -1.20, 0.22, 0.55, 0.08, 2.12, -1.62],
    ("B", 3): [0.07, -0.68, 1.34, 0.20, -1.28, 0.06, 0.83, -0.34, 2.19, -1.50],
    ("C", 1): [0.04, -1.38, 0.88, 0.14, -1.46, -0.29, 0.02, -0.46, 1.77, -1.49],
    ("C", 2): [-0.11, -1.13, 1.09, 0.20, -1.07, -0.67, 0.01, -0.56, 1.45, -1.77],
    ("C", 3): [-0.15, -0.96, 0.67, 0.11, -1.45, -0.49, 0.21, -0.49, 1.87, -2.16],
}


@pytest.fixture
def aiag_qc() -> mfgqc.QCData:
    rows = []
    for (op, trial), parts in _AIAG.items():
        for part_idx, value in enumerate(parts, start=1):
            rows.append({"part": part_idx, "operator": op, "trial": trial, "y": value})
    df = pd.DataFrame(rows)
    return mfgqc.load(
        df, measure="y",
        roles={"part": "part", "operator": "operator", "replicate": "trial"},
    )


# --------------------------------------------------------------------------- #
# Glass-container bursting strength (n=20), approximately normal.
# --------------------------------------------------------------------------- #
_GLASS = [197, 200, 215, 221, 231, 242, 245, 258, 265, 265,
          271, 275, 277, 278, 280, 283, 290, 301, 318, 346]


@pytest.fixture
def glass_values() -> np.ndarray:
    return np.array(_GLASS, dtype=float)


@pytest.fixture
def one_sided_qc() -> mfgqc.QCData:
    """A sample standardized to exactly mean=264, sample sd=32 (Montgomery Ex 8.2)."""
    base = np.linspace(-1, 1, 40)
    z = (base - base.mean()) / base.std(ddof=1)
    vals = z * 32.0 + 264.0
    df = pd.DataFrame({"strength": vals})
    return mfgqc.load(df, measure="strength").spec(lower=200.0)


@pytest.fixture
def skewed_qc() -> mfgqc.QCData:
    """Clearly non-normal (exponential) data with a spec, for the fail path."""
    rng = np.random.default_rng(12345)
    vals = rng.exponential(scale=2.0, size=200) + 0.5
    df = pd.DataFrame({"x": vals})
    return mfgqc.load(df, measure="x").spec(lower=0.0, upper=12.0)


# --------------------------------------------------------------------------- #
# Montgomery Table 9.1 individuals (mu0=10, sigma=1, n=30); upward drift in the
# last third. Shared by the CUSUM and EWMA oracle tests.
# --------------------------------------------------------------------------- #
_TABLE_9_1 = [
    9.45, 7.99, 9.29, 11.66, 12.16, 10.18, 8.04, 11.46, 9.20, 10.34,
    9.03, 11.47, 10.51, 9.40, 10.08, 9.37, 10.62, 10.31, 8.52, 10.84,
    10.90, 9.33, 12.29, 11.50, 10.60, 11.08, 10.38, 11.62, 11.31, 10.52,
]


@pytest.fixture
def table_9_1_qc() -> mfgqc.QCData:
    return mfgqc.load(pd.DataFrame({"x": _TABLE_9_1}), measure="x")
