"""Two-level design matrix construction (the wrapped piece).

mfgQC owns the generators, defining relation, resolution and alias structure (see
:mod:`mfgqc.doe.alias`); the only thing wrapped here is the construction of the
full and fractional 2-level matrices. For 2-level designs that construction is a
cartesian product of coded levels in standard (Yates) order with factor A
varying fastest, plus generator product columns, so it is implemented directly
in numpy and re-mapped to the oracle coding convention rather than taking a
dependency on a generator library for something this small. pyDOE3 is installed
as the intended wrap target for the higher-level designs deferred to v2.

Coding convention (the contr.FrF2 convention used throughout the oracle):
``x = (actual - midpoint) / half_range``, so a two-level factor codes to -1/+1
and ``effect = 2 * coefficient``.
"""

from __future__ import annotations

import numpy as np


def coded_full_matrix(k: int) -> np.ndarray:
    """The 2^k full-factorial design coded to -1/+1 in standard (Yates) order.

    Factor A varies fastest: for run ``i`` (0-based) and factor ``j`` (0-based,
    A=0), the coded level is ``+1`` when bit ``j`` of ``i`` is set else ``-1``.
    """
    if k < 1:
        raise ValueError(f"need at least one factor; got k={k}.")
    n = 1 << k
    runs = np.arange(n)
    cols = [np.where((runs >> j) & 1, 1, -1) for j in range(k)]
    return np.column_stack(cols).astype(float)


def code(actual: np.ndarray, low: float, high: float) -> np.ndarray:
    """Code an actual-units column to -1/+1 by ``(actual - midpoint)/half_range``."""
    actual = np.asarray(actual, dtype=float)
    midpoint = (high + low) / 2.0
    half = (high - low) / 2.0
    if half == 0:
        raise ValueError("factor low and high levels must differ.")
    return (actual - midpoint) / half


def decode(coded: np.ndarray, low: float, high: float) -> np.ndarray:
    """Inverse of :func:`code`: map -1/+1 back to actual units."""
    coded = np.asarray(coded, dtype=float)
    midpoint = (high + low) / 2.0
    half = (high - low) / 2.0
    return midpoint + coded * half


def product_column(matrix: np.ndarray, indices) -> np.ndarray:
    """Elementwise product of the given factor columns (a generator column).

    ``E = ABCD`` means column E is the product of columns A, B, C, D, which is
    how an added fractional factor is built.
    """
    out = np.ones(matrix.shape[0])
    for j in indices:
        out = out * matrix[:, j]
    return out


def randomized_order(n: int, seed: int | None) -> np.ndarray:
    """A randomized run order (0-based permutation). ``seed=None`` -> identity
    (standard order retained), so generation is reproducible by default."""
    if seed is None:
        return np.arange(n)
    return np.random.default_rng(seed).permutation(n)
