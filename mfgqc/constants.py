"""Reference constants for control charts and gage R&R.

These are transcribed standard tables. Control-chart constants follow the
conventional SQC tables (e.g. Montgomery, *Introduction to Statistical Quality
Control*, Appendix VI). Gage R&R ``K`` factors follow the AIAG *MSA Reference
Manual*, 4th ed.

Keeping these as plain data (not computed on the fly) is deliberate: it makes
the values auditable against the published tables a practitioner trusts.
"""

from __future__ import annotations

# Control-chart constants by subgroup size n.
# Columns: d2, d3, A2, A3, D3, D4, B3, B4, c4
# Source: standard SQC tables (Montgomery, Appendix VI).
_CONTROL_TABLE: dict[int, dict[str, float]] = {
    2:  dict(d2=1.128, d3=0.853, A2=1.880, A3=2.659, D3=0.000, D4=3.267, B3=0.000, B4=3.267, c4=0.7979),
    3:  dict(d2=1.693, d3=0.888, A2=1.023, A3=1.954, D3=0.000, D4=2.574, B3=0.000, B4=2.568, c4=0.8862),
    4:  dict(d2=2.059, d3=0.880, A2=0.729, A3=1.628, D3=0.000, D4=2.282, B3=0.000, B4=2.266, c4=0.9213),
    5:  dict(d2=2.326, d3=0.864, A2=0.577, A3=1.427, D3=0.000, D4=2.114, B3=0.000, B4=2.089, c4=0.9400),
    6:  dict(d2=2.534, d3=0.848, A2=0.483, A3=1.287, D3=0.000, D4=2.004, B3=0.030, B4=1.970, c4=0.9515),
    7:  dict(d2=2.704, d3=0.833, A2=0.419, A3=1.182, D3=0.076, D4=1.924, B3=0.118, B4=1.882, c4=0.9594),
    8:  dict(d2=2.847, d3=0.820, A2=0.373, A3=1.099, D3=0.136, D4=1.864, B3=0.185, B4=1.815, c4=0.9650),
    9:  dict(d2=2.970, d3=0.808, A2=0.337, A3=1.032, D3=0.184, D4=1.816, B3=0.239, B4=1.761, c4=0.9693),
    10: dict(d2=3.078, d3=0.797, A2=0.308, A3=0.975, D3=0.223, D4=1.777, B3=0.284, B4=1.716, c4=0.9727),
    11: dict(d2=3.173, d3=0.787, A2=0.285, A3=0.927, D3=0.256, D4=1.744, B3=0.321, B4=1.679, c4=0.9754),
    12: dict(d2=3.258, d3=0.778, A2=0.266, A3=0.886, D3=0.283, D4=1.717, B3=0.354, B4=1.646, c4=0.9776),
    13: dict(d2=3.336, d3=0.770, A2=0.249, A3=0.850, D3=0.307, D4=1.693, B3=0.382, B4=1.618, c4=0.9794),
    14: dict(d2=3.407, d3=0.763, A2=0.235, A3=0.817, D3=0.328, D4=1.672, B3=0.406, B4=1.594, c4=0.9810),
    15: dict(d2=3.472, d3=0.756, A2=0.223, A3=0.789, D3=0.347, D4=1.653, B3=0.428, B4=1.572, c4=0.9823),
    16: dict(d2=3.532, d3=0.750, A2=0.212, A3=0.763, D3=0.363, D4=1.637, B3=0.448, B4=1.552, c4=0.9835),
    17: dict(d2=3.588, d3=0.744, A2=0.203, A3=0.739, D3=0.378, D4=1.622, B3=0.466, B4=1.534, c4=0.9845),
    18: dict(d2=3.640, d3=0.739, A2=0.194, A3=0.718, D3=0.391, D4=1.608, B3=0.482, B4=1.518, c4=0.9854),
    19: dict(d2=3.689, d3=0.734, A2=0.187, A3=0.698, D3=0.403, D4=1.597, B3=0.497, B4=1.503, c4=0.9862),
    20: dict(d2=3.735, d3=0.729, A2=0.180, A3=0.680, D3=0.415, D4=1.585, B3=0.510, B4=1.490, c4=0.9869),
    21: dict(d2=3.778, d3=0.724, A2=0.173, A3=0.663, D3=0.425, D4=1.575, B3=0.523, B4=1.477, c4=0.9876),
    22: dict(d2=3.819, d3=0.720, A2=0.167, A3=0.647, D3=0.434, D4=1.566, B3=0.534, B4=1.466, c4=0.9882),
    23: dict(d2=3.858, d3=0.716, A2=0.162, A3=0.633, D3=0.443, D4=1.557, B3=0.545, B4=1.455, c4=0.9887),
    24: dict(d2=3.895, d3=0.712, A2=0.157, A3=0.619, D3=0.451, D4=1.548, B3=0.555, B4=1.445, c4=0.9892),
    25: dict(d2=3.931, d3=0.708, A2=0.153, A3=0.606, D3=0.459, D4=1.541, B3=0.565, B4=1.435, c4=0.9896),
}

MIN_SUBGROUP_SIZE = min(_CONTROL_TABLE)
MAX_SUBGROUP_SIZE = max(_CONTROL_TABLE)


def control_constant(name: str, n: int) -> float:
    """Return control-chart constant ``name`` for subgroup size ``n``.

    Parameters
    ----------
    name : str
        One of ``d2, d3, A2, A3, D3, D4, B3, B4, c4``.
    n : int
        Subgroup size (2..25).

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``n`` is outside the tabulated range or ``name`` is unknown.
    """
    if n not in _CONTROL_TABLE:
        raise ValueError(
            f"No control-chart constants for subgroup size n={n}; "
            f"tabulated range is {MIN_SUBGROUP_SIZE}..{MAX_SUBGROUP_SIZE}."
        )
    row = _CONTROL_TABLE[n]
    if name not in row:
        raise ValueError(f"Unknown control-chart constant '{name}'.")
    return row[name]


# AIAG gage R&R Average-and-Range method constants (MSA 4th ed.).
# K1 indexed by number of trials (replicates).
_K1_BY_TRIALS: dict[int, float] = {2: 0.8862, 3: 0.5908}
# K2 indexed by number of appraisers (operators).
_K2_BY_APPRAISERS: dict[int, float] = {2: 0.7071, 3: 0.5231}
# K3 indexed by number of parts.
_K3_BY_PARTS: dict[int, float] = {
    2: 0.7071, 3: 0.5231, 4: 0.4467, 5: 0.4030, 6: 0.3742,
    7: 0.3534, 8: 0.3375, 9: 0.3249, 10: 0.3146,
}


def gage_k1(trials: int) -> float:
    """AIAG ``K1`` factor for the given number of trials."""
    if trials not in _K1_BY_TRIALS:
        raise ValueError(
            f"AIAG K1 is only tabulated for {sorted(_K1_BY_TRIALS)} trials; got {trials}."
        )
    return _K1_BY_TRIALS[trials]


def gage_k2(appraisers: int) -> float:
    """AIAG ``K2`` factor for the given number of appraisers."""
    if appraisers not in _K2_BY_APPRAISERS:
        raise ValueError(
            f"AIAG K2 is only tabulated for {sorted(_K2_BY_APPRAISERS)} appraisers; got {appraisers}."
        )
    return _K2_BY_APPRAISERS[appraisers]


def gage_k3(parts: int) -> float:
    """AIAG ``K3`` factor for the given number of parts."""
    if parts not in _K3_BY_PARTS:
        raise ValueError(
            f"AIAG K3 is only tabulated for {sorted(_K3_BY_PARTS)} parts; got {parts}."
        )
    return _K3_BY_PARTS[parts]
