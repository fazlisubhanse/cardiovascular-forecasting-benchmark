"""Fold-local time-series transformations for direct-horizon forecasting."""

from __future__ import annotations

import numpy as np


def fit_linear_trend(years: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Fit an OLS intercept/slope using only the supplied fold history."""

    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(intercept), float(slope)


def apply_representation(years: np.ndarray, values: np.ndarray, representation: str) -> tuple[np.ndarray, dict[str, float]]:
    """Return the fold-local modeled series and auditable transform state."""

    years, values = np.asarray(years, dtype=int), np.asarray(values, dtype=float)
    if representation == "raw_level_direct":
        return values.copy(), {}
    if representation == "first_difference_direct":
        return np.diff(values), {}
    if representation == "linear_detrend_direct":
        intercept, slope = fit_linear_trend(years, values)
        return values - (intercept + slope * years), {"trend_intercept": intercept, "trend_slope": slope}
    raise ValueError(f"Unknown representation: {representation}")
