"""Rolling out-of-fold finite-sample conformal intervals."""

from __future__ import annotations

import math
import numpy as np


def finite_sample_quantile(residuals: list[float] | np.ndarray, nominal_level: float) -> tuple[float, int]:
    """Return the capped ceil((n+1)(1-alpha)) order statistic."""

    values = np.sort(np.asarray(residuals, dtype=float))
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, 0
    rank = min(len(values), int(math.ceil((len(values) + 1) * nominal_level)))
    return float(values[rank - 1]), rank


def conformal_bounds(point: float, residuals: list[float], nominal_level: float, minimum_n: int):
    if len(residuals) < minimum_n:
        return np.nan, np.nan, np.nan, 0, False
    quantile, rank = finite_sample_quantile(residuals, nominal_level)
    lower, upper = float(point - quantile), float(point + quantile)
    return lower, upper, quantile, rank, bool(np.isfinite([lower, upper]).all() and lower <= point <= upper)
