"""Random-walk-with-drift baseline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import ForecastBatch, ForecastOutput, empirical_error_intervals, historical_drift_errors


class DriftForecaster:
    """Extrapolate the average change from first to last training observation."""

    name = "Drift"
    family = "Simple baseline"

    def __init__(self, minimum_residual_count: int) -> None:
        self.minimum_residual_count = minimum_residual_count

    def forecast_many(self, years: np.ndarray, values: np.ndarray, horizons: list[int], seed: int,
                      context: dict[str, Any]) -> ForecastBatch:
        """Produce deterministic drift forecasts and empirical intervals."""

        start = time.perf_counter()
        if len(values) < 2:
            return ForecastBatch({h: ForecastOutput(warning="At least two observations are required.") for h in horizons}, [])
        slope = (float(values[-1]) - float(values[0])) / (len(values) - 1)
        elapsed = time.perf_counter() - start
        outputs = {}
        for horizon in horizons:
            point = float(values[-1] + horizon * slope)
            bounds = empirical_error_intervals(
                point, historical_drift_errors(values, horizon), self.minimum_residual_count
            )
            outputs[horizon] = ForecastOutput(
                point, *bounds, "success", "", f"random walk with drift; slope={slope:.12g}",
                "not applicable", np.nan, 0.0, elapsed / len(horizons), 0.0,
            )
        return ForecastBatch(outputs, [])
