"""Random-walk/last-observation naive baseline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import ForecastBatch, ForecastOutput, empirical_error_intervals, historical_naive_errors


class NaiveForecaster:
    """Forecast every horizon as the most recent training observation."""

    name = "Naive"
    family = "Simple baseline"

    def __init__(self, minimum_residual_count: int) -> None:
        self.minimum_residual_count = minimum_residual_count

    def forecast_many(self, years: np.ndarray, values: np.ndarray, horizons: list[int], seed: int,
                      context: dict[str, Any]) -> ForecastBatch:
        """Produce naive forecasts and training-only empirical intervals."""

        start = time.perf_counter()
        point = float(values[-1])
        outputs: dict[int, ForecastOutput] = {}
        elapsed = time.perf_counter() - start
        for horizon in horizons:
            lower80, upper80, lower95, upper95 = empirical_error_intervals(
                point, historical_naive_errors(values, horizon), self.minimum_residual_count
            )
            outputs[horizon] = ForecastOutput(
                point, lower80, upper80, lower95, upper95, "success", "",
                "last observation carried forward", "not applicable", np.nan,
                0.0, 0.0, elapsed / len(horizons),
            )
        return ForecastBatch(outputs, [])
