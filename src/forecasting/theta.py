"""Statsmodels Theta-method baseline with native prediction intervals."""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.forecasting.theta import ThetaModel

from .base import ForecastBatch, ForecastOutput, empirical_error_intervals


class ThetaForecaster:
    """Fit statsmodels ThetaModel without seasonal decomposition for annual data."""

    name = "Theta"
    family = "Theta"

    def __init__(self, minimum_residual_count: int) -> None:
        self.minimum_residual_count = minimum_residual_count

    def forecast_many(self, years: np.ndarray, values: np.ndarray, horizons: list[int], seed: int,
                      context: dict[str, Any]) -> ForecastBatch:
        """Fit Theta on current training data only; never substitute another model."""

        fit_start = time.perf_counter()
        try:
            series = pd.Series(np.asarray(values, dtype=float), index=pd.RangeIndex(int(years[0]), int(years[-1]) + 1))
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                fitted = ThetaModel(series, period=1, deseasonalize=False, use_test=False).fit()
            fit_seconds = time.perf_counter() - fit_start
            fit_warning_text = " | ".join(str(item.message) for item in captured)
            forecast_start = time.perf_counter()
            with warnings.catch_warnings(record=True) as forecast_warnings:
                warnings.simplefilter("always")
                points = np.asarray(fitted.forecast(steps=max(horizons)), dtype=float)
                native_available = True
                interval_error = ""
                try:
                    pi80 = np.asarray(fitted.prediction_intervals(steps=max(horizons), alpha=0.20), dtype=float)
                    pi95 = np.asarray(fitted.prediction_intervals(steps=max(horizons), alpha=0.05), dtype=float)
                except Exception as exc:
                    native_available = False
                    pi80 = pi95 = np.full((max(horizons), 2), np.nan)
                    interval_error = f"Native interval unavailable: {type(exc).__name__}: {exc}"
            warning_text = " | ".join(filter(None, [
                fit_warning_text,
                *(str(item.message) for item in forecast_warnings),
                interval_error,
            ]))
            forecast_seconds = time.perf_counter() - forecast_start
            outputs = {}
            residuals = np.diff(values) - np.mean(np.diff(values))
            for horizon in horizons:
                index = horizon - 1
                point = float(points[index])
                if not np.isfinite(point):
                    raise ValueError(f"Nonfinite Theta forecast at horizon {horizon}.")
                if native_available and np.isfinite(np.array([*pi80[index], *pi95[index]])).all():
                    bounds = (float(pi80[index, 0]), float(pi80[index, 1]), float(pi95[index, 0]), float(pi95[index, 1]))
                    interval_method = "statsmodels native Theta intervals"
                else:
                    bounds = empirical_error_intervals(point, residuals, self.minimum_residual_count)
                    interval_method = "training-difference residual-quantile intervals"
                outputs[horizon] = ForecastOutput(
                    point, *bounds, "success", warning_text,
                    f"ThetaModel(period=1,deseasonalize=False,use_test=False); {interval_method}",
                    "not applicable", np.nan, 0.0, fit_seconds / len(horizons), forecast_seconds / len(horizons),
                )
            return ForecastBatch(outputs, [])
        except Exception as exc:
            message = f"Theta fit/forecast failed: {type(exc).__name__}: {exc}"
            return ForecastBatch({h: ForecastOutput(warning=message,
                                                    specification="ThetaModel(period=1,deseasonalize=False,use_test=False)") for h in horizons}, [])
