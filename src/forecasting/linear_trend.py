"""Ordinary least-squares deterministic linear-trend baseline."""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import statsmodels.api as sm

from .base import ForecastBatch, ForecastOutput


class LinearTrendForecaster:
    """Fit target on the actual calendar year and use regression prediction intervals."""

    name = "LinearTrend"
    family = "Regression baseline"

    def forecast_many(self, years: np.ndarray, values: np.ndarray, horizons: list[int], seed: int,
                      context: dict[str, Any]) -> ForecastBatch:
        """Fit OLS once at an origin and forecast all requested years."""

        fit_start = time.perf_counter()
        captured: list[warnings.WarningMessage]
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                design = sm.add_constant(np.asarray(years, dtype=float), has_constant="add")
                result = sm.OLS(np.asarray(values, dtype=float), design).fit()
            fit_seconds = time.perf_counter() - fit_start
            warning_text = " | ".join(str(item.message) for item in captured)
            if not np.isfinite(result.params).all():
                raise ValueError("OLS returned nonfinite fitted parameters.")
            forecast_start = time.perf_counter()
            outputs = {}
            for horizon in horizons:
                target_year = float(years[-1] + horizon)
                prediction = result.get_prediction(np.array([[1.0, target_year]]))
                point = float(prediction.predicted_mean[0])
                sf80 = prediction.summary_frame(alpha=0.20).iloc[0]
                sf95 = prediction.summary_frame(alpha=0.05).iloc[0]
                outputs[horizon] = ForecastOutput(
                    point, float(sf80["obs_ci_lower"]), float(sf80["obs_ci_upper"]),
                    float(sf95["obs_ci_lower"]), float(sf95["obs_ci_upper"]), "success", warning_text,
                    f"OLS target ~ const + calendar year; beta1={result.params[1]:.12g}",
                    "not applicable", np.nan, 0.0, fit_seconds / len(horizons), 0.0,
                )
            elapsed = time.perf_counter() - forecast_start
            for output in outputs.values():
                output.forecast_seconds = elapsed / len(horizons)
            return ForecastBatch(outputs, [])
        except Exception as exc:
            warning_text = f"{type(exc).__name__}: {exc}"
            return ForecastBatch({h: ForecastOutput(warning=warning_text, specification="OLS calendar-year trend") for h in horizons}, [])
