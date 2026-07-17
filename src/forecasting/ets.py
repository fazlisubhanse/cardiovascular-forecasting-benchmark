"""Nonseasonal exponential-smoothing model selection by training-only AICc."""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .base import ForecastBatch, ForecastOutput, calculate_aicc, empirical_error_intervals


class EtsForecaster:
    """Select level/additive/damped ETS candidates independently at each origin."""

    name = "ETS"
    family = "Exponential smoothing"

    def __init__(self, config: dict[str, Any], minimum_residual_count: int) -> None:
        self.config = config
        self.minimum_residual_count = minimum_residual_count

    def forecast_many(self, years: np.ndarray, values: np.ndarray, horizons: list[int], seed: int,
                      context: dict[str, Any]) -> ForecastBatch:
        """Search ETS candidates on training data, refit the winner, and forecast."""

        selection_start = time.perf_counter()
        candidate_rows: list[dict[str, Any]] = []
        successes: list[tuple[float, str | None, bool, Any, str]] = []
        for trend in self.config["candidate_trends"]:
            for damped in self.config["damped_options"]:
                specification = f"trend={trend or 'none'},damped={damped},seasonal=none"
                base = {**context, "trend": trend or "none", "damped": damped, "seasonal": "none"}
                if trend is None and damped:
                    candidate_rows.append({**base, "fit_success": False, "warning": "Damping requires a trend.",
                                           "aic": np.nan, "aicc": np.nan, "selected": False})
                    continue
                try:
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        result = ExponentialSmoothing(
                            values, trend=trend, damped_trend=damped, seasonal=None,
                            initialization_method="estimated",
                        ).fit(optimized=True, remove_bias=False)
                    warning_text = " | ".join(str(item.message) for item in captured)
                    aic = float(result.aic)
                    aicc = float(result.aicc) if np.isfinite(result.aicc) else calculate_aicc(aic, int(result.k), len(values))
                    finite = np.isfinite(np.asarray(list(result.params_formatted["param"]), dtype=float)).all()
                    if not finite or not np.isfinite(aicc):
                        raise ValueError("Nonfinite fitted parameters or AICc.")
                    successes.append((aicc, trend, damped, result, warning_text))
                    candidate_rows.append({**base, "fit_success": True, "warning": warning_text,
                                           "aic": aic, "aicc": aicc, "selected": False})
                except Exception as exc:
                    candidate_rows.append({**base, "fit_success": False, "warning": f"{type(exc).__name__}: {exc}",
                                           "aic": np.nan, "aicc": np.nan, "selected": False})
        selection_seconds = time.perf_counter() - selection_start
        if not successes:
            message = "Every configured ETS candidate failed."
            outputs = {h: ForecastOutput(warning=message, specification="ETS candidate search", criterion="AICc",
                                         selection_seconds=selection_seconds / len(horizons)) for h in horizons}
            return ForecastBatch(outputs, self._expand_candidates(candidate_rows, horizons))
        best_score, best_trend, best_damped, _, selection_warning = min(successes, key=lambda item: item[0])
        for row in candidate_rows:
            row["selected"] = bool(row["fit_success"] and row["trend"] == (best_trend or "none") and row["damped"] == best_damped)
        fit_start = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                fitted = ExponentialSmoothing(
                    values, trend=best_trend, damped_trend=best_damped, seasonal=None,
                    initialization_method="estimated",
                ).fit(optimized=True, remove_bias=False)
            fit_seconds = time.perf_counter() - fit_start
            warnings_text = " | ".join(filter(None, [selection_warning, *(str(item.message) for item in captured)]))
            forecast_start = time.perf_counter()
            forecast = np.asarray(fitted.forecast(max(horizons)), dtype=float)
            residuals = np.asarray(fitted.resid, dtype=float)
            forecast_seconds = time.perf_counter() - forecast_start
            outputs = {}
            for horizon in horizons:
                point = float(forecast[horizon - 1])
                if not np.isfinite(point):
                    raise ValueError(f"Nonfinite ETS forecast at horizon {horizon}.")
                bounds = empirical_error_intervals(point, residuals, self.minimum_residual_count)
                outputs[horizon] = ForecastOutput(
                    point, *bounds, "success", warnings_text,
                    f"trend={best_trend or 'none'},damped={best_damped},seasonal=none; residual-quantile intervals",
                    "AICc", best_score, selection_seconds / len(horizons), fit_seconds / len(horizons),
                    forecast_seconds / len(horizons),
                )
            return ForecastBatch(outputs, self._expand_candidates(candidate_rows, horizons))
        except Exception as exc:
            message = f"Selected ETS refit/forecast failed: {type(exc).__name__}: {exc}"
            outputs = {h: ForecastOutput(warning=message, specification="selected ETS refit", criterion="AICc",
                                         score=best_score, selection_seconds=selection_seconds / len(horizons)) for h in horizons}
            return ForecastBatch(outputs, self._expand_candidates(candidate_rows, horizons))

    @staticmethod
    def _expand_candidates(rows: list[dict[str, Any]], horizons: list[int]) -> list[dict[str, Any]]:
        """Attach each evaluated training-only candidate to applicable horizons."""

        return [{**row, "horizon": horizon} for horizon in horizons for row in rows]
