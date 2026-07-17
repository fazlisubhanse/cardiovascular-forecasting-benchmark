"""Bounded ARIMA order/trend selection by training-only AICc."""

from __future__ import annotations

import itertools
import time
import warnings
from typing import Any

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from .base import ForecastBatch, ForecastOutput, calculate_aicc


class ArimaForecaster:
    """Grid-search nonseasonal ARIMA candidates at every expanding origin."""

    name = "ARIMA"
    family = "ARIMA"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def _trends(self, d_value: int) -> list[str]:
        if not self.config.get("include_drift_or_trend", True):
            return ["n"]
        if d_value == 0:
            return ["n", "c", "t", "ct"]
        if d_value == 1:
            return ["n", "t"]
        return ["n"]

    def forecast_many(self, years: np.ndarray, values: np.ndarray, horizons: list[int], seed: int,
                      context: dict[str, Any]) -> ForecastBatch:
        """Select the lowest finite-AICc ARIMA, refit it, and use native intervals."""

        selection_start = time.perf_counter()
        candidates: list[dict[str, Any]] = []
        successful: list[tuple[float, tuple[int, int, int], str, str]] = []
        for p_value, d_value, q_value in itertools.product(
            self.config["p_values"], self.config["d_values"], self.config["q_values"]
        ):
            for trend in self._trends(d_value):
                base = {**context, "order_p": p_value, "order_d": d_value, "order_q": q_value, "trend": trend}
                try:
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        fitted = ARIMA(
                            values, order=(p_value, d_value, q_value), trend=trend,
                            enforce_stationarity=False, enforce_invertibility=False,
                        ).fit()
                    warning_text = " | ".join(str(item.message) for item in captured)
                    parameters = np.asarray(fitted.params, dtype=float)
                    if not np.isfinite(parameters).all():
                        raise ValueError("Nonfinite fitted parameters.")
                    probe = np.asarray(fitted.forecast(steps=max(horizons)), dtype=float)
                    if probe.size < max(horizons) or not np.isfinite(probe).all():
                        raise ValueError("Candidate could not produce finite requested forecasts.")
                    aic = float(fitted.aic)
                    aicc = calculate_aicc(aic, len(parameters), len(values))
                    if not np.isfinite(aicc):
                        raise ValueError("AICc is undefined for this candidate.")
                    successful.append((aicc, (p_value, d_value, q_value), trend, warning_text))
                    candidates.append({**base, "fit_success": True, "warning": warning_text,
                                       "aic": aic, "aicc": aicc, "selected": False})
                except Exception as exc:
                    candidates.append({**base, "fit_success": False, "warning": f"{type(exc).__name__}: {exc}",
                                       "aic": np.nan, "aicc": np.nan, "selected": False})
        selection_seconds = time.perf_counter() - selection_start
        maximum_failed = self.config.get("maximum_failed_candidates_allowed")
        failed_count = sum(not row["fit_success"] for row in candidates)
        if maximum_failed is not None and failed_count > int(maximum_failed):
            successful = []
        if not successful:
            message = "Every ARIMA candidate failed or the configured failure limit was exceeded."
            outputs = {h: ForecastOutput(warning=message, specification="ARIMA grid search", criterion="AICc",
                                         selection_seconds=selection_seconds / len(horizons)) for h in horizons}
            return ForecastBatch(outputs, self._expand_candidates(candidates, horizons))
        best_score, best_order, best_trend, selection_warning = min(successful, key=lambda item: item[0])
        for row in candidates:
            row["selected"] = bool(
                row["fit_success"] and (row["order_p"], row["order_d"], row["order_q"]) == best_order
                and row["trend"] == best_trend
            )
        fit_start = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                fitted = ARIMA(
                    values, order=best_order, trend=best_trend,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit()
            fit_seconds = time.perf_counter() - fit_start
            warning_text = " | ".join(filter(None, [selection_warning, *(str(item.message) for item in captured)]))
            forecast_start = time.perf_counter()
            prediction = fitted.get_forecast(steps=max(horizons))
            points = np.asarray(prediction.predicted_mean, dtype=float)
            ci80 = np.asarray(prediction.conf_int(alpha=0.20), dtype=float)
            ci95 = np.asarray(prediction.conf_int(alpha=0.05), dtype=float)
            forecast_seconds = time.perf_counter() - forecast_start
            outputs = {}
            for horizon in horizons:
                index = horizon - 1
                values_to_check = np.array([points[index], *ci80[index], *ci95[index]], dtype=float)
                if not np.isfinite(values_to_check).all():
                    raise ValueError(f"Nonfinite ARIMA point/interval at horizon {horizon}.")
                outputs[horizon] = ForecastOutput(
                    float(points[index]), float(ci80[index, 0]), float(ci80[index, 1]),
                    float(ci95[index, 0]), float(ci95[index, 1]), "success", warning_text,
                    f"ARIMA{best_order},trend={best_trend}", "AICc", best_score,
                    selection_seconds / len(horizons), fit_seconds / len(horizons), forecast_seconds / len(horizons),
                )
            return ForecastBatch(outputs, self._expand_candidates(candidates, horizons))
        except Exception as exc:
            message = f"Selected ARIMA refit/forecast failed: {type(exc).__name__}: {exc}"
            outputs = {h: ForecastOutput(warning=message, specification=f"ARIMA{best_order},trend={best_trend}",
                                         criterion="AICc", score=best_score,
                                         selection_seconds=selection_seconds / len(horizons)) for h in horizons}
            return ForecastBatch(outputs, self._expand_candidates(candidates, horizons))

    @staticmethod
    def _expand_candidates(rows: list[dict[str, Any]], horizons: list[int]) -> list[dict[str, Any]]:
        return [{**row, "horizon": horizon} for horizon in horizons for row in rows]
