"""Shared forecast result types and training-only interval utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class ForecastOutput:
    """Output for one horizon from one fit at one forecast origin."""

    point: float = np.nan
    lower_80: float = np.nan
    upper_80: float = np.nan
    lower_95: float = np.nan
    upper_95: float = np.nan
    status: str = "failed"
    warning: str = ""
    specification: str = ""
    criterion: str = ""
    score: float = np.nan
    selection_seconds: float = 0.0
    fit_seconds: float = 0.0
    forecast_seconds: float = 0.0


@dataclass
class ForecastBatch:
    """Horizon outputs plus optional candidate-level selection records."""

    outputs: dict[int, ForecastOutput]
    candidates: list[dict[str, Any]]


class Forecaster(Protocol):
    """Protocol implemented by every Phase 2A baseline."""

    name: str
    family: str

    def forecast_many(
        self, years: np.ndarray, values: np.ndarray, horizons: list[int], seed: int, context: dict[str, Any]
    ) -> ForecastBatch:
        """Fit on training data and produce requested future horizons."""


def empirical_error_intervals(
    point: float, errors: np.ndarray, minimum_count: int
) -> tuple[float, float, float, float]:
    """Return deterministic empirical 80%/95% error-quantile intervals."""

    finite = np.asarray(errors, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < minimum_count or not np.isfinite(point):
        return np.nan, np.nan, np.nan, np.nan
    centered = finite - np.mean(finite)
    q025, q10, q90, q975 = np.quantile(centered, [0.025, 0.10, 0.90, 0.975])
    return min(point, point + q10), max(point, point + q90), min(point, point + q025), max(point, point + q975)


def calculate_aicc(aic: float, parameter_count: int, observation_count: int) -> float:
    """Calculate AICc, returning NaN when its denominator is not positive."""

    denominator = observation_count - parameter_count - 1
    if not np.isfinite(aic) or denominator <= 0:
        return np.nan
    return float(aic + (2 * parameter_count * (parameter_count + 1)) / denominator)


def historical_naive_errors(values: np.ndarray, horizon: int) -> np.ndarray:
    """Calculate historical h-step naive errors within training data only."""

    values = np.asarray(values, dtype=float)
    if len(values) <= horizon:
        return np.array([], dtype=float)
    return values[horizon:] - values[:-horizon]


def historical_drift_errors(values: np.ndarray, horizon: int) -> np.ndarray:
    """Calculate rolling h-step drift errors using only prior prefixes."""

    values = np.asarray(values, dtype=float)
    errors = []
    for target_index in range(horizon + 1, len(values)):
        origin_index = target_index - horizon
        history = values[: origin_index + 1]
        if len(history) < 2:
            continue
        forecast = history[-1] + horizon * (history[-1] - history[0]) / (len(history) - 1)
        errors.append(values[target_index] - forecast)
    return np.asarray(errors, dtype=float)
