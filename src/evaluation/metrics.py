"""Mathematically explicit original-scale point-forecast metrics."""

from __future__ import annotations

import numpy as np


def root_mean_squared_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Return RMSE across all paired forecast origins."""

    errors = np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float)
    return float(np.sqrt(np.mean(np.square(errors))))


def mean_absolute_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Return mean absolute forecast error."""

    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float))))


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Return SMAPE percentage, defining a zero/zero pair contribution as zero."""

    actual_values = np.asarray(actual, dtype=float)
    forecast_values = np.asarray(forecast, dtype=float)
    denominator = np.abs(actual_values) + np.abs(forecast_values)
    numerator = 2.0 * np.abs(actual_values - forecast_values)
    terms = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    return float(100.0 * np.mean(terms))


def mase(errors: np.ndarray, denominators: np.ndarray) -> float:
    """Aggregate origin-specific scaled absolute errors."""

    errors = np.asarray(errors, dtype=float)
    denominators = np.asarray(denominators, dtype=float)
    if np.any(~np.isfinite(denominators)) or np.any(denominators <= 0):
        return np.nan
    return float(np.mean(np.abs(errors) / denominators))


def rmsse(errors: np.ndarray, denominators: np.ndarray) -> float:
    """Aggregate origin-specific scaled squared errors and take the square root."""

    errors = np.asarray(errors, dtype=float)
    denominators = np.asarray(denominators, dtype=float)
    if np.any(~np.isfinite(denominators)) or np.any(denominators <= 0):
        return np.nan
    return float(np.sqrt(np.mean(np.square(errors) / denominators)))


def directional_accuracy(actual: np.ndarray, forecast: np.ndarray, last_observed: np.ndarray) -> float:
    """Return percent sign agreement; zero is a separate direction and matches only zero."""

    actual_direction = np.sign(np.asarray(actual, dtype=float) - np.asarray(last_observed, dtype=float))
    predicted_direction = np.sign(np.asarray(forecast, dtype=float) - np.asarray(last_observed, dtype=float))
    return float(100.0 * np.mean(actual_direction == predicted_direction))
