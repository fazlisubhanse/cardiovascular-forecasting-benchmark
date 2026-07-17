"""Tests that RMSE is aggregated across multiple origins."""

import numpy as np

from src.evaluation.metrics import root_mean_squared_error


def test_rmse_uses_all_forecast_errors() -> None:
    actual = np.array([2.0, 4.0, 7.0])
    forecast = np.array([1.0, 2.0, 4.0])
    result = root_mean_squared_error(actual, forecast)
    assert np.isclose(result, np.sqrt(14.0 / 3.0))
    assert not np.isclose(result, abs(actual[-1] - forecast[-1]))
