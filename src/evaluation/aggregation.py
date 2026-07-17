"""Completeness-aware aggregation of individual forecast records."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import directional_accuracy, mase, mean_absolute_error, rmsse, root_mean_squared_error, smape

GROUP_COLUMNS = ["analysis_window", "target", "model", "horizon"]


def aggregate_point_metrics(forecasts: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate primary and diagnostic metrics with expected/success/failure counts."""

    expected_lookup = manifest.groupby(["analysis_window", "target", "horizon"]).size().to_dict()
    primary_rows = []
    diagnostic_rows = []
    for keys, group in forecasts.groupby(GROUP_COLUMNS, sort=True):
        window, target, model, horizon = keys
        expected_n = int(expected_lookup[(window, target, horizon)])
        success = group.loc[
            group["fit_status"].eq("success")
            & np.isfinite(group["point_forecast"].to_numpy(dtype=float))
        ].copy()
        successful_n = len(success)
        failed_n = expected_n - successful_n
        completeness = 100.0 * successful_n / expected_n if expected_n else np.nan
        counts = {
            "analysis_window": window, "target": target, "model": model, "horizon": int(horizon),
            "expected_n": expected_n, "successful_n": successful_n, "failed_n": failed_n,
            "completeness_percent": completeness, "ranking_eligible": bool(failed_n == 0),
        }
        if successful_n:
            actual = success["actual"].to_numpy(dtype=float)
            forecast = success["point_forecast"].to_numpy(dtype=float)
            error = actual - forecast
            primary = {
                **counts,
                "rmse": root_mean_squared_error(actual, forecast),
                "mae": mean_absolute_error(actual, forecast),
                "mase": mase(error, success["in_sample_naive_mae"].to_numpy(dtype=float)),
                "rmsse": rmsse(error, success["in_sample_naive_mse"].to_numpy(dtype=float)),
                "smape_percent": smape(actual, forecast),
                "mean_error": float(np.mean(error)),
                "median_error": float(np.median(error)),
                "directional_accuracy_percent": directional_accuracy(
                    actual, forecast, success["last_observed_value"].to_numpy(dtype=float)
                ),
            }
            denominator = float(np.sum(actual))
            diagnostics = {
                **counts,
                "bias_ratio_forecast_sum_over_actual_sum": float(np.sum(forecast) / denominator) if denominator != 0 else np.nan,
                "error_standard_deviation": float(np.std(error, ddof=1)) if successful_n > 1 else np.nan,
                "median_absolute_error": float(np.median(np.abs(error))),
                "maximum_absolute_error": float(np.max(np.abs(error))),
                "mape_percent": float(100 * np.mean(np.abs(error / actual))) if np.all(actual != 0) else np.nan,
            }
        else:
            primary = {**counts, **{name: np.nan for name in (
                "rmse", "mae", "mase", "rmsse", "smape_percent", "mean_error", "median_error",
                "directional_accuracy_percent",
            )}}
            diagnostics = {**counts, **{name: np.nan for name in (
                "bias_ratio_forecast_sum_over_actual_sum", "error_standard_deviation",
                "median_absolute_error", "maximum_absolute_error", "mape_percent",
            )}}
        primary_rows.append(primary)
        diagnostic_rows.append(diagnostics)
    return pd.DataFrame(primary_rows), pd.DataFrame(diagnostic_rows)
