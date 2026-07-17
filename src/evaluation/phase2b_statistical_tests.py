"""Phase 2B-only fixed-reference paired comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .statistical_tests import _bootstrap_indices, _dm_hln, _holm_adjust, _stable_seed


def compare_models_to_fixed_references(
    forecasts: pd.DataFrame, metrics: pd.DataFrame, references: pd.DataFrame,
    comparison_models: list[str], config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare named alternatives with a prespecified reference in every result cell."""

    alpha = float(config["statistics"]["alpha"])
    repetitions = int(config["statistics"]["bootstrap_repetitions"])
    base_seed = int(config["statistics"]["bootstrap_seed"])
    dm_rows, bootstrap_rows = [], []
    for ref in references.itertuples(index=False):
        window, target, horizon = ref.analysis_window, ref.target, int(ref.horizon)
        reference = str(ref.reference_model)
        cell = forecasts.loc[(forecasts.analysis_window == window) & (forecasts.target == target)
                             & (forecasts.horizon == horizon) & forecasts.fit_status.eq("success")]
        best = cell.loc[cell.model == reference, ["origin_year", "actual", "point_forecast"]].rename(
            columns={"point_forecast": "forecast_reference"})
        cell_indices, p_values = [], []
        for alternative in comparison_models:
            other = cell.loc[cell.model == alternative, ["origin_year", "actual", "point_forecast"]].rename(
                columns={"actual": "actual_alt", "point_forecast": "forecast_alt"})
            paired = best.merge(other, on="origin_year", how="inner")
            if paired.empty:
                continue
            actual = paired.actual.to_numpy(dtype=float)
            if not np.allclose(actual, paired.actual_alt.to_numpy(dtype=float), rtol=0, atol=0):
                raise ValueError("Fixed-reference comparison has inconsistent actual values.")
            ref_error = actual - paired.forecast_reference.to_numpy(dtype=float)
            alt_error = actual - paired.forecast_alt.to_numpy(dtype=float)
            differential = np.square(alt_error) - np.square(ref_error)
            statistic, raw_p, reason = _dm_hln(differential, horizon)
            ref_metric = metrics.loc[(metrics.analysis_window == window) & (metrics.target == target)
                                     & (metrics.horizon == horizon) & (metrics.model == reference)].iloc[0]
            alt_metric = metrics.loc[(metrics.analysis_window == window) & (metrics.target == target)
                                     & (metrics.horizon == horizon) & (metrics.model == alternative)].iloc[0]
            dm_rows.append({
                "analysis_window": window, "target": target, "horizon": horizon,
                "reference_model": reference, "comparison_model": alternative, "paired_n": len(paired),
                "autocovariance_truncation_lag": horizon - 1, "dm_hln_statistic": statistic,
                "raw_p_value": raw_p, "holm_adjusted_p_value": np.nan,
                "mean_loss_differential_alt_minus_reference": float(np.mean(differential)),
                "rmse_difference_alt_minus_reference": float(alt_metric.rmse - ref_metric.rmse),
                "mae_difference_alt_minus_reference": float(alt_metric.mae - ref_metric.mae),
                "statistically_detected_difference": False,
                "interpretation": "test_undefined" if not np.isfinite(raw_p) else "pending_holm_adjustment",
                "undefined_reason": reason,
                "caution": "Non-rejection is not evidence of equivalence; paired annual samples are small.",
            })
            if np.isfinite(raw_p):
                cell_indices.append(len(dm_rows) - 1); p_values.append(float(raw_p))
            rng = np.random.default_rng(_stable_seed(base_seed, "fixed", window, target, horizon, alternative))
            block = 1 if horizon == 1 else max(horizon, int(np.ceil(np.sqrt(len(paired)))))
            indices = _bootstrap_indices(rng, len(paired), repetitions, block)
            ref_samples, alt_samples = ref_error[indices], alt_error[indices]
            rmse_diff = np.sqrt(np.mean(alt_samples**2, axis=1)) - np.sqrt(np.mean(ref_samples**2, axis=1))
            mae_diff = np.mean(np.abs(alt_samples), axis=1) - np.mean(np.abs(ref_samples), axis=1)
            bootstrap_rows.append({
                "analysis_window": window, "target": target, "horizon": horizon,
                "reference_model": reference, "comparison_model": alternative, "paired_n": len(paired),
                "bootstrap_repetitions": repetitions,
                "bootstrap_type": "iid paired" if block == 1 else "circular moving-block paired",
                "block_length": block,
                "rmse_difference_alt_minus_reference": float(alt_metric.rmse - ref_metric.rmse),
                "rmse_difference_ci_lower_95": float(np.quantile(rmse_diff, .025)),
                "rmse_difference_ci_upper_95": float(np.quantile(rmse_diff, .975)),
                "mae_difference_alt_minus_reference": float(alt_metric.mae - ref_metric.mae),
                "mae_difference_ci_lower_95": float(np.quantile(mae_diff, .025)),
                "mae_difference_ci_upper_95": float(np.quantile(mae_diff, .975)),
            })
        for index, adjusted in zip(cell_indices, _holm_adjust(p_values)):
            dm_rows[index]["holm_adjusted_p_value"] = adjusted
            detected = bool(adjusted < alpha)
            dm_rows[index]["statistically_detected_difference"] = detected
            dm_rows[index]["interpretation"] = (
                "statistically_detected_difference" if detected else "no_detected_difference_with_limited_power")
    return pd.DataFrame(dm_rows), pd.DataFrame(bootstrap_rows)
