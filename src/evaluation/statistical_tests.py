"""HLN-corrected Diebold-Mariano tests and paired bootstrap effect intervals."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Apply Holm's step-down familywise-error adjustment."""

    count = len(p_values)
    if count == 0:
        return []
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted_sorted[rank] = running
    adjusted = np.empty(count, dtype=float)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted.tolist()


def _dm_hln(loss_differential: np.ndarray, horizon: int) -> tuple[float, float, str]:
    """Return HLN-corrected DM statistic, two-sided p value, and reason."""

    differential = np.asarray(loss_differential, dtype=float)
    count = len(differential)
    if count < 3:
        return np.nan, np.nan, "fewer than three paired losses"
    centered = differential - np.mean(differential)
    gamma0 = float(np.dot(centered, centered) / count)
    long_run_variance = gamma0
    for lag in range(1, horizon):
        if lag >= count:
            break
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / count)
        long_run_variance += 2.0 * gamma
    if not np.isfinite(long_run_variance) or long_run_variance <= np.finfo(float).eps:
        return np.nan, np.nan, "nonpositive or numerically invalid long-run variance estimate"
    variance_mean = long_run_variance / count
    dm = float(np.mean(differential) / np.sqrt(variance_mean))
    correction_term = (count + 1 - 2 * horizon + (horizon * (horizon - 1) / count)) / count
    if not np.isfinite(correction_term) or correction_term <= 0:
        return np.nan, np.nan, "invalid Harvey-Leybourne-Newbold correction term"
    corrected = dm * np.sqrt(correction_term)
    p_value = float(2.0 * student_t.sf(abs(corrected), df=count - 1))
    return float(corrected), p_value, ""


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    increment = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
    return (base_seed + increment) % (2**32)


def _bootstrap_indices(rng: np.random.Generator, count: int, repetitions: int, block_length: int) -> np.ndarray:
    """Generate iid or circular moving-block paired indices."""

    if block_length <= 1:
        return rng.integers(0, count, size=(repetitions, count))
    output = np.empty((repetitions, count), dtype=int)
    blocks_needed = int(np.ceil(count / block_length))
    offsets = np.arange(block_length)
    for repetition in range(repetitions):
        starts = rng.integers(0, count, size=blocks_needed)
        output[repetition] = np.concatenate([(start + offsets) % count for start in starts])[:count]
    return output


def compare_models(
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every complete model with the lowest-RMSE model per window/target/horizon."""

    alpha = float(config["statistics"]["alpha"])
    repetitions = int(config["statistics"]["bootstrap_repetitions"])
    base_seed = int(config["statistics"]["bootstrap_seed"])
    dm_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    grouping = ["analysis_window", "target", "horizon"]
    for keys, metric_group in metrics.groupby(grouping, sort=True):
        eligible = metric_group.loc[metric_group["ranking_eligible"].astype(bool) & np.isfinite(metric_group["rmse"])]
        if eligible.empty:
            continue
        best_row = eligible.sort_values(["rmse", "model"]).iloc[0]
        best_model = str(best_row["model"])
        window, target, horizon = keys
        group_forecasts = forecasts.loc[
            (forecasts["analysis_window"] == window)
            & (forecasts["target"] == target)
            & (forecasts["horizon"] == horizon)
            & forecasts["fit_status"].eq("success")
        ]
        best = group_forecasts.loc[group_forecasts["model"] == best_model, ["origin_year", "actual", "point_forecast"]].rename(
            columns={"point_forecast": "forecast_best"}
        )
        group_dm_indices = []
        raw_p_values = []
        for _, alternative_metric in eligible.loc[eligible["model"] != best_model].iterrows():
            alternative = str(alternative_metric["model"])
            other = group_forecasts.loc[group_forecasts["model"] == alternative, ["origin_year", "actual", "point_forecast"]].rename(
                columns={"actual": "actual_alt", "point_forecast": "forecast_alt"}
            )
            paired = best.merge(other, on="origin_year", how="inner")
            if not np.allclose(paired["actual"], paired["actual_alt"], rtol=0, atol=0):
                raise ValueError("Paired model comparison has inconsistent actual values.")
            actual = paired["actual"].to_numpy(dtype=float)
            error_best = actual - paired["forecast_best"].to_numpy(dtype=float)
            error_alt = actual - paired["forecast_alt"].to_numpy(dtype=float)
            differential = np.square(error_alt) - np.square(error_best)
            statistic, raw_p, reason = _dm_hln(differential, int(horizon))
            row = {
                "analysis_window": window, "target": target, "horizon": int(horizon),
                "reference_best_rmse_model": best_model, "comparison_model": alternative,
                "paired_n": len(paired), "autocovariance_truncation_lag": int(horizon) - 1,
                "dm_hln_statistic": statistic, "raw_p_value": raw_p, "holm_adjusted_p_value": np.nan,
                "mean_loss_differential_alt_minus_best": float(np.mean(differential)),
                "median_loss_differential_alt_minus_best": float(np.median(differential)),
                "rmse_difference_alt_minus_best": float(alternative_metric["rmse"] - best_row["rmse"]),
                "mae_difference_alt_minus_best": float(alternative_metric["mae"] - best_row["mae"]),
                "raw_rmse_winner": best_model, "statistically_detected_difference": False,
                "interpretation": "test_undefined" if not np.isfinite(raw_p) else "pending_holm_adjustment",
                "undefined_reason": reason,
                "caution": "Non-rejection is not evidence of equivalence; paired annual samples are small.",
            }
            dm_rows.append(row)
            if np.isfinite(raw_p):
                group_dm_indices.append(len(dm_rows) - 1)
                raw_p_values.append(float(raw_p))

            rng = np.random.default_rng(_stable_seed(base_seed, window, target, horizon, alternative))
            block_length = 1 if int(horizon) == 1 else max(int(horizon), int(np.ceil(np.sqrt(len(paired)))))
            indices = _bootstrap_indices(rng, len(paired), repetitions, block_length)
            best_samples = error_best[indices]
            alt_samples = error_alt[indices]
            rmse_differences = np.sqrt(np.mean(np.square(alt_samples), axis=1)) - np.sqrt(
                np.mean(np.square(best_samples), axis=1)
            )
            mae_differences = np.mean(np.abs(alt_samples), axis=1) - np.mean(np.abs(best_samples), axis=1)
            bootstrap_rows.append(
                {
                    "analysis_window": window, "target": target, "horizon": int(horizon),
                    "reference_best_rmse_model": best_model, "comparison_model": alternative,
                    "paired_n": len(paired), "bootstrap_repetitions": repetitions,
                    "bootstrap_type": "iid paired" if block_length == 1 else "circular moving-block paired",
                    "block_length": block_length,
                    "rmse_difference_alt_minus_best": float(alternative_metric["rmse"] - best_row["rmse"]),
                    "rmse_difference_ci_lower_95": float(np.quantile(rmse_differences, 0.025)),
                    "rmse_difference_ci_upper_95": float(np.quantile(rmse_differences, 0.975)),
                    "mae_difference_alt_minus_best": float(alternative_metric["mae"] - best_row["mae"]),
                    "mae_difference_ci_lower_95": float(np.quantile(mae_differences, 0.025)),
                    "mae_difference_ci_upper_95": float(np.quantile(mae_differences, 0.975)),
                    "seed": _stable_seed(base_seed, window, target, horizon, alternative),
                }
            )
        adjusted = _holm_adjust(raw_p_values)
        for index, adjusted_p in zip(group_dm_indices, adjusted):
            dm_rows[index]["holm_adjusted_p_value"] = adjusted_p
            detected = bool(adjusted_p < alpha)
            dm_rows[index]["statistically_detected_difference"] = detected
            dm_rows[index]["interpretation"] = (
                "statistically_detected_difference" if detected else "no_detected_difference_with_limited_power"
            )
    return pd.DataFrame(dm_rows), pd.DataFrame(bootstrap_rows)
