"""Prediction-interval validation and coverage/width/score aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winkler_interval_score(actual: float, lower: float, upper: float, alpha: float) -> float:
    """Calculate the Winkler interval score for a valid interval."""

    if not np.isfinite([actual, lower, upper]).all() or lower > upper or not 0 < alpha < 1:
        return np.nan
    score = upper - lower
    if actual < lower:
        score += (2.0 / alpha) * (lower - actual)
    elif actual > upper:
        score += (2.0 / alpha) * (actual - upper)
    return float(score)


def aggregate_interval_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid intervals; invalid/missing intervals are never counted covered."""

    rows = []
    group_columns = ["analysis_window", "target", "model", "horizon"]
    for keys, group in forecasts.groupby(group_columns, sort=True):
        for level, suffix in ((0.80, "80"), (0.95, "95")):
            lower_name, upper_name = f"lower_{suffix}", f"upper_{suffix}"
            valid_name = f"interval_{suffix}_valid"
            successful = group.loc[group["fit_status"].eq("success")].copy()
            valid = successful.loc[successful[valid_name].astype(bool)].copy()
            widths = valid[upper_name].to_numpy(dtype=float) - valid[lower_name].to_numpy(dtype=float)
            actual = valid["actual"].to_numpy(dtype=float)
            lower = valid[lower_name].to_numpy(dtype=float)
            upper = valid[upper_name].to_numpy(dtype=float)
            covered = (actual >= lower) & (actual <= upper)
            scores = np.array([
                winkler_interval_score(a, lo, hi, 1.0 - level) for a, lo, hi in zip(actual, lower, upper)
            ])
            rows.append(
                {
                    "analysis_window": keys[0], "target": keys[1], "model": keys[2], "horizon": int(keys[3]),
                    "nominal_level": level, "successful_forecasts": len(successful),
                    "valid_interval_n": len(valid), "missing_or_invalid_interval_n": len(successful) - len(valid),
                    "coverage_probability": float(np.mean(covered)) if len(valid) else np.nan,
                    "mean_interval_width": float(np.mean(widths)) if len(valid) else np.nan,
                    "median_interval_width": float(np.median(widths)) if len(valid) else np.nan,
                    "mean_interval_score": float(np.mean(scores)) if len(valid) else np.nan,
                    "interval_order_or_point_violations": int((~successful[valid_name].astype(bool)).sum()),
                }
            )
    return pd.DataFrame(rows)


def validate_intervals(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return record-level interval violations for failure reporting."""

    violations = []
    for _, row in forecasts.loc[forecasts["fit_status"].eq("success")].iterrows():
        for level, suffix in ((0.80, "80"), (0.95, "95")):
            if not bool(row[f"interval_{suffix}_valid"]):
                violations.append(
                    {"analysis_window": row["analysis_window"], "target": row["target"], "model": row["model"],
                     "horizon": row["horizon"], "origin_year": row["origin_year"], "nominal_level": level,
                     "reason": "interval missing, nonfinite, unordered, or does not contain point forecast"}
                )
    return pd.DataFrame(violations)
