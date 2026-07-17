"""Direct-horizon supervised sample construction without future access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .transformations import apply_representation


@dataclass(frozen=True)
class DirectDataset:
    X: np.ndarray
    y: np.ndarray
    forecast_X: np.ndarray
    sample_end_years: np.ndarray
    sample_target_years: np.ndarray
    state: dict[str, float]
    last_observed: float
    forecast_target_year: int

    def invert(self, prediction: float) -> float:
        if "trend_intercept" in self.state:
            return float(self.state["trend_intercept"] + self.state["trend_slope"] * self.forecast_target_year + prediction)
        if self.state.get("difference", 0.0) == 1.0:
            return float(self.last_observed + prediction)
        return float(prediction)


def build_direct_dataset(series: pd.Series, years: pd.Series, training_end_year: int,
                         horizon: int, lookback: int, representation: str) -> DirectDataset:
    """Build samples whose labels and transformation fit end no later than the fold origin."""

    mask = years.to_numpy(dtype=int) <= int(training_end_year)
    fold_years = years.to_numpy(dtype=int)[mask]
    fold_values = series.to_numpy(dtype=float)[mask]
    modeled, state = apply_representation(fold_years, fold_values, representation)
    index_offset = 1 if representation == "first_difference_direct" else 0
    rows, labels, ends, targets = [], [], [], []
    # s is the original-level index of the sample end within this fold.
    first_s = lookback if representation == "first_difference_direct" else lookback - 1
    last_s = len(fold_values) - horizon - 1
    for s in range(first_s, last_s + 1):
        if representation == "first_difference_direct":
            x = modeled[s - lookback:s]
            label = fold_values[s + horizon] - fold_values[s]
        else:
            x = modeled[s - lookback + 1:s + 1]
            label = modeled[s + horizon]
        if len(x) == lookback and np.isfinite(x).all() and np.isfinite(label):
            rows.append(x); labels.append(label); ends.append(fold_years[s]); targets.append(fold_years[s + horizon])
    if representation == "first_difference_direct":
        forecast_x = modeled[-lookback:]
        state = {**state, "difference": 1.0}
    else:
        forecast_x = modeled[-lookback:]
    if len(forecast_x) != lookback:
        raise ValueError("Insufficient fold history for requested lookback.")
    return DirectDataset(
        X=np.asarray(rows, dtype=float).reshape(-1, lookback), y=np.asarray(labels, dtype=float),
        forecast_X=np.asarray(forecast_x, dtype=float).reshape(1, -1),
        sample_end_years=np.asarray(ends, dtype=int), sample_target_years=np.asarray(targets, dtype=int),
        state=state, last_observed=float(fold_values[-1]),
        forecast_target_year=int(training_end_year + horizon),
    )


def build_sample_manifest(data: pd.DataFrame, origin_manifest: pd.DataFrame, models: list[str],
                          representations: list[str], lookbacks: list[int]) -> pd.DataFrame:
    """Describe every candidate sample and explicitly flag temporal/lag invalidity."""

    rows: list[dict[str, Any]] = []
    primary = origin_manifest.loc[origin_manifest["analysis_window"] == "primary_2000_2023"]
    first_year = int(data["year"].min())
    for outer in primary.itertuples(index=False):
        for model in models:
            for rep in representations:
                for lookback in lookbacks:
                    for sample_end in range(first_year, int(outer.origin_year) + 1):
                        enough_lags = sample_end - first_year >= (lookback if rep == "first_difference_direct" else lookback - 1)
                        target_inside = sample_end + int(outer.horizon) <= int(outer.origin_year)
                        valid = enough_lags and target_inside
                        reason = "" if valid else ("insufficient_lag_history" if not enough_lags else "target_after_outer_origin")
                        rows.append({
                            "analysis_window": outer.analysis_window, "target": outer.target, "model": model,
                            "representation": rep, "outer_origin_year": int(outer.origin_year),
                            "forecast_horizon": int(outer.horizon), "lookback": int(lookback),
                            "sample_end_year": sample_end, "sample_target_year": sample_end + int(outer.horizon),
                            "sample_is_valid": valid, "invalid_reason": reason,
                        })
    return pd.DataFrame(rows)
