"""Expanding-window execution engine for Phase 2A baseline forecasts."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.forecasting.base import Forecaster

from .records import ForecastRecord

LOGGER = logging.getLogger(__name__)


def _interval_valid(lower: float, point: float, upper: float) -> bool:
    return bool(np.isfinite([lower, point, upper]).all() and lower <= point <= upper)


def run_backtest(
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    models: Iterable[Forecaster],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run each model once per target/origin and retain every horizon record."""

    year_column = config["data"]["year_column"]
    seed = int(config["runtime"]["random_seed"])
    primary = manifest.loc[manifest["analysis_window"] == "primary_2000_2023"].copy()
    records: list[dict[str, Any]] = []
    arima_candidates: list[dict[str, Any]] = []
    ets_candidates: list[dict[str, Any]] = []
    model_list = list(models)
    for target in config["data"]["targets"]:
        target_manifest = primary.loc[primary["target"] == target]
        for origin_year, origin_rows in target_manifest.groupby("origin_year", sort=True):
            horizons = sorted(origin_rows["horizon"].astype(int).unique().tolist())
            training = data.loc[data[year_column] <= origin_year, [year_column, target]]
            years = training[year_column].to_numpy(dtype=int)
            values = training[target].to_numpy(dtype=float)
            naive_errors = np.diff(values)
            naive_mae = float(np.mean(np.abs(naive_errors)))
            naive_mse = float(np.mean(np.square(naive_errors)))
            if not np.isfinite(naive_mae) or naive_mae <= 0 or not np.isfinite(naive_mse) or naive_mse <= 0:
                raise ValueError(f"Invalid in-sample scaling denominator for {target} at origin {origin_year}.")
            context = {
                "target": target,
                "analysis_window": "primary_2000_2023",
                "origin_year": int(origin_year),
            }
            for model in model_list:
                batch = model.forecast_many(years, values, horizons, seed, context)
                candidate_destination = arima_candidates if model.name == "ARIMA" else ets_candidates if model.name == "ETS" else None
                if candidate_destination is not None:
                    for candidate in batch.candidates:
                        candidate = dict(candidate)
                        candidate["target_year"] = int(origin_year) + int(candidate["horizon"])
                        candidate_destination.append(candidate)
                for _, manifest_row in origin_rows.iterrows():
                    horizon = int(manifest_row["horizon"])
                    output = batch.outputs.get(horizon)
                    if output is None:
                        raise RuntimeError(f"{model.name} omitted horizon {horizon} at {target}/{origin_year}.")
                    actual_row = data.loc[data[year_column] == int(manifest_row["target_year"]), target]
                    if len(actual_row) != 1:
                        raise RuntimeError("Target lookup did not resolve to exactly one observed value.")
                    valid80 = _interval_valid(output.lower_80, output.point, output.upper_80)
                    valid95 = _interval_valid(output.lower_95, output.point, output.upper_95)
                    warning_text = output.warning
                    if output.status == "success" and not valid80:
                        warning_text = " | ".join(filter(None, [warning_text, "80% interval missing or invalid."]))
                    if output.status == "success" and not valid95:
                        warning_text = " | ".join(filter(None, [warning_text, "95% interval missing or invalid."]))
                    total_seconds = output.selection_seconds + output.fit_seconds + output.forecast_seconds
                    record = ForecastRecord(
                        analysis_window="primary_2000_2023",
                        target=target,
                        model=model.name,
                        model_family=model.family,
                        horizon=horizon,
                        origin_year=int(origin_year),
                        target_year=int(manifest_row["target_year"]),
                        training_start_year=int(years[0]),
                        training_end_year=int(years[-1]),
                        training_n=len(years),
                        actual=float(actual_row.iloc[0]),
                        point_forecast=float(output.point),
                        lower_80=float(output.lower_80),
                        upper_80=float(output.upper_80),
                        lower_95=float(output.lower_95),
                        upper_95=float(output.upper_95),
                        fit_status=output.status,
                        fit_warning=warning_text,
                        selected_specification=output.specification,
                        selection_criterion=output.criterion,
                        selection_score=float(output.score),
                        selection_seconds=float(output.selection_seconds),
                        fit_seconds=float(output.fit_seconds),
                        forecast_seconds=float(output.forecast_seconds),
                        total_seconds=float(total_seconds),
                        seed=seed,
                        last_observed_value=float(values[-1]),
                        in_sample_naive_mae=naive_mae,
                        in_sample_naive_mse=naive_mse,
                        interval_80_valid=valid80,
                        interval_95_valid=valid95,
                    )
                    records.append(record.to_dict())
            LOGGER.info("Completed %s origin %d (%d horizons)", target, origin_year, len(horizons))
    primary_forecasts = pd.DataFrame(records)
    recent = primary_forecasts.loc[
        primary_forecasts["target_year"] >= int(config["backtest"]["recent_window_start_year"])
    ].copy()
    recent["analysis_window"] = "recent_2016_2023"
    forecasts = pd.concat([primary_forecasts, recent], ignore_index=True)
    forecasts = forecasts.sort_values(["analysis_window", "target", "model", "horizon", "origin_year"]).reset_index(drop=True)

    arima = _candidate_windows(pd.DataFrame(arima_candidates), config)
    ets = _candidate_windows(pd.DataFrame(ets_candidates), config)
    return forecasts, arima, ets


def _candidate_windows(candidates: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    recent = candidates.loc[candidates["target_year"] >= int(config["backtest"]["recent_window_start_year"])].copy()
    recent["analysis_window"] = "recent_2016_2023"
    return pd.concat([candidates, recent], ignore_index=True).sort_values(
        ["analysis_window", "target", "horizon", "origin_year"]
    ).reset_index(drop=True)


def validate_forecast_records(
    forecasts: pd.DataFrame, manifest: pd.DataFrame, model_names: list[str]
) -> None:
    """Validate temporal integrity, uniqueness, completeness, and schema before aggregation."""

    required = set(ForecastRecord.__dataclass_fields__)
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"Forecast records are missing fields: {sorted(missing)}")
    temporal_invalid = forecasts.loc[
        (forecasts["training_end_year"] != forecasts["origin_year"])
        | (forecasts["target_year"] != forecasts["origin_year"] + forecasts["horizon"])
        | (forecasts["target_year"] <= forecasts["training_end_year"])
    ]
    if not temporal_invalid.empty:
        raise ValueError("Forecast records contain future leakage/temporal-order violations.")
    key = ["analysis_window", "target", "model", "horizon", "origin_year"]
    if forecasts.duplicated(key).any():
        raise ValueError("Duplicate forecast records detected.")
    expected_total = len(manifest) * len(model_names)
    if len(forecasts) != expected_total:
        raise ValueError(f"Forecast record count {len(forecasts)} does not equal expected {expected_total}.")
    expected = manifest.groupby(["analysis_window", "target", "horizon"]).size().rename("expected")
    observed = forecasts.groupby(["analysis_window", "target", "horizon", "model"]).size().rename("observed")
    for (window, target, horizon, model), count in observed.items():
        if count != expected.loc[(window, target, horizon)]:
            raise ValueError(f"Incomplete stored records for {window}/{target}/h={horizon}/{model}.")
    if set(forecasts["model"]) != set(model_names):
        raise ValueError("Forecast records do not contain exactly the configured model names.")
