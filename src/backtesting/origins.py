"""Leakage-free expanding-window forecast-origin generation."""

from __future__ import annotations

from typing import Any

import pandas as pd


def generate_forecast_origin_manifest(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Generate primary and recent-window origin rows from observed years."""

    settings = config["backtest"]
    targets = config["data"]["targets"]
    year_column = config["data"]["year_column"]
    years = data[year_column].astype(int).tolist()
    if years != sorted(years) or len(years) != len(set(years)):
        raise ValueError("Validated data years must be unique and chronological.")
    available = set(years)
    initial = int(settings["initial_train_end_year"])
    final = int(settings["final_year"])
    if initial not in available or final not in available:
        raise ValueError("Configured initial or final year is absent from validated data.")
    primary_rows: list[dict[str, Any]] = []
    for target in targets:
        if target not in data.columns:
            raise ValueError(f"Configured target is absent: {target}")
        for horizon in settings["horizons"]:
            for origin in range(initial, final + 1):
                target_year = origin + int(horizon)
                if target_year not in available or target_year > final:
                    continue
                training_years = [year for year in years if year <= origin]
                primary_rows.append(
                    {
                        "analysis_window": "primary_2000_2023",
                        "target": target,
                        "horizon": int(horizon),
                        "origin_year": origin,
                        "target_year": target_year,
                        "training_start_year": min(training_years),
                        "training_end_year": max(training_years),
                        "training_n": len(training_years),
                        "target_available": True,
                    }
                )
    primary = pd.DataFrame(primary_rows)
    recent = primary.loc[primary["target_year"] >= int(settings["recent_window_start_year"])].copy()
    recent["analysis_window"] = "recent_2016_2023"
    manifest = pd.concat([primary, recent], ignore_index=True)
    if manifest.empty:
        raise ValueError("No valid forecast origins were generated.")
    invalid = manifest.loc[
        (manifest["training_end_year"] != manifest["origin_year"])
        | (manifest["target_year"] != manifest["origin_year"] + manifest["horizon"])
        | (manifest["target_year"] <= manifest["training_end_year"])
    ]
    if not invalid.empty:
        raise RuntimeError("Generated origin manifest violates temporal ordering.")
    return manifest.sort_values(["analysis_window", "target", "horizon", "origin_year"]).reset_index(drop=True)
