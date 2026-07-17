"""Configuration loading and validation for the public analysis workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration using UTF-8 and validate required assumptions."""

    if not path.is_file():
        raise FileNotFoundError(f"Audit configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Audit configuration must be a YAML mapping.")
    return config

def load_phase2a_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate the Phase 2A baseline configuration."""

    config = load_config(path)
    required_sections = {"data", "backtest", "intervals", "arima", "ets", "statistics", "runtime"}
    missing = required_sections - set(config)
    if missing:
        raise ValueError(f"Phase 2A configuration is missing sections: {sorted(missing)}")
    data = config["data"]
    backtest = config["backtest"]
    intervals = config["intervals"]
    arima = config["arima"]
    ets = config["ets"]
    statistics = config["statistics"]
    runtime = config["runtime"]
    if data.get("targets") != ["asph", "mtc"] or data.get("year_column") != "year":
        raise ValueError("Phase 2A data targets must be [asph, mtc] and year_column must be year.")
    if not isinstance(data.get("expected_sha256"), str) or len(data["expected_sha256"]) != 64:
        raise ValueError("A 64-character validated-data SHA-256 is required.")
    if backtest.get("strategy") != "expanding" or backtest.get("refit_every_origin") is not True:
        raise ValueError("Only expanding backtesting with refit_every_origin=true is supported.")
    if backtest.get("initial_train_end_year") != 1999 or backtest.get("final_year") != 2023:
        raise ValueError("Phase 2A requires initial_train_end_year=1999 and final_year=2023.")
    horizons = backtest.get("horizons")
    if horizons != [1, 2, 5] or any(not isinstance(value, int) or value <= 0 for value in horizons):
        raise ValueError("Supported horizons are exactly [1, 2, 5].")
    if backtest.get("recent_window_start_year") != 2016:
        raise ValueError("The recent-window target start year must be 2016.")
    if intervals.get("nominal_levels") != [0.80, 0.95] or intervals.get("minimum_residual_count", 0) < 2:
        raise ValueError("Intervals must use nominal levels [0.80, 0.95] and at least two residuals.")
    for name in ("p_values", "d_values", "q_values"):
        values = arima.get(name)
        if not isinstance(values, list) or not values or any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError(f"ARIMA {name} must be a nonempty list of nonnegative integers.")
    if arima.get("selection_criterion") != "AICc":
        raise ValueError("Only AICc ARIMA selection is supported.")
    if ets.get("candidate_trends") != [None, "add"] or ets.get("damped_options") != [False, True]:
        raise ValueError("ETS candidates must be trends [null, add] and damped options [false, true].")
    if ets.get("seasonal") is not None or ets.get("selection_criterion") != "AICc":
        raise ValueError("Annual Phase 2A ETS must be nonseasonal and selected by AICc.")
    if statistics.get("dm_loss") != "squared_error" or statistics.get("multiple_testing_adjustment") != "holm":
        raise ValueError("Only squared-error DM loss with Holm adjustment is supported.")
    if not 0 < float(statistics.get("alpha", 0)) < 1:
        raise ValueError("statistics.alpha must be between zero and one.")
    if int(statistics.get("bootstrap_repetitions", 0)) < 100:
        raise ValueError("At least 100 bootstrap repetitions are required.")
    if not isinstance(runtime.get("random_seed"), int) or runtime.get("overwrite_phase2a_outputs") is not True:
        raise ValueError("A deterministic integer seed and overwrite_phase2a_outputs=true are required.")
    return config
