"""Strict configuration loading owned exclusively by Phase 2B and later corrections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config


def load_phase2b_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate the Phase 2B classical-ML configuration."""

    config = load_config(path)
    required = {
        "data", "backtest", "representations", "lookbacks", "inner_validation",
        "models", "scaling", "conformal", "statistics", "runtime",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Phase 2B configuration is missing sections: {sorted(missing)}")
    data, backtest = config["data"], config["backtest"]
    if data.get("targets") != ["asph", "mtc"] or data.get("year_column") != "year":
        raise ValueError("Phase 2B targets must be [asph, mtc] with year_column=year.")
    if not isinstance(data.get("expected_sha256"), str) or len(data["expected_sha256"]) != 64:
        raise ValueError("Phase 2B requires the 64-character validated-data SHA-256.")
    expected_backtest = {
        "initial_train_end_year": 1999, "final_year": 2023,
        "horizons": [1, 2, 5], "recent_window_start_year": 2016,
        "strategy": "expanding", "refit_every_origin": True,
    }
    for key, value in expected_backtest.items():
        if backtest.get(key) != value:
            raise ValueError(f"Phase 2B backtest.{key} must equal {value!r}.")
    reps = config["representations"]
    if reps.get("primary") != ["first_difference_direct", "linear_detrend_direct"]:
        raise ValueError("Primary representations must be first difference and linear detrend.")
    if reps.get("diagnostic") != ["raw_level_direct"]:
        raise ValueError("raw_level_direct must be the sole diagnostic representation.")
    if config["lookbacks"] != [3, 5, 6, 8]:
        raise ValueError("Phase 2B lookbacks must be exactly [3, 5, 6, 8].")
    inner = config["inner_validation"]
    if inner != {"recent_origins": 8, "minimum_training_years": 20,
                 "minimum_supervised_samples": 12, "selection_metrics": ["rmse", "mae"]}:
        raise ValueError("Inner validation settings differ from the prespecified design.")
    models = config["models"]
    if set(models) != {"SVR", "RandomForest", "XGBoost"}:
        raise ValueError("Phase 2B models must be exactly SVR, RandomForest, and XGBoost.")
    expected_counts = {"SVR": 48, "RandomForest": 36, "XGBoost": 32}
    for model, expected_count in expected_counts.items():
        count = 1
        for value in models[model].values():
            count *= len(value) if isinstance(value, list) else 1
        if count != expected_count:
            raise ValueError(f"{model} grid must contain {expected_count} combinations, got {count}.")
    if config["scaling"] != {"SVR": {"features": True, "target": True},
                             "RandomForest": {"features": False, "target": False},
                             "XGBoost": {"features": False, "target": False}}:
        raise ValueError("Scaling is permitted only for SVR features and target.")
    conformal = config["conformal"]
    if conformal.get("minimum_residual_count") != 8 or conformal.get("nominal_levels") != [0.80, 0.95]:
        raise ValueError("Conformal settings require at least eight residuals and levels [0.80, 0.95].")
    stats, runtime = config["statistics"], config["runtime"]
    if stats.get("bootstrap_repetitions") != 5000 or stats.get("bootstrap_seed") != 20260810:
        raise ValueError("Phase 2B requires 5,000 bootstrap repetitions and seed 20260810.")
    if stats.get("alpha") != 0.05 or stats.get("multiple_testing_adjustment") != "holm":
        raise ValueError("Phase 2B statistics require alpha=.05 and Holm adjustment.")
    if runtime != {"random_seed": 20260810, "n_jobs": 1, "overwrite_phase2b_outputs": True}:
        raise ValueError("Phase 2B runtime must use seed 20260810, n_jobs=1, and overwrite enabled.")
    return config
