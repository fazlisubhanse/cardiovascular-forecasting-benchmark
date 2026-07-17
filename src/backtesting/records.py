"""Typed long-format forecast record schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ForecastRecord:
    """One model/target/origin/horizon forecast and its audit metadata."""

    analysis_window: str
    target: str
    model: str
    model_family: str
    horizon: int
    origin_year: int
    target_year: int
    training_start_year: int
    training_end_year: int
    training_n: int
    actual: float
    point_forecast: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float
    fit_status: str
    fit_warning: str
    selected_specification: str
    selection_criterion: str
    selection_score: float
    selection_seconds: float
    fit_seconds: float
    forecast_seconds: float
    total_seconds: float
    seed: int
    last_observed_value: float
    in_sample_naive_mae: float
    in_sample_naive_mse: float
    interval_80_valid: bool
    interval_95_valid: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert the immutable record to a serializable mapping."""

        return asdict(self)
