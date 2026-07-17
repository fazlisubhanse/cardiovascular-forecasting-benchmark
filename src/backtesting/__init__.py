"""Expanding-window forecast-origin and record infrastructure."""

from .origins import generate_forecast_origin_manifest
from .records import ForecastRecord

__all__ = ["ForecastRecord", "generate_forecast_origin_manifest"]
