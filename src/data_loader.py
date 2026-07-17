"""Strict loading and lossless validation-copy creation for the raw CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_raw_data(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Load the source CSV and reject any unexpected schema."""

    if not path.is_file():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8", float_precision="round_trip")
    expected_columns = config["expected"]["raw_columns"]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(
            "Raw CSV columns do not exactly match the configured schema. "
            f"Observed={frame.columns.tolist()!r} expected={expected_columns!r}"
        )
    return frame


def standardized_data(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Return standardized names/order without altering source values."""

    mapping = config["expected"]["column_map"]
    standardized = raw.rename(columns=mapping).loc[:, ["sex", "year", "age_group", "asph", "mtc"]]
    return standardized.copy(deep=True)


def write_validated_copy(frame: pd.DataFrame, path: Path) -> None:
    """Write a round-trip-safe CSV and verify exact numerical preservation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.17g", lineterminator="\n")
    check = pd.read_csv(path, encoding="utf-8", float_precision="round_trip")
    if check.columns.tolist() != frame.columns.tolist():
        raise RuntimeError("Validated-copy column order changed during serialization.")
    for column in ("asph", "mtc"):
        if not np.array_equal(
            check[column].to_numpy(dtype=np.float64),
            frame[column].to_numpy(dtype=np.float64),
        ):
            raise RuntimeError(f"Validated-copy numerical values changed in column {column!r}.")
    if not check[["sex", "year", "age_group"]].equals(frame[["sex", "year", "age_group"]]):
        raise RuntimeError("Validated-copy identifier values changed during serialization.")
