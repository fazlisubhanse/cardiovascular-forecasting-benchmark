"""Fold-local direct-horizon tensors and scaling audit records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from ..features.supervised import DirectDataset, build_direct_dataset


@dataclass(frozen=True)
class ScaledFold:
    direct: DirectDataset
    train_X: torch.Tensor
    train_y: torch.Tensor
    forecast_X: torch.Tensor
    x_scaler: StandardScaler
    y_scaler: StandardScaler

    def inverse_transformed_target(self, value: float) -> float:
        return float(self.y_scaler.inverse_transform([[value]])[0, 0])

    def reconstruct(self, scaled_prediction: float) -> tuple[float, float]:
        transformed = self.inverse_transformed_target(scaled_prediction)
        return transformed, self.direct.invert(transformed)


def build_scaled_fold(data: pd.DataFrame, target: str, training_end_year: int, horizon: int,
                      lookback: int, representation: str) -> ScaledFold:
    """Fit feature and target scalers using supervised training rows only."""
    direct = build_direct_dataset(data[target], data["year"], training_end_year, horizon, lookback, representation)
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    X = x_scaler.fit_transform(direct.X)
    y = y_scaler.fit_transform(direct.y.reshape(-1, 1)).ravel()
    future = x_scaler.transform(direct.forecast_X)
    return ScaledFold(direct, torch.tensor(X[:, :, None], dtype=torch.float32),
                      torch.tensor(y[:, None], dtype=torch.float32),
                      torch.tensor(future[:, :, None], dtype=torch.float32), x_scaler, y_scaler)


def scaler_audit_record(fold: ScaledFold, **metadata: Any) -> dict[str, Any]:
    """Serialize fold-local scaler parameters and supervised target boundaries."""
    import json
    return {**metadata, "supervised_training_n": len(fold.direct.y),
            "latest_supervised_target_year": int(fold.direct.sample_target_years.max()),
            "x_mean_json": json.dumps(fold.x_scaler.mean_.tolist()),
            "x_scale_json": json.dumps(fold.x_scaler.scale_.tolist()),
            "y_mean": float(fold.y_scaler.mean_[0]), "y_scale": float(fold.y_scaler.scale_[0])}
