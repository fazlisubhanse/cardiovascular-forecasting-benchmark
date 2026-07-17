"""Train-only standardization with auditable parameters."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def scale_train_and_forecast(X: np.ndarray, y: np.ndarray, forecast_X: np.ndarray):
    """Fit both scalers on training samples only and return inverse metadata."""

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    Xs = x_scaler.fit_transform(X)
    ys = y_scaler.fit_transform(np.asarray(y).reshape(-1, 1)).ravel()
    return Xs, ys, x_scaler.transform(forecast_X), x_scaler, y_scaler
