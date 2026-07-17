"""Deterministic candidate enumeration and tie-breaking."""

from __future__ import annotations

from itertools import product
from typing import Any


def candidate_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    keys = list(grid)
    values = [value if isinstance(value, list) else [value] for value in grid.values()]
    return [dict(zip(keys, combination)) for combination in product(*values)]


def complexity_key(model: str, params: dict[str, Any]) -> tuple:
    """Order lower-capacity configurations first for exact score ties."""

    if model == "SVR":
        return (float(params["C"]), float(params["epsilon"]), str(params["gamma"]))
    if model == "RandomForest":
        depth = float("inf") if params["max_depth"] is None else int(params["max_depth"])
        return (int(params["n_estimators"]), depth, -int(params["min_samples_leaf"]), str(params["max_features"]))
    return (int(params["n_estimators"]), int(params["max_depth"]), float(params["learning_rate"]),
            -float(params["min_child_weight"]), float(params["reg_lambda"]))
