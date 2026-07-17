"""Deterministic CPU XGBoost regression factory."""

from xgboost import XGBRegressor


def build_xgboost(params: dict, seed: int, n_jobs: int):
    return XGBRegressor(**params, random_state=seed, n_jobs=n_jobs, verbosity=0)
