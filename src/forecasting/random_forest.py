"""Deterministic random-forest regression factory."""

from sklearn.ensemble import RandomForestRegressor


def build_random_forest(params: dict, seed: int, n_jobs: int):
    return RandomForestRegressor(**params, random_state=seed, n_jobs=n_jobs)
