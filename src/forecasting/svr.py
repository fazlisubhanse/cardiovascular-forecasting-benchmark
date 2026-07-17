"""Deterministic support-vector regression factory."""

from sklearn.svm import SVR


def build_svr(params: dict):
    return SVR(**params)
