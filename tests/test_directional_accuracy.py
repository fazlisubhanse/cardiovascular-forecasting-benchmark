"""Directional-accuracy tests including exact zero changes."""

import numpy as np

from src.evaluation.metrics import directional_accuracy


def test_positive_negative_and_zero_directions() -> None:
    last = np.array([10.0, 10.0, 10.0, 10.0])
    actual = np.array([12.0, 8.0, 10.0, 10.0])
    forecast = np.array([11.0, 9.0, 10.0, 11.0])
    assert directional_accuracy(actual, forecast, last) == 75.0
