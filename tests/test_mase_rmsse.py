"""Manual MASE and RMSSE calculation tests."""

import numpy as np

from src.evaluation.metrics import mase, rmsse


def test_origin_specific_mase_and_rmsse() -> None:
    errors = np.array([1.0, -2.0, 3.0])
    mae_denominators = np.array([2.0, 4.0, 3.0])
    mse_denominators = np.array([4.0, 8.0, 9.0])
    assert np.isclose(mase(errors, mae_denominators), np.mean([0.5, 0.5, 1.0]))
    assert np.isclose(rmsse(errors, mse_denominators), np.sqrt(np.mean([0.25, 0.5, 1.0])))
