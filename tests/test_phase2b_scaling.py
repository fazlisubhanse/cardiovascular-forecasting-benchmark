import numpy as np
from src.features.scaling import scale_train_and_forecast


def test_scalers_ignore_forecast_row_when_fitting():
    X=np.array([[1.,2.],[3.,4.],[5.,6.]]); y=np.array([2.,4.,6.]); future=np.array([[100.,200.]])
    _,_,_,xs,ys=scale_train_and_forecast(X,y,future)
    assert np.allclose(xs.mean_,[3,4]) and np.isclose(ys.mean_[0],4)
