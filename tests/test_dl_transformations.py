import numpy as np
import pandas as pd
from src.features.supervised import build_direct_dataset


def test_neural_difference_and_detrend_inverse_reconstruction():
    years=pd.Series(range(1970,2001)); values=pd.Series(np.arange(31,dtype=float)**1.1)
    diff=build_direct_dataset(values,years,1999,2,3,"first_difference_direct")
    assert diff.invert(2.0)==values.iloc[29]+2.0
    trend=build_direct_dataset(pd.Series(2+0.5*np.arange(31)),years,1999,2,3,"linear_detrend_direct")
    assert np.isclose(trend.invert(0),2+0.5*31)
