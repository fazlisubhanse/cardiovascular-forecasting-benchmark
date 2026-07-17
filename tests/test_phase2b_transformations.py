import numpy as np
import pandas as pd
from src.features.supervised import build_direct_dataset
from src.features.transformations import fit_linear_trend


def test_difference_inversion_is_cumulative_change():
    years=pd.Series(range(1970,2001)); values=pd.Series(np.arange(31,dtype=float)**1.1)
    ds=build_direct_dataset(values,years,1999,2,3,"first_difference_direct")
    assert ds.invert(2.5)==values.iloc[29]+2.5


def test_linear_trend_fit_is_fold_local_and_exact_for_line():
    years=np.arange(1970,2000); values=1.2+0.03*years
    intercept,slope=fit_linear_trend(years,values)
    assert np.isclose(intercept,1.2) and np.isclose(slope,.03)
