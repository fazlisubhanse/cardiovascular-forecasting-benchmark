import pandas as pd
from src.features.supervised import build_direct_dataset


def test_future_mutation_does_not_change_outer_features():
    years=pd.Series(range(1970,2005)); base=pd.Series(range(35),dtype=float)
    changed=base.copy(); changed.loc[years>1999]=99999
    a=build_direct_dataset(base,years,1999,2,5,"linear_detrend_direct")
    b=build_direct_dataset(changed,years,1999,2,5,"linear_detrend_direct")
    assert (a.X==b.X).all() and (a.y==b.y).all() and (a.forecast_X==b.forecast_X).all()
