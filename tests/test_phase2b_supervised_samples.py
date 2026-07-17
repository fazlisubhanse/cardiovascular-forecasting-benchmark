import pandas as pd
from src.features.supervised import build_direct_dataset


def test_direct_samples_never_label_after_fold_origin():
    years=pd.Series(range(1970,2001)); values=pd.Series([float(x) for x in range(len(years))])
    ds=build_direct_dataset(values,years,1999,5,3,"raw_level_direct")
    assert ds.sample_target_years.max()<=1999
    assert ds.forecast_target_year==2004
