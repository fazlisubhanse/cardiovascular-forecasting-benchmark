import pandas as pd
from src.features.supervised import build_direct_dataset


def test_neural_direct_target_alignment():
    years=pd.Series(range(1970,2001)); values=pd.Series(range(31),dtype=float)
    fold=build_direct_dataset(values,years,1999,5,3,"first_difference_direct")
    assert (fold.sample_target_years==fold.sample_end_years+5).all()
    assert fold.forecast_target_year==2004
