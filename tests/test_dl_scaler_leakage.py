import pandas as pd
from src.deep_learning.datasets import build_scaled_fold


def test_dl_scalers_exclude_outer_target():
    years=pd.Series(range(1970,2005)); data=pd.DataFrame({"year":years,"x":range(35)})
    changed=data.copy(); changed.loc[changed.year>1999,"x"]=99999
    a=build_scaled_fold(data,"x",1999,2,3,"first_difference_direct")
    b=build_scaled_fold(changed,"x",1999,2,3,"first_difference_direct")
    assert (a.x_scaler.mean_==b.x_scaler.mean_).all() and (a.y_scaler.mean_==b.y_scaler.mean_).all()
