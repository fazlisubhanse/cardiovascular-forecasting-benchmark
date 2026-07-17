from src.forecasting.random_forest import build_random_forest
from src.forecasting.xgboost_model import build_xgboost


def test_tree_models_receive_deterministic_runtime_controls():
    rf=build_random_forest({"n_estimators":2},123,1)
    xgb=build_xgboost({"n_estimators":2,"objective":"reg:squarederror","tree_method":"hist"},123,1)
    assert rf.random_state==123 and rf.n_jobs==1 and xgb.random_state==123 and xgb.n_jobs==1
