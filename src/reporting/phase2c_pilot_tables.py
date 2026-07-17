"""Phase 2C.0 comparison, ablation, and computational summary tables."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from ..deep_learning.architectures import architecture_size_candidates, build_architecture
from ..deep_learning.parameter_audit import parameter_counts


STATISTICAL={"Naive","Drift","LinearTrend","ETS","ARIMA","Theta"}
CLASSICAL_ML={"SVR","RandomForest","XGBoost"}


def architecture_parameter_table(config: dict) -> pd.DataFrame:
    rows=[]
    for role in ("primary","diagnostic"):
        for architecture in config["architectures"][role]:
            for index,size in enumerate(architecture_size_candidates(architecture),1):
                model=build_architecture(architecture,size,0.0); rows.append({"architecture":architecture,
                    "model_role":"primary" if role=="primary" else "diagnostic_ablation",
                    "size_candidate_index":index,"size_json":json.dumps(size,sort_keys=True),**parameter_counts(model)})
    return pd.DataFrame(rows)


def comparison_summary(metrics: pd.DataFrame, primary_neural: list[str]) -> pd.DataFrame:
    rows=[]; primary=metrics.loc[metrics.analysis_window.eq("primary_2000_2023")]
    for (target,horizon),group in primary.groupby(["target","horizon"],sort=True):
        stat=group.loc[group.model.isin(STATISTICAL)].nsmallest(1,"rmse").iloc[0]
        ml=group.loc[group.model.isin(CLASSICAL_ML)].nsmallest(1,"rmse").iloc[0]
        neural=group.loc[group.model.isin(primary_neural)].nsmallest(1,"rmse").iloc[0]
        overall=group.nsmallest(1,"rmse").iloc[0]
        nonneural=group.loc[~group.model.isin(primary_neural)].nsmallest(1,"rmse").iloc[0]
        rows.append({"target":target,"horizon":int(horizon),"best_statistical_model":stat.model,
            "best_statistical_rmse":stat.rmse,"best_classical_ml_model":ml.model,"best_classical_ml_rmse":ml.rmse,
            "best_neural_model":neural.model,"best_neural_rmse":neural.rmse,"best_overall_model":overall.model,
            "best_overall_rmse":overall.rmse,"best_nonneural_model":nonneural.model,"best_nonneural_rmse":nonneural.rmse,
            "neural_minus_nonneural_rmse":neural.rmse-nonneural.rmse,
            "neural_minus_nonneural_mae":neural.mae-nonneural.mae,
            "neural_rmse_percent_difference":100*(neural.rmse-nonneural.rmse)/nonneural.rmse})
    return pd.DataFrame(rows)


def architecture_ablation_table(metrics: pd.DataFrame, parameters: pd.DataFrame,
                                seed_forecasts: pd.DataFrame, ensemble: pd.DataFrame) -> pd.DataFrame:
    pairs=(("attention_operation","CNN_BiLSTM_Attention_Compact","CNN_BiLSTM_Compact_NoAttention"),
           ("capacity","CNN_BiLSTM_Attention_Compact","CNN_BiLSTM_Attention_Original23K"))
    primary=metrics.loc[metrics.analysis_window.eq("primary_2000_2023")]; rows=[]
    for label,compact,other in pairs:
        for (target,horizon),group in primary.groupby(["target","horizon"],sort=True):
            a=group.loc[group.model.eq(compact)].iloc[0]; b=group.loc[group.model.eq(other)].iloc[0]
            pa=int(parameters.loc[parameters.architecture.eq(compact),"trainable_parameter_count"].iloc[0])
            pb=int(parameters.loc[parameters.architecture.eq(other),"trainable_parameter_count"].iloc[0])
            sf=seed_forecasts.loc[(seed_forecasts.target==target)&(seed_forecasts.horizon==horizon)]
            ta=sf.loc[sf.architecture.eq(compact),"fit_seconds"].sum(); tb=sf.loc[sf.architecture.eq(other),"fit_seconds"].sum()
            ef=ensemble.loc[(ensemble.analysis_window=="primary_2000_2023")&(ensemble.target==target)&(ensemble.horizon==horizon)]
            va=ef.loc[ef.architecture.eq(compact),"seed_prediction_std"].mean(); vb=ef.loc[ef.architecture.eq(other),"seed_prediction_std"].mean()
            rows.append({"ablation":label,"target":target,"horizon":int(horizon),"reference_model":compact,
                "comparison_model":other,"rmse_comparison_minus_reference":b.rmse-a.rmse,
                "mae_comparison_minus_reference":b.mae-a.mae,"parameter_count_reference":pa,
                "parameter_count_comparison":pb,"parameter_count_ratio_comparison_over_reference":pb/pa,
                "fit_seconds_reference":ta,"fit_seconds_comparison":tb,"training_time_ratio":tb/ta if ta else np.nan,
                "mean_seed_std_reference":va,"mean_seed_std_comparison":vb})
    return pd.DataFrame(rows)


def computational_cost(candidate_search: pd.DataFrame, seed_forecasts: pd.DataFrame) -> pd.DataFrame:
    development=candidate_search.groupby("architecture").fit_seconds.sum().rename("development_search_fit_seconds")
    outer=seed_forecasts.groupby("architecture").agg(final_fit_seconds=("fit_seconds","sum"),
        forecast_seconds=("forecast_seconds","sum"),total_seed_level_seconds=("total_seconds","sum"),
        median_seconds_per_seed_origin=("total_seconds","median"),seed_forecast_n=("seed","size")).reset_index()
    return outer.merge(development,on="architecture",how="left").assign(
        memory_measurement="not collected; no additional memory dependency installed")
