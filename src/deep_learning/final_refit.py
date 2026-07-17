"""Fixed-epoch, no-validation rolling outer refitting for three pilot seeds."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd

from .datasets import build_scaled_fold, scaler_audit_record
from .training import final_fixed_epoch_train


def run_outer_cell(payload):
    data,manifest,selected_row,config=payload; selected=dict(selected_row); target=selected["target"]
    horizon=int(selected["horizon"]); architecture=selected["architecture"]; spec=json.loads(selected["configuration_json"])
    primary=manifest.loc[manifest.analysis_window.eq("primary_2000_2023")]; lookup=data.set_index("year")
    records=[]; scalers=[]; ratios=[]; failures=[]
    for outer in primary.sort_values("origin_year").itertuples(index=False):
        try:
            fold=build_scaled_fold(data,target,int(outer.origin_year),horizon,int(spec["lookback"]),spec["representation"])
            scalers.append(scaler_audit_record(fold,stage="outer_final",target=target,horizon=horizon,
                architecture=architecture,representation=spec["representation"],lookback=spec["lookback"],
                configuration_id=selected["configuration_id"],origin_year=int(outer.origin_year),forecast_target_year=int(outer.target_year)))
            ratios.append({"target":target,"horizon":horizon,"architecture":architecture,"origin_year":int(outer.origin_year),
                "supervised_training_n":len(fold.direct.y),"parameter_count":int(selected["trainable_parameter_count"]),
                "samples_to_parameter_ratio":len(fold.direct.y)/int(selected["trainable_parameter_count"]),
                "parameters_per_training_sample":int(selected["trainable_parameter_count"])/len(fold.direct.y),
                "more_parameters_than_samples":int(selected["trainable_parameter_count"])>len(fold.direct.y)})
            history=data.loc[data.year<=outer.origin_year,target].to_numpy(float); actual=float(lookup.loc[outer.target_year,target])
            for seed in config["training"]["pilot_seeds"]:
                started=time.perf_counter()
                try:
                    result=final_fixed_epoch_train(fold,architecture,spec["size"],float(spec["dropout"]),
                        float(spec["learning_rate"]),float(spec["weight_decay"]),int(selected["selected_epoch_budget"]),
                        int(seed),config["training"]); transformed,prediction=fold.reconstruct(result.scaled_prediction)
                    status="success"; warning=""
                except Exception as exc:
                    transformed=prediction=np.nan; result=None; status="failed"; warning=f"{type(exc).__name__}: {exc}"
                    failures.append({"target":target,"architecture":architecture,"horizon":horizon,
                                     "origin_year":outer.origin_year,"seed":seed,"failure_reason":warning})
                records.append({"analysis_window":"primary_2000_2023","target":target,"architecture":architecture,
                    "model":architecture,"model_role":selected["model_role"],"horizon":horizon,"origin_year":int(outer.origin_year),
                    "target_year":int(outer.target_year),"training_start_year":int(outer.training_start_year),
                    "training_end_year":int(outer.origin_year),"training_n":int(outer.training_n),
                    "supervised_training_n":len(fold.direct.y),"representation":spec["representation"],"lookback":int(spec["lookback"]),
                    "configuration_json":selected["configuration_json"],"parameter_count":int(selected["trainable_parameter_count"]),
                    "selected_epoch_count":int(selected["selected_epoch_budget"]),"seed":int(seed),"actual":actual,
                    "predicted_transformed_target":transformed,"point_forecast":prediction,
                    "absolute_error":abs(actual-prediction) if np.isfinite(prediction) else np.nan,
                    "squared_error":(actual-prediction)**2 if np.isfinite(prediction) else np.nan,
                    "fit_status":status,"fit_warning":warning,"fit_seconds":result.fit_seconds if result else time.perf_counter()-started,
                    "forecast_seconds":result.forecast_seconds if result else np.nan,"total_seconds":time.perf_counter()-started,
                    "last_observed_value":float(history[-1]),"in_sample_naive_mae":float(np.mean(np.abs(np.diff(history)))),
                    "in_sample_naive_mse":float(np.mean(np.square(np.diff(history)))),
                    "validation_n":0,"early_stopping_used":False,"used_all_supervised_samples":True})
        except Exception as exc:
            for seed in config["training"]["pilot_seeds"]:
                failures.append({"target":target,"architecture":architecture,"horizon":horizon,
                                 "origin_year":outer.origin_year,"seed":seed,"failure_reason":f"{type(exc).__name__}: {exc}"})
                records.append({"analysis_window":"primary_2000_2023","target":target,"architecture":architecture,
                    "model":architecture,"model_role":selected["model_role"],"horizon":horizon,"origin_year":int(outer.origin_year),
                    "target_year":int(outer.target_year),"seed":int(seed),"actual":float(lookup.loc[outer.target_year,target]),
                    "point_forecast":np.nan,"fit_status":"failed","fit_warning":failures[-1]["failure_reason"]})
    return {"target":target,"horizon":horizon,"architecture":architecture,"records":records,
            "scalers":scalers,"ratios":ratios,"failures":failures}
