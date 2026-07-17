"""Development-only neural configuration selection ending at target year 1999."""

from __future__ import annotations

import hashlib
import json
import time
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from .architectures import architecture_size_candidates, build_architecture
from .datasets import build_scaled_fold, scaler_audit_record
from .parameter_audit import parameter_counts
from .training import development_train


def development_origins(first_year: int, horizon: int, settings: dict[str,Any]) -> list[int]:
    # The common fold set starts only when the least-demanding primary design
    # (first differences, lookback 3) has the configured minimum samples.
    earliest=max(first_year+int(settings["minimum_training_years"])-1,
                 first_year+horizon+int(settings["minimum_supervised_samples"])+3-1)
    latest=int(settings["final_target_year"])-horizon
    return list(range(earliest,latest+1))[-int(settings["validation_origins"]):]


def candidate_configurations(architecture: str, config: dict[str,Any],
                             fixed_structure: dict[str,Any]|None=None) -> list[dict[str,Any]]:
    """Create structural-screen or optimizer-refinement configurations."""
    rows=[]
    if fixed_structure is None:
        combinations=product(config["representations"],config["lookbacks"],architecture_size_candidates(architecture),
                             [config["training"]["learning_rates"][0]],[config["training"]["weight_decay"][0]],
                             [config["training"]["dropout"][0]])
    else:
        combinations=product([fixed_structure["representation"]],[fixed_structure["lookback"]],[fixed_structure["size"]],
                             config["training"]["learning_rates"],config["training"]["weight_decay"],config["training"]["dropout"])
    for representation,lookback,size,learning_rate,weight_decay,dropout in combinations:
        specification={"architecture":architecture,"representation":representation,"lookback":int(lookback),
                       "size":size,"learning_rate":float(learning_rate),"weight_decay":float(weight_decay),"dropout":float(dropout)}
        payload=json.dumps(specification,sort_keys=True,separators=(",",":")); specification["configuration_json"]=payload
        specification["configuration_id"]=hashlib.sha256(payload.encode()).hexdigest()[:16]; rows.append(specification)
    return rows


def _validation_transformed_target(fold, actual: float) -> float:
    if fold.direct.state.get("difference",0)==1.0: return actual-fold.direct.last_observed
    if "trend_intercept" in fold.direct.state:
        return actual-(fold.direct.state["trend_intercept"]+fold.direct.state["trend_slope"]*fold.direct.forecast_target_year)
    raise ValueError("Unsupported neural representation.")


def evaluate_development_cell(payload):
    """Evaluate all candidates for one target/horizon/architecture cell."""
    data,config,target,horizon,architecture=payload; lookup=data.set_index("year"); first=int(data.year.min())
    origins=development_origins(first,horizon,config["development"]); candidates=[]; details=[]; scaler_rows=[]
    started=time.perf_counter()
    def evaluate_candidates(candidate_list, search_stage):
      nonlocal candidates,details,scaler_rows
      for candidate in candidate_list:
        model=build_architecture(architecture,candidate["size"],candidate["dropout"]); counts=parameter_counts(model)
        fold_success=0; predictions=[]; actuals=[]; epochs=[]; reasons=[]; fit_seconds=0.0
        for origin in origins:
            try:
                fold=build_scaled_fold(data,target,origin,horizon,candidate["lookback"],candidate["representation"])
                if len(fold.direct.y)<int(config["development"]["minimum_supervised_samples"]):
                    raise ValueError(f"only {len(fold.direct.y)} supervised samples")
                actual=float(lookup.loc[origin+horizon,target]); transformed=_validation_transformed_target(fold,actual)
                scaled=float(fold.y_scaler.transform([[transformed]])[0,0]); seed_predictions=[]; seed_epochs=[]; seed_ok=True
                scaler_rows.append(scaler_audit_record(fold,stage="development",target=target,horizon=horizon,
                    architecture=architecture,representation=candidate["representation"],lookback=candidate["lookback"],
                    configuration_id=candidate["configuration_id"],origin_year=origin,forecast_target_year=origin+horizon))
                for seed in config["development"]["tuning_seeds"]:
                    try:
                        result=development_train(fold,scaled,architecture,candidate["size"],candidate["dropout"],
                            candidate["learning_rate"],candidate["weight_decay"],int(seed),config["training"])
                        _,prediction=fold.reconstruct(result.scaled_prediction); seed_predictions.append(prediction)
                        seed_epochs.append(result.best_epoch); fit_seconds+=result.fit_seconds
                        details.append({"target":target,"horizon":horizon,"architecture":architecture,"search_stage":search_stage,
                            "configuration_id":candidate["configuration_id"],"validation_origin_year":origin,
                            "validation_target_year":origin+horizon,"seed":seed,"actual":actual,"point_forecast":prediction,
                            "best_epoch":result.best_epoch,"fit_status":"success","failure_reason":""})
                    except Exception as exc:
                        seed_ok=False; reasons.append(f"origin={origin},seed={seed}: {type(exc).__name__}: {exc}")
                        details.append({"target":target,"horizon":horizon,"architecture":architecture,"search_stage":search_stage,
                            "configuration_id":candidate["configuration_id"],"validation_origin_year":origin,
                            "validation_target_year":origin+horizon,"seed":seed,"actual":actual,"point_forecast":np.nan,
                            "best_epoch":np.nan,"fit_status":"failed","failure_reason":reasons[-1]})
                if seed_ok and len(seed_predictions)==len(config["development"]["tuning_seeds"]):
                    fold_success+=1; predictions.extend(seed_predictions); actuals.extend([actual]*len(seed_predictions)); epochs.extend(seed_epochs)
            except Exception as exc:
                reasons.append(f"origin={origin}: {type(exc).__name__}: {exc}")
        errors=np.asarray(actuals)-np.asarray(predictions); eligible=fold_success==len(origins)
        candidates.append({"target":target,"horizon":horizon,"architecture":architecture,"search_stage":search_stage,
            "model_role":"diagnostic_ablation" if architecture in config["architectures"]["diagnostic"] else "primary",
            "representation":candidate["representation"],"lookback":candidate["lookback"],
            "configuration_id":candidate["configuration_id"],"configuration_json":candidate["configuration_json"],
            **counts,"parameter_count":int(counts["trainable_parameter_count"]),
            "expected_folds":len(origins),"successful_folds":fold_success,"failed_folds":len(origins)-fold_success,
            "successful_seed_forecasts":len(predictions),"mean_rmse":float(np.sqrt(np.mean(errors**2))) if eligible else np.nan,
            "mean_mae":float(np.mean(np.abs(errors))) if eligible else np.nan,
            "mean_best_epoch":float(np.mean(epochs)) if eligible else np.nan,
            "median_best_epoch":float(np.median(epochs)) if eligible else np.nan,
            "development_max_target_year":max(origin+horizon for origin in origins),"selection_eligible":eligible,
            "selected":False,"rejection_reason":" | ".join(sorted(set(reasons))),"fit_seconds":fit_seconds})
    def tie(row):
        spec=json.loads(row.configuration_json)
        return (row.mean_rmse,row.mean_mae,row.trainable_parameter_count,row.lookback,spec["dropout"],row.configuration_id)
    evaluate_candidates(candidate_configurations(architecture,config),"structural_screen")
    structural=pd.DataFrame(candidates); eligible_structural=structural.loc[structural.selection_eligible.astype(bool)]
    if eligible_structural.empty: raise RuntimeError(f"No eligible structural candidate for {target}/{horizon}/{architecture}")
    structural_winner=eligible_structural.loc[min(eligible_structural.index,key=lambda index:tie(eligible_structural.loc[index]))]
    structure=json.loads(structural_winner.configuration_json)
    evaluate_candidates(candidate_configurations(architecture,config,structure),"optimizer_refinement")
    frame=pd.DataFrame(candidates); eligible=frame.loc[frame.search_stage.eq("optimizer_refinement") & frame.selection_eligible.astype(bool)].copy()
    if eligible.empty: raise RuntimeError(f"No eligible optimizer candidate for {target}/{horizon}/{architecture}")
    winner=min(eligible.index,key=lambda index:tie(eligible.loc[index])); frame.loc[winner,"selected"]=True
    selected=frame.loc[[winner]].copy(); epoch=int(np.clip(round(selected.iloc[0].median_best_epoch),
        int(config["training"]["minimum_epochs"]),int(config["training"]["maximum_epochs"])))
    selected["selected_epoch_budget"]=epoch; selected["development_search_seconds"]=time.perf_counter()-started
    return {"target":target,"horizon":horizon,"architecture":architecture,"candidates":frame.to_dict("records"),
            "selected":selected.to_dict("records"),"details":details,"scalers":scaler_rows}
