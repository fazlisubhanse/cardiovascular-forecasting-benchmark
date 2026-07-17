"""Strict Phase 2C.0 pilot configuration validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config


def load_phase2c_config(path: Path) -> dict[str,Any]:
    config=load_config(path)
    required={"data","development","evaluation","representations","lookbacks","training","selection","architectures","runtime"}
    if set(config)!=required: raise ValueError(f"Phase 2C sections must be exactly {sorted(required)}")
    if config["data"].get("targets")!=["asph","mtc"] or config["data"].get("year_column")!="year": raise ValueError("Invalid data targets/year column.")
    if len(str(config["data"].get("expected_sha256","")))!=64: raise ValueError("Validated-data SHA-256 required.")
    development=config["development"]
    expected_development={"final_target_year":1999,"strategy":"expanding","validation_origins":8,
        "minimum_training_years":18,"minimum_supervised_samples":12,"tuning_seeds":[1201,1202,1203],"search_strategy":"staged_factorial"}
    if development!=expected_development: raise ValueError("Development settings differ from the approved pilot boundary.")
    evaluation=config["evaluation"]
    if evaluation!={"initial_train_end_year":1999,"final_year":2023,"horizons":[1,2,5],
                   "recent_window_start_year":2016,"strategy":"expanding"}: raise ValueError("Invalid evaluation settings.")
    if config["representations"]!=["first_difference_direct","linear_detrend_direct"]: raise ValueError("Raw levels are excluded from neural primary evaluation.")
    if config["lookbacks"]!=[3,5,6,8]: raise ValueError("Invalid lookbacks.")
    training=config["training"]
    expected={"pilot_seeds":[2201,2202,2203],"optimizer":"adam","learning_rates":[0.001,0.0003],
        "weight_decay":[0.0,0.0001],"dropout":[0.0,0.1],"maximum_epochs":200,
        "early_stopping_patience":20,"minimum_epochs":10,"batch_size":8,"loss":"mse",
        "gradient_clip_norm":1.0,"device":"cpu","deterministic_algorithms":True,"torch_num_threads":1}
    if training!=expected: raise ValueError("Training settings differ from the approved deterministic CPU pilot.")
    if config["selection"]!={"primary_metric":"rmse","secondary_metric":"mae",
        "tie_breakers":["lower_parameter_count","smaller_lookback","lower_dropout","lexical_configuration_id"]}: raise ValueError("Invalid selection rules.")
    if config["architectures"].get("primary")!=["LSTM","GRU","BiLSTM","CausalCNN","CNN_LSTM","CNN_GRU","CNN_BiLSTM_Attention_Compact"]: raise ValueError("Invalid primary architecture set.")
    if config["architectures"].get("diagnostic")!=["CNN_BiLSTM_Compact_NoAttention","CNN_BiLSTM_Attention_Original23K"]: raise ValueError("Invalid diagnostic architecture set.")
    runtime=config["runtime"]
    if runtime.get("overwrite_phase2c_pilot_outputs") is not True or not isinstance(runtime.get("development_workers"),int) or runtime["development_workers"]<1: raise ValueError("Invalid runtime controls.")
    return config
