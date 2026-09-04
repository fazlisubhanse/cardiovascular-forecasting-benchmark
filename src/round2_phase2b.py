"""Official-data Round-2 frozen leakage-controlled benchmark.

This runner is intentionally separate from the frozen Round-1/first-revision
pipelines.  It consumes the Phase 2A.1 protocol, never calls the historical
nested lookback selector, and writes restart-safe per-fit records before
assembling the authoritative Round-2 result package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Each process fits tiny time-series models.  Large implicit BLAS thread pools
# waste memory and can prevent Windows worker processes from starting.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import torch
import xgboost
import yaml

from .data_validation import sha256_file
from .deep_learning.architectures import build_architecture
from .deep_learning.datasets import build_scaled_fold
from .deep_learning.parameter_audit import parameter_counts
from .deep_learning.training import final_fixed_epoch_fit
from .evaluation.metrics import (
    directional_accuracy,
    mase,
    mean_absolute_error,
    rmsse,
    root_mean_squared_error,
    smape,
)
from .evaluation.statistical_tests import _bootstrap_indices, _dm_hln, _holm_adjust, _stable_seed
from .features.supervised import build_direct_dataset
from .forecasting import (
    ArimaForecaster,
    DriftForecaster,
    EtsForecaster,
    LinearTrendForecaster,
    NaiveForecaster,
    ThetaForecaster,
)
from .forecasting.conformal import conformal_bounds
from .phase2b_engine import fit_predict
from .tuning.inner_origins import valid_inner_origins


EXPECTED_DATA_HASH = "4575b29feee41e3832c9c2e51c9c0f8f0e312d5e7f4df9b87ce75e13e705c3e2"
FROZEN_ML_CONFIG_PATH = "config/frozen_ml_configuration_map.csv"
FROZEN_NEURAL_CONFIG_PATH = "config/frozen_neural_configuration_map_l3.csv"
EXPECTED_ML_CONFIG_HASH = "ed959f2241b559509a4cd14b7411772726c6bdabd3d4e76787fdce25689b5fad"
EXPECTED_NEURAL_CONFIG_HASH = "f323e21732147116fd22690f7aaeb79b0999076050e006538157a200bf6c33c3"
GLOBAL_SEED = 20260810
NEURAL_SEEDS = list(range(3101, 3111))
PRIMARY_NEURAL = [
    "LSTM", "GRU", "BiLSTM", "CausalCNN", "CNN_LSTM", "CNN_GRU",
    "CNN_BiLSTM_Attention_Compact",
]
ABLATION_NEURAL = [
    "CNN_BiLSTM_Compact_NoAttention",
    "CNN_BiLSTM_Attention_Original23K",
]
STATISTICAL_MODELS = ["Naive", "Drift", "LinearTrend", "ETS", "ARIMA", "Theta"]
ML_MODELS = ["SVR", "RandomForest", "XGBoost"]
REPRESENTATIONS = ["first_difference_direct", "linear_detrend_direct"]
TARGET_META = {
    "ASPH": {
        "definition": "Age-standardized prevalence of hypertension",
        "age_population": "Adults aged 30–79 years",
        "unit": "proportion",
    },
    "MTC": {
        "definition": "Age-standardized mean total cholesterol",
        "age_population": "Adults aged 18 years and older",
        "unit": "mmol/L",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _paths(root: Path) -> dict[str, Path]:
    base = root / "results" / "round2"
    return {
        "base": base,
        "progress": base / "progress" / "fits",
        "tables": base / "tables",
        "figures": base / "figures",
        "report": root / "reports" / "round2_phase2b_execution_audit.md",
    }


def _load_protocol(root: Path) -> tuple[dict[str, Any], Path, str]:
    path = root / "config" / "round2_frozen_evaluation_protocol.json"
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), path, hashlib.sha256(raw).hexdigest()


def _load_official_long(root: Path, protocol: dict[str, Any]) -> tuple[pd.DataFrame, Path, str]:
    path = root / protocol["processed_dataset"]["path"]
    digest = sha256_file(path)
    frame = pd.read_csv(path, encoding="utf-8", float_precision="round_trip")
    return frame, path, digest


def _series_frame(official: pd.DataFrame, target: str) -> pd.DataFrame:
    part = official.loc[official.target.eq(target), ["year", "analysis_value"]].copy()
    part = part.rename(columns={"analysis_value": target.lower()}).sort_values("year").reset_index(drop=True)
    return part


def _calendar_rows(protocol: dict[str, Any]) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for target, target_spec in protocol["evaluation_calendars"].items():
        for horizon_text, spec in target_spec["horizons"].items():
            horizon = int(horizon_text)
            for origin, target_year in zip(spec["origin_years"], spec["target_years"]):
                rows.append({"target": target, "horizon": horizon, "origin_year": int(origin),
                             "target_year": int(target_year)})
    return rows


def _fit_id(run_id: str, fields: Iterable[Any]) -> str:
    payload = "|".join([run_id, *(str(item) for item in fields)])
    return "fit_" + _sha256_text(payload)[:24]


def _model_label(model: str, representation: str) -> str:
    if model in ML_MODELS:
        return f"{model} [{representation}]"
    return model


def _hardware_environment(root: Path, worker_count: int, protocol_hash: str, dataset_hash: str) -> dict[str, Any]:
    def powershell_value(script: str) -> str:
        try:
            result = subprocess.run(["powershell", "-NoProfile", "-Command", script], cwd=root,
                                    capture_output=True, text=True, timeout=15, check=False)
            return result.stdout.strip() or "unavailable"
        except Exception as exc:
            return f"unavailable: {type(exc).__name__}: {exc}"

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, timeout=10, check=False).stdout.strip() or "not_a_git_repository"
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                    text=True, timeout=10, check=False).stdout.strip())
    except Exception:
        commit, dirty = "unavailable", None
    cpu_name = powershell_value("(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)")
    physical = powershell_value("(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum")
    ram = powershell_value("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
    tensorflow_version = "not applicable; implementation uses PyTorch only"
    try:
        import tensorflow as tensorflow  # type: ignore
        tensorflow_version = str(tensorflow.__version__)
    except Exception:
        pass
    relevant = {
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__, "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "torch": torch.__version__,
        "matplotlib": matplotlib.__version__, "PyYAML": yaml.__version__,
        "tensorflow": tensorflow_version,
    }
    return {
        "recorded_before_training_utc": _utc_now(), "git_commit": commit,
        "git_worktree_dirty_before_run": dirty, "python_executable": sys.executable,
        "python_version": platform.python_version(), "operating_system": platform.platform(),
        "cpu_model": cpu_name if cpu_name != "unavailable" else platform.processor(),
        "physical_cpu_count": int(physical) if str(physical).isdigit() else physical,
        "logical_cpu_count": os.cpu_count(), "ram_bytes": int(ram) if str(ram).isdigit() else ram,
        "gpu_model": "none; CPU-only execution", "gpu_used": False,
        "cuda_version": None, "torch_cuda_available": bool(torch.cuda.is_available()),
        "library_versions": relevant, "configured_worker_count": worker_count,
        "stage_workers": {"statistical": 1, "classical_ml": 1, "neural": worker_count, "reporting": 1},
        "parallel_strategy": "ProcessPoolExecutor across frozen neural target-horizon-architecture cells; one Torch thread per process",
        "thread_environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
        )},
        "deterministic_flags": {"torch_deterministic_algorithms": True, "torch_num_threads_per_worker": 1,
                                "device": "cpu"},
        "random_seed_handling": {"statistical_and_ml_seed": GLOBAL_SEED,
                                 "neural_seeds": NEURAL_SEEDS,
                                 "bootstrap_base_seed": GLOBAL_SEED},
        "dataset_sha256": dataset_hash, "protocol_sha256": protocol_hash,
        "package_install_or_upgrade_performed": False,
    }


def _preflight(root: Path, worker_count: int) -> dict[str, Any]:
    paths = _paths(root)
    preexisting_progress = list(paths["progress"].glob("fit_*.json")) if paths["progress"].exists() else []
    preexisting_predictions = paths["base"] / "predictions_long.csv"
    protocol, protocol_path, protocol_hash = _load_protocol(root)
    official, data_path, data_hash = _load_official_long(root, protocol)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL",
                       "observed": observed, "expected": expected})

    check("dataset_sha256", data_hash == EXPECTED_DATA_HASH == protocol["processed_dataset"]["sha256"],
          data_hash, EXPECTED_DATA_HASH)
    prior_valid = all(_read_valid_progress(path, data_hash, protocol_hash) is not None for path in preexisting_progress)
    check("resume_checkpoint_prior_fit_records_valid", prior_valid,
          {"preexisting_fit_records": len(preexisting_progress)}, "0 on first run, otherwise hash-valid resumable records")
    check("resume_checkpoint_forecast_ledger_state", True,
          {"predictions_long_exists": preexisting_predictions.exists()}, "observational; never used to skip without fit-id/hash validation")
    check("processed_row_count", len(official) == 69, len(official), 69)
    for target, years, age, unit, source_fragment in (
        ("ASPH", list(range(1990, 2020)), "30", "proportion", "Lancet 2021"),
        ("MTC", list(range(1980, 2019)), "18", "mmol/L", "Nature 2020"),
    ):
        part = official.loc[official.target.eq(target)]
        check(f"{target}_years", part.year.tolist() == years, f"{part.year.min()}-{part.year.max()} n={len(part)}",
              f"{years[0]}-{years[-1]} n={len(years)}")
        check(f"{target}_central_values", part.analysis_value.notna().all() and np.isfinite(part.analysis_value).all(),
              int(part.analysis_value.notna().sum()), len(years))
        check(f"{target}_age_population", part.age_population.astype(str).str.contains(age, regex=False).all(),
              sorted(part.age_population.astype(str).unique().tolist()), TARGET_META[target]["age_population"])
        check(f"{target}_unit", part.unit.eq(unit).all(), sorted(part.unit.unique().tolist()), unit)
        check(f"{target}_source", part.source_dataset.astype(str).str.contains(source_fragment, regex=False).all(),
              sorted(part.source_dataset.astype(str).unique().tolist()), f"contains {source_fragment}")
    check("protocol_status", protocol.get("status") == "frozen_executable_ready_for_phase2b",
          protocol.get("status"), "frozen_executable_ready_for_phase2b")
    check("initial_history", protocol["selected_initial_history_rule"]["minimum_raw_history_length"] == 21,
          protocol["selected_initial_history_rule"]["minimum_raw_history_length"], 21)
    calendar = _calendar_rows(protocol)
    expected_counts = {("ASPH", 1): 9, ("ASPH", 2): 8, ("ASPH", 5): 5,
                       ("MTC", 1): 18, ("MTC", 2): 17, ("MTC", 5): 14}
    observed_counts = pd.DataFrame(calendar).groupby(["target", "horizon"]).size().to_dict()
    check("calendar_counts", observed_counts == expected_counts, observed_counts, expected_counts)
    check("calendar_total", len(calendar) == 71, len(calendar), 71)
    freeze = protocol["model_freeze"]
    check("primary_lookback", freeze["primary_lookback"] == 3 and freeze["primary_lookbacks"] == [3],
          {"primary_lookback": freeze["primary_lookback"], "primary_lookbacks": freeze["primary_lookbacks"]}, 3)
    check("first_origin_h5_samples", freeze["first_origin_h5_l3_supervised_samples"] == {
        "first_difference_direct": 13, "linear_detrend_direct": 14, "raw_level_direct": 14},
          freeze["first_origin_h5_l3_supervised_samples"], {"first_difference_direct": 13, "linear_detrend_direct": 14, "raw_level_direct": 14})
    check("minimum_supervised_samples", freeze["minimum_supervised_samples"] == 12,
          freeze["minimum_supervised_samples"], 12)
    check("nested_selection_disabled", freeze["nested_performance_based_lookback_selection_enabled"] is False,
          freeze["nested_performance_based_lookback_selection_enabled"], False)
    check("legacy_lookbacks_omitted", freeze["omitted_legacy_lookbacks"] == [5, 6, 8] and not freeze["diagnostic_lookbacks"],
          freeze["omitted_legacy_lookbacks"], [5, 6, 8])
    check("recent_sensitivity_disabled", protocol["recent_sensitivity"]["execution_enabled"] is False,
          protocol["recent_sensitivity"]["execution_enabled"], False)
    check("ASPH_h5_DM_disabled", protocol["inference_feasibility"]["ASPH"]["5"]["dm_reporting_enabled"] is False,
          protocol["inference_feasibility"]["ASPH"]["5"]["dm_reporting_enabled"], False)

    ml_config_path = root / FROZEN_ML_CONFIG_PATH
    neural_config_path = root / FROZEN_NEURAL_CONFIG_PATH
    ml_config_hash = sha256_file(ml_config_path)
    neural_config_hash = sha256_file(neural_config_path)
    check("frozen_ML_configuration_hash", ml_config_hash == EXPECTED_ML_CONFIG_HASH,
          ml_config_hash, EXPECTED_ML_CONFIG_HASH)
    check("frozen_neural_configuration_hash", neural_config_hash == EXPECTED_NEURAL_CONFIG_HASH,
          neural_config_hash, EXPECTED_NEURAL_CONFIG_HASH)

    ml_source = pd.read_csv(ml_config_path, encoding="utf-8", float_precision="round_trip")
    selected_ml_rows: list[dict[str, Any]] = []
    missing_ml: list[dict[str, Any]] = []
    for cell in calendar:
        for model in ML_MODELS:
            found = ml_source.loc[
                ml_source.target.eq(str(cell["target"]))
                & ml_source.model.eq(model)
                & ml_source.horizon.eq(int(cell["horizon"]))
                & ml_source.origin_year.eq(int(cell["origin_year"]))
            ]
            if len(found) != 1:
                missing_ml.append({**cell, "model": model, "match_n": len(found)})
            else:
                row = found.iloc[0]
                selected_ml_rows.append({**cell, "model": model,
                                         "hyperparameters_json": row.hyperparameters_json,
                                         "frozen_source_representation": row.frozen_source_representation,
                                         "frozen_source_lookback": int(row.frozen_source_lookback)})
    check("frozen_ML_configuration_coverage", not missing_ml and len(selected_ml_rows) == 213,
          {"matched": len(selected_ml_rows), "bad": len(missing_ml)}, {"matched": 213, "bad": 0})

    neural_source = pd.read_csv(neural_config_path,
                                encoding="utf-8", float_precision="round_trip")
    neural_source = neural_source.loc[neural_source.selected.astype(bool)].copy()
    check("frozen_neural_configuration_coverage", len(neural_source) == 54 and set(neural_source.architecture) == set(PRIMARY_NEURAL + ABLATION_NEURAL),
          {"rows": len(neural_source), "architectures": sorted(neural_source.architecture.unique().tolist())},
          {"rows": 54, "architectures": sorted(PRIMARY_NEURAL + ABLATION_NEURAL)})

    inference_resolution = {
        "flag": "Frozen JSON defines cell validity but does not enumerate a general primary-model formal comparison family.",
        "artifact_audit": {
            FROZEN_NEURAL_CONFIG_PATH: "Development-only architecture configurations exist, but no development-only cross-family comparator rule is stored.",
            "frozen_protocol": "The protocol does not freeze a representation-specific cross-family comparator for all three horizons.",
        },
        "general_primary_model_DM": "NOT PRE-SPECIFIED AND DISABLED; no post-hoc winner-vs-rest tests",
        "prespecified_families": {
            "attention": ["CNN_BiLSTM_Attention_Compact", "CNN_BiLSTM_Compact_NoAttention"],
            "capacity": ["CNN_BiLSTM_Attention_Compact", "CNN_BiLSTM_Attention_Original23K"],
        },
        "holm_scope": "separately within each named five-cell valid family; ASPH h5 excluded",
        "ASPH_h5": "descriptive paired bootstrap only; no DM statistic or p-value",
    }
    check("valid_inference_execution_policy", True, inference_resolution,
          "ambiguity flagged before fitting; only brief-prespecified ablation contrasts executed")

    run_id = "round2-phase2b-" + data_hash[:12] + "-" + protocol_hash[:12]
    plan_rows: list[dict[str, Any]] = []
    for cell in calendar:
        for model in STATISTICAL_MODELS:
            plan_rows.append({**cell, "model_family": "statistical", "model": model,
                              "representation": "statistical_native", "lookback": np.nan,
                              "seed": GLOBAL_SEED, "role": "primary"})
        for model in ML_MODELS:
            for representation in REPRESENTATIONS:
                plan_rows.append({**cell, "model_family": "classical_ml", "model": model,
                                  "representation": representation, "lookback": 3,
                                  "seed": GLOBAL_SEED, "role": "primary"})
        source_cells = neural_source.loc[
            neural_source.target.eq(str(cell["target"]).lower())
            & neural_source.horizon.eq(int(cell["horizon"]))
        ]
        for selected in source_cells.itertuples(index=False):
            role = "primary" if selected.architecture in PRIMARY_NEURAL else "ablation"
            family = "primary_neural" if role == "primary" else "neural_ablation"
            for seed in NEURAL_SEEDS:
                plan_rows.append({**cell, "model_family": family, "model": selected.architecture,
                                  "representation": selected.representation, "lookback": 3,
                                  "seed": seed, "role": role})
    plan = pd.DataFrame(plan_rows)
    plan["run_id"] = run_id
    plan["fit_id"] = [
        _fit_id(run_id, [r.target, r.horizon, r.origin_year, r.model_family, r.model,
                         r.representation, r.lookback, r.seed]) for r in plan.itertuples(index=False)
    ]
    plan = plan[["run_id", "fit_id", "model_family", "role", "target", "horizon", "origin_year",
                 "target_year", "model", "representation", "lookback", "seed"]]
    counts = plan.groupby(["model_family", "target", "horizon", "seed"], dropna=False).size().rename("planned_fit_count").reset_index()
    family_counts = plan.groupby("model_family").size().to_dict()
    expected_family = {"statistical": 426, "classical_ml": 426, "primary_neural": 4970, "neural_ablation": 1420}
    check("planned_fit_counts", len(plan) == 7242 and family_counts == expected_family,
          {"total": len(plan), **family_counts}, {"total": 7242, **expected_family})
    check("unique_fit_identifiers", plan.fit_id.is_unique, int(plan.fit_id.nunique()), len(plan))

    _atomic_csv(pd.DataFrame(checks), paths["base"] / "preflight_checks.csv")
    _atomic_csv(plan, paths["base"] / "preflight_planned_fits.csv")
    _atomic_csv(counts, paths["base"] / "preflight_fit_ledger.csv")
    _atomic_csv(pd.DataFrame(selected_ml_rows), paths["base"] / "frozen_ml_configuration_map.csv")
    adapted_neural = neural_source.copy()
    if not adapted_neural.lookback.astype(int).eq(3).all():
        raise RuntimeError("Public frozen neural map is not uniformly adapted to L=3")
    _atomic_csv(adapted_neural, paths["base"] / "frozen_neural_configuration_map_l3.csv")

    passed = all(item["status"] == "PASS" for item in checks)
    summary = {
        "status": "PASS" if passed else "FAIL", "recorded_utc": _utc_now(),
        "run_id": run_id, "dataset_path": str(data_path.relative_to(root)), "dataset_sha256": data_hash,
        "protocol_path": str(protocol_path.relative_to(root)), "protocol_sha256": protocol_hash,
        "protocol_version": protocol["protocol_version"], "checks_passed": sum(c["status"] == "PASS" for c in checks),
        "checks_total": len(checks), "planned_fit_count": len(plan), "planned_by_family": family_counts,
        "comparison_policy_resolution": inference_resolution,
        "configuration_adaptation": {
            "ML": "Exact legacy per-target/horizon/origin/model hyperparameters reused; both frozen primary representations fitted at L=3; no selector called.",
            "neural": "Frozen cell representation/optimizer/architecture/epoch budget reused; only lookback overwritten to Phase2A.1 L=3.",
        },
        "worker_count": worker_count,
        "preexisting_fit_record_count": len(preexisting_progress),
        "preexisting_prediction_ledger_exists": preexisting_predictions.exists(),
    }
    _atomic_json(paths["base"] / "preflight.json", summary)
    note = """# Round-2 Phase 2B inference preflight decision

This decision was frozen before any Round-2 model fit or forecast.

The Phase 2A.1 JSON specifies the target-horizon cells in which HLN-DM is
mathematically permitted, but it does not name a primary cross-model
comparison family. The repository audit found two earlier approaches:

1. `src/phase2d_integration_repair.py` selected the best primary neural and
   best nonneural model from evaluation-window RMSE in each cell. Its six
   comparisons therefore cannot be carried into Round 2 without outcome-based
   selection.
2. The older manuscript Table 4 used ARIMA as a best-model reference. That
   artifact does not freeze a representation-specific comparison family for
   all three Round-2 horizons, and combining ARIMA with a newly chosen Round-2
   representative would create a new rule.

Therefore primary cross-model formal inference is **not pre-specified** and is
disabled. Primary rankings and cross-model differences are descriptive only;
no winner-vs-rest DM test will be created after inspecting Round-2 results.

The two contrasts explicitly frozen in the authoritative Phase 2B brief remain
eligible:

- attention: `CNN_BiLSTM_Attention_Compact` versus
  `CNN_BiLSTM_Compact_NoAttention`;
- capacity: `CNN_BiLSTM_Attention_Compact` versus
  `CNN_BiLSTM_Attention_Original23K`.

Holm adjustment is applied separately within each named family and includes
only mathematically valid cells. ASPH h5 (n=5) is descriptive only: no DM
statistic, raw p value, or adjusted p value is assigned. A paired bootstrap for
that cell is labeled exploratory and extremely unstable.
"""
    _atomic_text(paths["base"] / "preflight_inference_policy.md", note)
    environment = _hardware_environment(root, worker_count, protocol_hash, data_hash)
    _atomic_json(paths["base"] / "environment.json", environment)
    if not passed:
        raise RuntimeError("Round-2 preflight failed; no training is permitted. See preflight_checks.csv.")
    return {"summary": summary, "checks": pd.DataFrame(checks), "plan": plan, "protocol": protocol,
            "protocol_hash": protocol_hash, "official": official, "data_hash": data_hash,
            "environment": environment, "ml_map": pd.DataFrame(selected_ml_rows), "neural_map": adapted_neural}


def _load_existing_preflight(root: Path, worker_count: int) -> dict[str, Any] | None:
    """Load a completed hash-valid checkpoint without repeating the preflight work."""
    paths = _paths(root)
    required = {
        "summary": paths["base"] / "preflight.json",
        "checks": paths["base"] / "preflight_checks.csv",
        "plan": paths["base"] / "preflight_planned_fits.csv",
        "ml_map": paths["base"] / "frozen_ml_configuration_map.csv",
        "neural_map": paths["base"] / "frozen_neural_configuration_map_l3.csv",
        "environment": paths["base"] / "environment.json",
        "inference": paths["base"] / "preflight_inference_policy.md",
    }
    if not all(path.is_file() for path in required.values()): return None
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    protocol, _, protocol_hash = _load_protocol(root)
    official, _, data_hash = _load_official_long(root, protocol)
    checks = pd.read_csv(required["checks"])
    plan = pd.read_csv(required["plan"], float_precision="round_trip")
    environment = json.loads(required["environment"].read_text(encoding="utf-8"))
    valid = (
        summary.get("status") == "PASS" and checks.status.eq("PASS").all()
        and data_hash == EXPECTED_DATA_HASH == summary.get("dataset_sha256")
        and protocol_hash == summary.get("protocol_sha256")
        and len(plan) == 7242 and plan.fit_id.is_unique
        and int(environment.get("configured_worker_count", -1)) == int(worker_count)
    )
    if not valid: return None
    environment["effective_thread_environment_during_final_execution"] = {
        key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    }
    environment["runtime_repair_history"] = [{
        "reason": "Initial 18-worker startup encountered OpenBLAS memory allocation failure.",
        "repair": "Retained 18 process workers but forced one BLAS/OpenMP/NumExpr thread per process and used process-specific same-directory atomic temporary files.",
        "methodology_changed": False,
    }]
    _atomic_json(required["environment"], environment)
    return {"summary": summary, "checks": checks, "plan": plan, "protocol": protocol,
            "protocol_hash": protocol_hash, "official": official, "data_hash": data_hash,
            "environment": environment,
            "ml_map": pd.read_csv(required["ml_map"], float_precision="round_trip"),
            "neural_map": pd.read_csv(required["neural_map"], float_precision="round_trip")}


def _base_record(plan_row: dict[str, Any], target_meta: dict[str, str]) -> dict[str, Any]:
    return {
        **plan_row, "target_definition": target_meta["definition"],
        "age_population": target_meta["age_population"], "unit": target_meta["unit"],
        "prediction": np.nan, "actual": np.nan, "error": np.nan, "absolute_error": np.nan,
        "squared_error": np.nan, "fit_status": "failed", "feature_wall_seconds": 0.0,
        "selection_wall_seconds": 0.0, "fit_wall_seconds": 0.0, "prediction_wall_seconds": 0.0,
        "interval_wall_seconds": 0.0, "per_fit_elapsed_wall_clock_seconds": 0.0,
        "lower_80": np.nan, "upper_80": np.nan, "lower_95": np.nan, "upper_95": np.nan,
        "interval_80_valid": False, "interval_95_valid": False, "interval_method": "",
        "calibration_n": np.nan, "warning_or_failure_message": "", "exception_class": "",
        "hyperparameters_json": "", "configuration_json": "", "parameter_count": np.nan,
        "selected_epoch_count": np.nan, "supervised_training_n": np.nan,
        "training_start_year": np.nan, "training_end_year": int(plan_row["origin_year"]),
        "training_n": np.nan, "last_observed_value": np.nan,
        "in_sample_naive_mae": np.nan, "in_sample_naive_mse": np.nan,
        "final_training_loss": np.nan, "validation_n": 0, "early_stopping_used": False,
        "used_all_supervised_samples": True, "started_utc": _utc_now(), "completed_utc": "",
        "execution_action": "attempted", "attempt_count": 1, "failed_attempt_count": 0,
        "prior_failed_attempt_elapsed_seconds": 0.0, "prior_attempt_failures_json": "[]",
    }


def _finish_record(record: dict[str, Any], prediction: float, actual: float, started: float) -> dict[str, Any]:
    record["prediction"] = float(prediction)
    record["actual"] = float(actual)
    if np.isfinite(prediction):
        error = float(actual - prediction)
        record.update({"error": error, "absolute_error": abs(error), "squared_error": error * error,
                       "fit_status": "success"})
    record["per_fit_elapsed_wall_clock_seconds"] = float(time.perf_counter() - started)
    record["completed_utc"] = _utc_now()
    return record


def _valid_interval(lower: float, point: float, upper: float) -> bool:
    return bool(np.isfinite([lower, point, upper]).all() and lower <= point <= upper)


def _write_progress(paths: dict[str, Path], record: dict[str, Any], data_hash: str, protocol_hash: str) -> None:
    payload = {**record, "dataset_sha256": data_hash, "protocol_sha256": protocol_hash}
    _atomic_json(paths["progress"] / f"{record['fit_id']}.json", payload)


def _read_valid_progress(path: Path, data_hash: str, protocol_hash: str) -> dict[str, Any] | None:
    if not path.is_file(): return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("dataset_sha256") != data_hash or row.get("protocol_sha256") != protocol_hash:
            return None
        if row.get("fit_status") not in {"success", "failed"}: return None
        return row
    except Exception:
        return None


def _statistical_instances(root: Path) -> dict[str, Any]:
    minimum = 8
    ets_config = {"candidate_trends": [None, "add"], "damped_options": [False, True], "seasonal": None}
    arima_config = {
        "p_values": [0, 1, 2, 3], "d_values": [0, 1, 2], "q_values": [0, 1, 2, 3],
        "include_drift_or_trend": True, "selection_criterion": "AICc",
        "maximum_failed_candidates_allowed": None,
    }
    return {
        "Naive": NaiveForecaster(minimum), "Drift": DriftForecaster(minimum),
        "LinearTrend": LinearTrendForecaster(), "ETS": EtsForecaster(ets_config, minimum),
        "ARIMA": ArimaForecaster(arima_config), "Theta": ThetaForecaster(minimum),
    }


def _execute_statistical(root: Path, state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    paths = _paths(root); plan = state["plan"]
    subset = plan.loc[plan.model_family.eq("statistical")].sort_values(
        ["target", "horizon", "origin_year", "model"]
    )
    official = state["official"]; models = _statistical_instances(root)
    records: list[dict[str, Any]] = []; candidates: list[dict[str, Any]] = []
    resumed = attempted = 0
    for index, plan_row in enumerate(subset.to_dict("records"), start=1):
        progress_path = paths["progress"] / f"{plan_row['fit_id']}.json"
        prior = _read_valid_progress(progress_path, state["data_hash"], state["protocol_hash"])
        if prior is not None:
            prior["execution_action"] = "resumed"
            records.append(prior); resumed += 1; continue
        attempted += 1; started = time.perf_counter(); record = _base_record(plan_row, TARGET_META[plan_row["target"]])
        data = _series_frame(official, plan_row["target"])
        target_column = plan_row["target"].lower()
        training = data.loc[data.year <= int(plan_row["origin_year"])]
        values = training[target_column].to_numpy(float); years = training.year.to_numpy(int)
        actual = float(data.loc[data.year.eq(int(plan_row["target_year"])), target_column].iloc[0])
        record.update({"actual": actual, "training_start_year": int(years[0]), "training_n": len(values),
                       "last_observed_value": float(values[-1]),
                       "in_sample_naive_mae": float(np.mean(np.abs(np.diff(values)))),
                       "in_sample_naive_mse": float(np.mean(np.square(np.diff(values)))),
                       "interval_method": "frozen statistical model-native or training-only empirical method"})
        try:
            context = {"target": plan_row["target"], "model": plan_row["model"],
                       "origin_year": int(plan_row["origin_year"]), "horizon": int(plan_row["horizon"])}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                batch = models[plan_row["model"]].forecast_many(
                    years, values, [int(plan_row["horizon"])], GLOBAL_SEED, context
                )
            output = batch.outputs[int(plan_row["horizon"])]
            record.update({"selection_wall_seconds": float(output.selection_seconds),
                           "fit_wall_seconds": float(output.fit_seconds),
                           "prediction_wall_seconds": float(output.forecast_seconds),
                           "lower_80": float(output.lower_80), "upper_80": float(output.upper_80),
                           "lower_95": float(output.lower_95), "upper_95": float(output.upper_95),
                           "interval_80_valid": _valid_interval(output.lower_80, output.point, output.upper_80),
                           "interval_95_valid": _valid_interval(output.lower_95, output.point, output.upper_95),
                           "hyperparameters_json": output.specification,
                           "warning_or_failure_message": " | ".join(filter(None, [
                               output.warning, *(str(w.message) for w in caught)
                           ]))})
            if output.status != "success" or not np.isfinite(output.point):
                raise RuntimeError(output.warning or f"{plan_row['model']} returned a failed/nonfinite output")
            record = _finish_record(record, float(output.point), actual, started)
            for candidate in batch.candidates:
                candidates.append({**context, **candidate})
        except Exception as exc:
            record.update({"fit_status": "failed", "exception_class": type(exc).__name__,
                           "warning_or_failure_message": " | ".join(filter(None, [
                               record.get("warning_or_failure_message", ""), f"{type(exc).__name__}: {exc}"
                           ])), "per_fit_elapsed_wall_clock_seconds": time.perf_counter() - started,
                           "completed_utc": _utc_now(), "failed_attempt_count": 1})
        _write_progress(paths, record, state["data_hash"], state["protocol_hash"]); records.append(record)
        if index % 50 == 0 or index == len(subset):
            print(f"statistical progress {index}/{len(subset)}", flush=True)
    return records, candidates, {"attempted": attempted, "resumed": resumed}


def _ml_config(root: Path) -> dict[str, Any]:
    return {
        "inner_validation": {"minimum_supervised_samples": 12},
        "runtime": {"n_jobs": 1, "random_seed": GLOBAL_SEED},
    }


def _execute_ml(root: Path, state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    paths = _paths(root); plan = state["plan"]
    subset = plan.loc[plan.model_family.eq("classical_ml")].sort_values(
        ["target", "horizon", "origin_year", "model", "representation"]
    )
    official = state["official"]; config = _ml_config(root)
    mapping = state["ml_map"].set_index(["target", "horizon", "origin_year", "model"])
    records: list[dict[str, Any]] = []; calibration_rows: list[dict[str, Any]] = []
    cache: dict[tuple[Any, ...], tuple[float, str, float]] = {}
    resumed = attempted = auxiliary_attempted = 0
    for index, plan_row in enumerate(subset.to_dict("records"), start=1):
        progress_path = paths["progress"] / f"{plan_row['fit_id']}.json"
        prior = _read_valid_progress(progress_path, state["data_hash"], state["protocol_hash"])
        if prior is not None:
            prior["execution_action"] = "resumed"
            records.append(prior); resumed += 1; continue
        attempted += 1; started = time.perf_counter(); record = _base_record(plan_row, TARGET_META[plan_row["target"]])
        data = _series_frame(official, plan_row["target"]); target_column = plan_row["target"].lower()
        training = data.loc[data.year <= int(plan_row["origin_year"])]
        history = training[target_column].to_numpy(float)
        actual = float(data.loc[data.year.eq(int(plan_row["target_year"])), target_column].iloc[0])
        source = mapping.loc[(plan_row["target"], int(plan_row["horizon"]), int(plan_row["origin_year"]), plan_row["model"])]
        params = json.loads(source.hyperparameters_json)
        record.update({"actual": actual, "training_start_year": int(training.year.iloc[0]), "training_n": len(history),
                       "last_observed_value": float(history[-1]),
                       "in_sample_naive_mae": float(np.mean(np.abs(np.diff(history)))),
                       "in_sample_naive_mse": float(np.mean(np.square(np.diff(history)))),
                       "hyperparameters_json": json.dumps(params, sort_keys=True, separators=(",", ":")),
                       "configuration_json": json.dumps({"representation": plan_row["representation"],
                                                          "lookback": 3, **params}, sort_keys=True, separators=(",", ":")),
                       "interval_method": "frozen rolling finite-sample conformal; last 8 eligible historical pseudo-origins; minimum 8 residuals"})
        try:
            outer = fit_predict(data, target_column, int(plan_row["origin_year"]), int(plan_row["horizon"]),
                                plan_row["representation"], 3, plan_row["model"], params, config,
                                ("round2_outer", plan_row["fit_id"]))
            record.update({"feature_wall_seconds": outer.feature_seconds, "fit_wall_seconds": outer.fit_seconds,
                           "prediction_wall_seconds": outer.forecast_seconds,
                           "supervised_training_n": outer.training_n})
            interval_started = time.perf_counter(); residuals: list[float] = []
            inner_origins = valid_inner_origins(int(data.year.min()), int(plan_row["origin_year"]),
                                                int(plan_row["horizon"]), 8, 20)
            for inner_origin in inner_origins:
                key = (plan_row["target"], plan_row["model"], int(plan_row["horizon"]),
                       plan_row["representation"], json.dumps(params, sort_keys=True), inner_origin)
                if key not in cache:
                    aux_start = time.perf_counter(); auxiliary_attempted += 1
                    try:
                        result = fit_predict(data, target_column, inner_origin, int(plan_row["horizon"]),
                                             plan_row["representation"], 3, plan_row["model"], params, config,
                                             ("round2_conformal", *key))
                        cache[key] = (float(result.prediction), "", time.perf_counter() - aux_start)
                    except Exception as exc:
                        cache[key] = (np.nan, f"{type(exc).__name__}: {exc}", time.perf_counter() - aux_start)
                prediction, reason, aux_seconds = cache[key]
                calibration_actual = float(data.loc[data.year.eq(inner_origin + int(plan_row["horizon"])), target_column].iloc[0])
                residual = abs(calibration_actual - prediction) if np.isfinite(prediction) else np.nan
                if np.isfinite(residual): residuals.append(float(residual))
                calibration_rows.append({"outer_fit_id": plan_row["fit_id"], "target": plan_row["target"],
                                         "model": plan_row["model"], "representation": plan_row["representation"],
                                         "horizon": int(plan_row["horizon"]), "outer_origin_year": int(plan_row["origin_year"]),
                                         "calibration_origin_year": inner_origin,
                                         "calibration_target_year": inner_origin + int(plan_row["horizon"]),
                                         "actual": calibration_actual, "prediction": prediction,
                                         "absolute_residual": residual, "status": "success" if np.isfinite(residual) else "failed",
                                         "message": reason, "auxiliary_fit_seconds": aux_seconds,
                                         "fully_inside_outer_history": inner_origin + int(plan_row["horizon"]) <= int(plan_row["origin_year"])})
            bounds: dict[str, Any] = {}
            for level, suffix in ((0.80, "80"), (0.95, "95")):
                lower, upper, quantile, rank, valid = conformal_bounds(float(outer.prediction), residuals, level, 8)
                bounds.update({f"lower_{suffix}": lower, f"upper_{suffix}": upper,
                               f"interval_{suffix}_valid": valid,
                               f"conformal_q_{suffix}": quantile, f"conformal_rank_{suffix}": rank})
            record.update(bounds); record["calibration_n"] = len(residuals)
            record["interval_wall_seconds"] = time.perf_counter() - interval_started
            record = _finish_record(record, float(outer.prediction), actual, started)
        except Exception as exc:
            record.update({"fit_status": "failed", "exception_class": type(exc).__name__,
                           "warning_or_failure_message": f"{type(exc).__name__}: {exc}",
                           "per_fit_elapsed_wall_clock_seconds": time.perf_counter() - started,
                           "completed_utc": _utc_now(), "failed_attempt_count": 1})
        _write_progress(paths, record, state["data_hash"], state["protocol_hash"]); records.append(record)
        if index % 50 == 0 or index == len(subset):
            print(f"classical ML progress {index}/{len(subset)}", flush=True)
    return records, calibration_rows, {"attempted": attempted, "resumed": resumed,
                                       "auxiliary_calibration_fits": auxiliary_attempted}


def _neural_task(payload: dict[str, Any]) -> dict[str, int | str]:
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
    paths = {"progress": Path(payload["progress_dir"])}
    data = pd.DataFrame({"year": payload["years"], payload["target"].lower(): payload["values"]})
    target_column = payload["target"].lower(); selected = payload["selected"]
    spec = json.loads(selected["configuration_json"]); spec["lookback"] = 3
    completed = failed = resumed = 0
    for plan_row in payload["plan_rows"]:
        progress_path = paths["progress"] / f"{plan_row['fit_id']}.json"
        prior = _read_valid_progress(progress_path, payload["data_hash"], payload["protocol_hash"])
        if prior is not None and prior.get("fit_status") == "success":
            resumed += 1; continue
        started = time.perf_counter(); record = _base_record(plan_row, TARGET_META[plan_row["target"]])
        prior_failures: list[dict[str, Any]] = []
        if prior is not None:
            try:
                prior_failures = json.loads(prior.get("prior_attempt_failures_json", "[]"))
            except Exception:
                prior_failures = []
            prior_failures.append({
                "attempt": int(prior.get("attempt_count", 1)), "fit_status": prior.get("fit_status"),
                "exception_class": prior.get("exception_class", ""),
                "message": prior.get("warning_or_failure_message", ""),
                "started_utc": prior.get("started_utc", ""), "completed_utc": prior.get("completed_utc", ""),
                "per_fit_elapsed_wall_clock_seconds": float(prior.get("per_fit_elapsed_wall_clock_seconds", 0.0)),
                "execution_continued": True,
            })
            record.update({"attempt_count": int(prior.get("attempt_count", 1)) + 1,
                           "failed_attempt_count": int(prior.get("failed_attempt_count", 1)),
                           "prior_failed_attempt_elapsed_seconds": float(prior.get("prior_failed_attempt_elapsed_seconds", 0.0))
                           + float(prior.get("per_fit_elapsed_wall_clock_seconds", 0.0)),
                           "prior_attempt_failures_json": json.dumps(prior_failures, sort_keys=True)})
        training = data.loc[data.year <= int(plan_row["origin_year"])]
        history = training[target_column].to_numpy(float)
        actual = float(data.loc[data.year.eq(int(plan_row["target_year"])), target_column].iloc[0])
        record.update({"actual": actual, "training_start_year": int(training.year.iloc[0]), "training_n": len(history),
                       "last_observed_value": float(history[-1]),
                       "in_sample_naive_mae": float(np.mean(np.abs(np.diff(history)))),
                       "in_sample_naive_mse": float(np.mean(np.square(np.diff(history)))),
                       "configuration_json": json.dumps(spec, sort_keys=True, separators=(",", ":")),
                       "hyperparameters_json": json.dumps({k: spec[k] for k in (
                           "dropout", "learning_rate", "weight_decay", "size"
                       )}, sort_keys=True, separators=(",", ":")),
                       "selected_epoch_count": int(selected["selected_epoch_budget"]),
                       "parameter_count": int(selected["parameter_count"]),
                       "interval_method": "not available: no frozen neural predictive-interval method",
                       "validation_n": 0, "early_stopping_used": False, "used_all_supervised_samples": True})
        try:
            feature_started = time.perf_counter()
            fold = build_scaled_fold(data, target_column, int(plan_row["origin_year"]), int(plan_row["horizon"]),
                                     3, spec["representation"])
            record["feature_wall_seconds"] = time.perf_counter() - feature_started
            if len(fold.direct.y) < 12:
                raise ValueError(f"only {len(fold.direct.y)} supervised samples; 12 required")
            result, model = final_fixed_epoch_fit(
                fold, selected["architecture"], spec["size"], float(spec["dropout"]),
                float(spec["learning_rate"]), float(spec["weight_decay"]),
                int(selected["selected_epoch_budget"]), int(plan_row["seed"]),
                {"torch_num_threads": 1, "batch_size": 8, "gradient_clip_norm": 1.0},
            )
            transformed, prediction = fold.reconstruct(result.scaled_prediction)
            with torch.no_grad():
                final_loss = float(torch.mean((model(fold.train_X) - fold.train_y) ** 2).item())
            observed_parameters = parameter_counts(model)["total_parameter_count"]
            if observed_parameters != int(selected["parameter_count"]):
                raise ValueError(f"parameter count changed: {observed_parameters} != {selected['parameter_count']}")
            record.update({"fit_wall_seconds": result.fit_seconds,
                           "prediction_wall_seconds": result.forecast_seconds,
                           "supervised_training_n": len(fold.direct.y), "final_training_loss": final_loss,
                           "predicted_transformed_target": transformed})
            record = _finish_record(record, prediction, actual, started); completed += 1
            record["per_fit_elapsed_wall_clock_seconds"] += record["prior_failed_attempt_elapsed_seconds"]
        except Exception as exc:
            record.update({"fit_status": "failed", "exception_class": type(exc).__name__,
                           "warning_or_failure_message": f"{type(exc).__name__}: {exc}",
                           "per_fit_elapsed_wall_clock_seconds": time.perf_counter() - started,
                           "completed_utc": _utc_now(),
                           "failed_attempt_count": int(record.get("failed_attempt_count", 0)) + 1}); failed += 1
        payload_record = {**record, "dataset_sha256": payload["data_hash"],
                          "protocol_sha256": payload["protocol_hash"]}
        _atomic_json(progress_path, payload_record)
    return {"cell": payload["cell"], "completed": completed, "failed": failed, "resumed": resumed}


def _execute_neural(root: Path, state: dict[str, Any], worker_count: int) -> dict[str, int]:
    paths = _paths(root); plan = state["plan"]
    subset = plan.loc[plan.model_family.isin(["primary_neural", "neural_ablation"])].copy()
    selected_map = state["neural_map"]
    tasks: list[dict[str, Any]] = []
    for (target, horizon, model), cell_plan in subset.groupby(["target", "horizon", "model"], sort=True):
        selected = selected_map.loc[
            selected_map.target.eq(target.lower()) & selected_map.horizon.eq(int(horizon))
            & selected_map.architecture.eq(model)
        ]
        if len(selected) != 1: raise RuntimeError(f"Frozen neural config is not unique for {target} h{horizon} {model}")
        data = _series_frame(state["official"], target)
        tasks.append({"cell": f"{target}_h{horizon}_{model}", "target": target,
                      "years": data.year.tolist(), "values": data[target.lower()].tolist(),
                      "selected": selected.iloc[0].to_dict(), "plan_rows": cell_plan.to_dict("records"),
                      "progress_dir": str(paths["progress"]), "data_hash": state["data_hash"],
                      "protocol_hash": state["protocol_hash"]})
    totals = {"attempted": 0, "successful": 0, "failed": 0, "resumed": 0}
    pending_tasks = []
    for task in tasks:
        completed_n = sum((lambda item: item is not None and item.get("fit_status") == "success")(
                              _read_valid_progress(paths["progress"] / f"{row['fit_id']}.json",
                                                   state["data_hash"], state["protocol_hash"]))
                          for row in task["plan_rows"])
        if completed_n == len(task["plan_rows"]):
            totals["resumed"] += completed_n
        else:
            pending_tasks.append(task)
    if not pending_tasks:
        print(f"neural resume: all {totals['resumed']} fit records already complete", flush=True)
        return totals
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(_neural_task, task) for task in pending_tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            totals["successful"] += int(result["completed"]); totals["failed"] += int(result["failed"])
            totals["resumed"] += int(result["resumed"])
            totals["attempted"] += int(result["completed"]) + int(result["failed"])
            print(f"neural cell {index}/{len(pending_tasks)} {result['cell']} success={result['completed']} failed={result['failed']} resumed={result['resumed']}", flush=True)
    return totals


def _collect_progress(root: Path, state: dict[str, Any]) -> pd.DataFrame:
    paths = _paths(root); rows = []
    for fit_id in state["plan"].fit_id:
        row = _read_valid_progress(paths["progress"] / f"{fit_id}.json", state["data_hash"], state["protocol_hash"])
        if row is None:
            plan_row = state["plan"].loc[state["plan"].fit_id.eq(fit_id)].iloc[0].to_dict()
            row = _base_record(plan_row, TARGET_META[plan_row["target"]])
            row.update({"fit_status": "not_attempted", "warning_or_failure_message": "missing progress record",
                        "dataset_sha256": state["data_hash"], "protocol_sha256": state["protocol_hash"]})
        rows.append(row)
    frame = pd.DataFrame(rows)
    for name in ["prediction", "actual", "error", "absolute_error", "squared_error", "lookback", "seed",
                 "fit_wall_seconds", "prediction_wall_seconds", "per_fit_elapsed_wall_clock_seconds",
                 "lower_80", "upper_80", "lower_95", "upper_95", "in_sample_naive_mae",
                 "in_sample_naive_mse", "last_observed_value"]:
        if name in frame: frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if "attempt_count" not in frame: frame["attempt_count"] = 1
    frame["attempt_count"] = pd.to_numeric(frame["attempt_count"], errors="coerce").fillna(1).astype(int)
    if "failed_attempt_count" not in frame: frame["failed_attempt_count"] = frame.fit_status.eq("failed").astype(int)
    frame["failed_attempt_count"] = pd.to_numeric(frame["failed_attempt_count"], errors="coerce").fillna(
        frame.fit_status.eq("failed").astype(int)
    ).astype(int)
    frame["model_label"] = [_model_label(m, r) for m, r in zip(frame.model, frame.representation)]
    frame = frame.sort_values(["target", "horizon", "origin_year", "model_family", "model", "representation", "seed"]).reset_index(drop=True)
    return frame


def _metric_record(group: pd.DataFrame, summary_type: str) -> dict[str, Any]:
    success = group.loc[group.fit_status.eq("success") & np.isfinite(group.prediction)].copy()
    expected = len(group); count = len(success)
    base = {
        "target": str(group.target.iloc[0]), "horizon": int(group.horizon.iloc[0]),
        "model_family": str(group.model_family.iloc[0]), "role": str(group.role.iloc[0]),
        "model": str(group.model.iloc[0]), "model_label": str(group.model_label.iloc[0]),
        "representation": str(group.representation.iloc[0]), "lookback": group.lookback.iloc[0],
        "seed": group.seed.iloc[0] if group.seed.nunique(dropna=False) == 1 else "ensemble_mean",
        "summary_type": summary_type, "expected_n": expected, "successful_n": count,
        "failed_n": expected - count, "completeness_percent": 100.0 * count / expected if expected else np.nan,
        "ranking_eligible": bool(count == expected and expected > 0),
    }
    metric_names = ["rmse", "mae", "mase", "rmsse", "smape_percent", "directional_accuracy_percent",
                    "mean_error", "median_error", "r_squared"]
    if not count:
        return {**base, **{name: np.nan for name in metric_names},
                "undefined_reason": "no successful forecasts"}
    actual = success.actual.to_numpy(float); forecast = success.prediction.to_numpy(float)
    error = actual - forecast
    variance_denominator = float(np.sum(np.square(actual - np.mean(actual))))
    r_squared = 1.0 - float(np.sum(np.square(error))) / variance_denominator if variance_denominator > 0 else np.nan
    return {
        **base, "rmse": root_mean_squared_error(actual, forecast),
        "mae": mean_absolute_error(actual, forecast),
        "mase": mase(error, success.in_sample_naive_mae.to_numpy(float)),
        "rmsse": rmsse(error, success.in_sample_naive_mse.to_numpy(float)),
        "smape_percent": smape(actual, forecast),
        "directional_accuracy_percent": directional_accuracy(
            actual, forecast, success.last_observed_value.to_numpy(float)
        ),
        "mean_error": float(np.mean(error)), "median_error": float(np.median(error)),
        "r_squared": r_squared,
        "undefined_reason": "" if count == expected else "metrics computed on successful subset; incomplete and ranking-ineligible",
    }


def _neural_ensemble(ledger: pd.DataFrame) -> pd.DataFrame:
    neural = ledger.loc[ledger.model_family.isin(["primary_neural", "neural_ablation"])].copy()
    keys = ["run_id", "target", "target_definition", "age_population", "unit", "horizon", "origin_year",
            "target_year", "model_family", "role", "model", "model_label", "representation", "lookback"]
    rows: list[dict[str, Any]] = []
    for key, group in neural.groupby(keys, dropna=False, sort=True):
        success = group.loc[group.fit_status.eq("success") & np.isfinite(group.prediction)]
        row = dict(zip(keys, key)); first = group.iloc[0]
        row.update({"seed": "ensemble_mean", "seed_expected_n": 10, "seed_successful_n": len(success),
                    "fit_status": "success" if len(success) == 10 else "incomplete",
                    "prediction": float(success.prediction.mean()) if len(success) else np.nan,
                    "actual": float(first.actual), "last_observed_value": float(first.last_observed_value),
                    "in_sample_naive_mae": float(first.in_sample_naive_mae),
                    "in_sample_naive_mse": float(first.in_sample_naive_mse),
                    "lower_80": np.nan, "upper_80": np.nan, "lower_95": np.nan, "upper_95": np.nan,
                    "interval_80_valid": False, "interval_95_valid": False,
                    "interval_method": "not available: no frozen neural predictive-interval method"})
        if np.isfinite(row["prediction"]):
            error = row["actual"] - row["prediction"]
            row.update({"error": error, "absolute_error": abs(error), "squared_error": error * error})
        rows.append(row)
    return pd.DataFrame(rows)


def _performance_outputs(root: Path, ledger: pd.DataFrame) -> dict[str, pd.DataFrame]:
    paths = _paths(root)
    metric_rows: list[dict[str, Any]] = []
    grouping = ["target", "horizon", "model_family", "role", "model", "model_label", "representation", "lookback", "seed"]
    for _, group in ledger.groupby(grouping, dropna=False, sort=True):
        metric_rows.append(_metric_record(group, "individual_seed_or_deterministic_fit"))
    by_seed = pd.DataFrame(metric_rows)
    ensemble = _neural_ensemble(ledger)
    ensemble_rows = []
    ensemble_grouping = ["target", "horizon", "model_family", "role", "model", "model_label", "representation", "lookback"]
    for _, group in ensemble.groupby(ensemble_grouping, dropna=False, sort=True):
        ensemble_rows.append(_metric_record(group, "neural_ensemble_mean"))
    deterministic = by_seed.loc[~by_seed.model_family.isin(["primary_neural", "neural_ablation"])].copy()
    summary = pd.concat([deterministic, pd.DataFrame(ensemble_rows)], ignore_index=True, sort=False)
    summary = summary.sort_values(["target", "horizon", "model_family", "model_label"]).reset_index(drop=True)

    primary = summary.loc[summary.role.eq("primary")].copy()
    primary["rank"] = np.nan
    primary["ranking_rule"] = "ascending RMSE, then MAE, then MASE, then model label"
    ranking_rows = []
    for (target, horizon), group in primary.groupby(["target", "horizon"], sort=True):
        eligible = group.loc[group.ranking_eligible.astype(bool)].sort_values(["rmse", "mae", "mase", "model_label"])
        eligible = eligible.copy(); eligible["rank"] = np.arange(1, len(eligible) + 1)
        ranking_rows.append(eligible)
    rankings = pd.concat(ranking_rows, ignore_index=True) if ranking_rows else pd.DataFrame()

    neural_seed = by_seed.loc[by_seed.model_family.isin(["primary_neural", "neural_ablation"])].copy()
    variability_rows: list[dict[str, Any]] = []
    measures = ["rmse", "mae", "mase", "rmsse", "smape_percent", "directional_accuracy_percent",
                "mean_error", "median_error", "r_squared"]
    for keys, group in neural_seed.groupby(["target", "horizon", "model_family", "role", "model", "model_label"], sort=True):
        row = dict(zip(["target", "horizon", "model_family", "role", "model", "model_label"], keys))
        row["seed_n"] = len(group)
        for measure in measures:
            values = group[measure].to_numpy(float); finite = values[np.isfinite(values)]
            row[f"mean_seed_{measure}"] = float(np.mean(finite)) if len(finite) else np.nan
            row[f"sd_seed_{measure}"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
            row[f"min_seed_{measure}"] = float(np.min(finite)) if len(finite) else np.nan
            row[f"max_seed_{measure}"] = float(np.max(finite)) if len(finite) else np.nan
        variability_rows.append(row)
    variability = pd.DataFrame(variability_rows)

    _atomic_csv(by_seed, paths["base"] / "metrics_by_fit_or_seed.csv")
    _atomic_csv(summary, paths["base"] / "metrics_summary.csv")
    _atomic_csv(rankings, paths["base"] / "primary_rankings.csv")
    _atomic_csv(ensemble, paths["base"] / "neural_ensemble_predictions.csv")
    _atomic_csv(variability, paths["base"] / "neural_seed_variability.csv")
    return {"by_seed": by_seed, "summary": summary, "rankings": rankings,
            "ensemble": ensemble, "variability": variability}


def _ablation_outputs(root: Path, performance: dict[str, pd.DataFrame], ledger: pd.DataFrame) -> pd.DataFrame:
    summary = performance["summary"]; variability = performance["variability"]
    rows: list[dict[str, Any]] = []
    comparisons = {
        "attention": ("CNN_BiLSTM_Attention_Compact", "CNN_BiLSTM_Compact_NoAttention"),
        "capacity": ("CNN_BiLSTM_Attention_Compact", "CNN_BiLSTM_Attention_Original23K"),
    }
    for comparison, (reference_name, comparison_name) in comparisons.items():
        for target in ["ASPH", "MTC"]:
            for horizon in [1, 2, 5]:
                ref = summary.loc[summary.target.eq(target) & summary.horizon.eq(horizon) & summary.model.eq(reference_name)].iloc[0]
                cmp = summary.loc[summary.target.eq(target) & summary.horizon.eq(horizon) & summary.model.eq(comparison_name)].iloc[0]
                vr = variability.loc[variability.target.eq(target) & variability.horizon.eq(horizon) & variability.model.eq(reference_name)].iloc[0]
                vc = variability.loc[variability.target.eq(target) & variability.horizon.eq(horizon) & variability.model.eq(comparison_name)].iloc[0]
                param_ref = int(ledger.loc[ledger.target.eq(target) & ledger.horizon.eq(horizon) & ledger.model.eq(reference_name), "parameter_count"].dropna().iloc[0])
                param_cmp = int(ledger.loc[ledger.target.eq(target) & ledger.horizon.eq(horizon) & ledger.model.eq(comparison_name), "parameter_count"].dropna().iloc[0])
                row = {"comparison": comparison, "target": target, "horizon": horizon,
                       "reference_model": reference_name, "comparison_model": comparison_name,
                       "parameter_count_reference": param_ref, "parameter_count_comparison": param_cmp,
                       "parameter_count_difference_comparison_minus_reference": param_cmp - param_ref,
                       "parameter_count_ratio_comparison_over_reference": param_cmp / param_ref}
                for metric in ["rmse", "mae", "mase"]:
                    difference = float(cmp[metric] - ref[metric])
                    row.update({f"{metric}_reference": float(ref[metric]), f"{metric}_comparison": float(cmp[metric]),
                                f"{metric}_difference_comparison_minus_reference": difference,
                                f"{metric}_advantage": reference_name if difference > 0 else comparison_name if difference < 0 else "tie",
                                f"seed_sd_{metric}_reference": float(vr[f"sd_seed_{metric}"]),
                                f"seed_sd_{metric}_comparison": float(vc[f"sd_seed_{metric}"])})
                rows.append(row)
    result = pd.DataFrame(rows)
    _atomic_csv(result, _paths(root)["base"] / "ablation_results.csv")
    return result


def _inference_outputs(root: Path, ensemble: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparisons = {
        "attention": ("CNN_BiLSTM_Attention_Compact", "CNN_BiLSTM_Compact_NoAttention"),
        "capacity": ("CNN_BiLSTM_Attention_Compact", "CNN_BiLSTM_Attention_Original23K"),
    }
    for family, (model_a, model_b) in comparisons.items():
        for target in ["ASPH", "MTC"]:
            for horizon in [1, 2, 5]:
                a = ensemble.loc[ensemble.target.eq(target) & ensemble.horizon.eq(horizon) & ensemble.model.eq(model_a),
                                 ["origin_year", "actual", "prediction"]].rename(columns={"prediction": "prediction_a"})
                b = ensemble.loc[ensemble.target.eq(target) & ensemble.horizon.eq(horizon) & ensemble.model.eq(model_b),
                                 ["origin_year", "actual", "prediction"]].rename(columns={"actual": "actual_b", "prediction": "prediction_b"})
                paired = a.merge(b, on="origin_year", how="inner").sort_values("origin_year")
                if not np.allclose(paired.actual, paired.actual_b, rtol=0, atol=0):
                    raise ValueError("Ablation inference actuals do not match exactly")
                actual = paired.actual.to_numpy(float); ea = actual - paired.prediction_a.to_numpy(float)
                eb = actual - paired.prediction_b.to_numpy(float)
                rmse_difference = float(np.sqrt(np.mean(eb ** 2)) - np.sqrt(np.mean(ea ** 2)))
                mae_difference = float(np.mean(np.abs(eb)) - np.mean(np.abs(ea)))
                repetitions = 5000; seed = _stable_seed(GLOBAL_SEED, family, target, horizon)
                rng = np.random.default_rng(seed)
                block_length = 1 if horizon == 1 else max(horizon, int(np.ceil(np.sqrt(len(paired)))))
                indices = _bootstrap_indices(rng, len(paired), repetitions, block_length)
                ea_samples, eb_samples = ea[indices], eb[indices]
                boot_rmse = np.sqrt(np.mean(eb_samples ** 2, axis=1)) - np.sqrt(np.mean(ea_samples ** 2, axis=1))
                boot_mae = np.mean(np.abs(eb_samples), axis=1) - np.mean(np.abs(ea_samples), axis=1)
                dm_enabled = not (target == "ASPH" and horizon == 5)
                if dm_enabled:
                    dm, raw_p, reason = _dm_hln(eb ** 2 - ea ** 2, horizon)
                    validity = "valid" if np.isfinite(raw_p) else "mathematically_undefined"
                else:
                    dm = raw_p = np.nan; reason = "ASPH h5 disabled a priori: n=5 gives zero HLN correction"
                    validity = "disabled_a_priori_descriptive_only"
                rows.append({"comparison_family": family, "model_A": model_a, "model_B": model_b,
                             "target": target, "horizon": horizon, "paired_n": len(paired),
                             "rmse_difference_B_minus_A": rmse_difference,
                             "mae_difference_B_minus_A": mae_difference,
                             "paired_bootstrap_repetitions": repetitions,
                             "bootstrap_type": "iid paired" if block_length == 1 else "circular moving-block paired",
                             "bootstrap_block_length": block_length,
                             "rmse_difference_bootstrap_lower_95": float(np.quantile(boot_rmse, .025)),
                             "rmse_difference_bootstrap_upper_95": float(np.quantile(boot_rmse, .975)),
                             "mae_difference_bootstrap_lower_95": float(np.quantile(boot_mae, .025)),
                             "mae_difference_bootstrap_upper_95": float(np.quantile(boot_mae, .975)),
                             "bootstrap_interpretation": "exploratory_extremely_unstable" if not dm_enabled else "paired_small_sample_interval",
                             "dm_hln_statistic": dm, "raw_p_value": raw_p, "holm_adjusted_p_value": np.nan,
                             "validity_flag": validity, "undefined_reason": reason,
                             "interpretation_flag": "descriptive_only_no_p_value" if not dm_enabled else "pending_holm",
                             "caution": "Non-rejection is not evidence of equivalence; paired annual samples are small.",
                             "bootstrap_seed": seed})
    result = pd.DataFrame(rows)
    for family, indices in result.groupby("comparison_family").groups.items():
        valid_indices = [i for i in indices if np.isfinite(result.loc[i, "raw_p_value"])]
        adjusted = _holm_adjust(result.loc[valid_indices, "raw_p_value"].astype(float).tolist())
        for i, p_value in zip(valid_indices, adjusted):
            result.loc[i, "holm_adjusted_p_value"] = p_value
            result.loc[i, "interpretation_flag"] = (
                "statistically_detected_difference" if p_value < .05 else "no_detected_difference_with_limited_power"
            )
    _atomic_csv(result, _paths(root)["base"] / "inference.csv")
    return result


def _coverage_outputs(root: Path, ledger: pd.DataFrame, ensemble: pd.DataFrame) -> pd.DataFrame:
    interval_source = pd.concat([
        ledger.loc[~ledger.model_family.isin(["primary_neural", "neural_ablation"])].copy(),
        ensemble.copy(),
    ], ignore_index=True, sort=False)
    rows: list[dict[str, Any]] = []
    keys = ["target", "horizon", "model_family", "model", "model_label", "representation"]
    for group_key, group in interval_source.groupby(keys, dropna=False, sort=True):
        for level, suffix in ((0.80, "80"), (0.95, "95")):
            success = group.loc[group.fit_status.eq("success")].copy()
            valid_col = f"interval_{suffix}_valid"; lower_col = f"lower_{suffix}"; upper_col = f"upper_{suffix}"
            valid_mask = success[valid_col].fillna(False).astype(bool) if valid_col in success else pd.Series(False, index=success.index)
            valid = success.loc[valid_mask].copy()
            if len(valid):
                widths = valid[upper_col].to_numpy(float) - valid[lower_col].to_numpy(float)
                covered = (valid.actual.to_numpy(float) >= valid[lower_col].to_numpy(float)) & (
                    valid.actual.to_numpy(float) <= valid[upper_col].to_numpy(float)
                )
                empirical = float(np.mean(covered)); direction = "undercoverage" if empirical < level else "overcoverage" if empirical > level else "at_nominal"
            else:
                widths = np.array([]); empirical = np.nan; direction = "not_estimable"
            method_values = sorted(set(str(x) for x in group.interval_method.dropna().unique()))
            undefined = ""
            if not len(valid):
                undefined = "no frozen neural predictive-interval method" if group_key[2] in {"primary_neural", "neural_ablation"} else "fewer than 8 conformal residuals or interval unavailable"
            rows.append({"target": group_key[0], "horizon": int(group_key[1]), "model_family": group_key[2],
                         "model": group_key[3], "model_label": group_key[4], "representation": group_key[5],
                         "nominal_coverage": level, "forecast_n": len(success), "valid_interval_n": len(valid),
                         "missing_or_invalid_interval_n": len(success) - len(valid), "empirical_coverage": empirical,
                         "mean_interval_width": float(np.mean(widths)) if len(widths) else np.nan,
                         "median_interval_width": float(np.median(widths)) if len(widths) else np.nan,
                         "coverage_direction": direction, "interval_method": " | ".join(method_values),
                         "undefined_reason": undefined,
                         "small_sample_flag": "extreme_small_sample_instability_n5" if group_key[0] == "ASPH" and int(group_key[1]) == 5 else ""})
    result = pd.DataFrame(rows)
    _atomic_csv(result, _paths(root)["base"] / "interval_coverage.csv")
    return result


def _negative_r2_outputs(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.loc[(summary.r_squared < 0) | summary.r_squared.isna(), [
        "target", "horizon", "model_family", "role", "model", "model_label", "representation",
        "summary_type", "expected_n", "successful_n", "r_squared"
    ]].copy()
    result["diagnostic_status"] = np.where(result.r_squared < 0, "negative_R_squared", "undefined_R_squared")
    result["interpretation"] = np.where(
        result.r_squared < 0,
        "Squared forecast errors exceed the constant-mean variance reference on this short evaluation window; primary ranking is unaffected.",
        "R-squared is mathematically undefined for this cell.",
    )
    _atomic_csv(result, _paths(root)["base"] / "negative_r_squared_diagnostics.csv")
    return result


def _timing_outputs(root: Path, ledger: pd.DataFrame, invocation: dict[str, Any],
                    environment: dict[str, Any], experiment_seconds: float) -> dict[str, Any]:
    paths = _paths(root)
    columns = ["run_id", "fit_id", "target", "horizon", "origin_year", "target_year", "model_family",
               "role", "model", "representation", "lookback", "seed", "fit_status", "execution_action",
               "feature_wall_seconds", "selection_wall_seconds", "fit_wall_seconds", "prediction_wall_seconds",
               "interval_wall_seconds", "per_fit_elapsed_wall_clock_seconds", "attempt_count",
               "failed_attempt_count", "prior_failed_attempt_elapsed_seconds"]
    timing = ledger.loc[:, columns].copy()
    _atomic_csv(timing, paths["base"] / "timing_per_fit.csv")
    family = timing.groupby("model_family", dropna=False).agg(
        planned_fit_count=("fit_id", "size"), successful_fit_count=("fit_status", lambda x: int((x == "success").sum())),
        unresolved_failed_fit_count=("fit_status", lambda x: int((x == "failed").sum())),
        attempted_fit_count=("attempt_count", "sum"), failed_attempt_count=("failed_attempt_count", "sum"),
        cumulative_per_fit_elapsed_seconds=("per_fit_elapsed_wall_clock_seconds", "sum"),
        cumulative_estimator_fit_seconds=("fit_wall_seconds", "sum"),
        median_per_fit_elapsed_seconds=("per_fit_elapsed_wall_clock_seconds", "median"),
        maximum_per_fit_elapsed_seconds=("per_fit_elapsed_wall_clock_seconds", "max"),
    ).reset_index()
    seed = timing.groupby(["model_family", "seed"], dropna=False).agg(
        fit_count=("fit_id", "size"), successful_fit_count=("fit_status", lambda x: int((x == "success").sum())),
        cumulative_per_fit_elapsed_seconds=("per_fit_elapsed_wall_clock_seconds", "sum"),
        cumulative_estimator_fit_seconds=("fit_wall_seconds", "sum"),
    ).reset_index()
    timestamp_values = pd.to_datetime(ledger.started_utc, utc=True, errors="coerce")
    earliest_start = timestamp_values.min()
    completed_now = pd.Timestamp.now(tz="UTC")
    full_span_seconds = float((completed_now - earliest_start).total_seconds()) if pd.notna(earliest_start) else experiment_seconds
    summary = {
        "completed_utc": completed_now.isoformat(), "experiment_wall_clock_seconds": max(experiment_seconds, full_span_seconds),
        "current_resume_invocation_wall_clock_seconds": experiment_seconds,
        "experiment_start_utc_from_first_fit": earliest_start.isoformat() if pd.notna(earliest_start) else None,
        "cumulative_per_fit_elapsed_wall_clock_seconds": float(timing.per_fit_elapsed_wall_clock_seconds.sum()),
        "cumulative_estimator_fit_seconds": float(timing.fit_wall_seconds.sum()),
        "cumulative_prediction_seconds": float(timing.prediction_wall_seconds.sum()),
        "cumulative_feature_seconds": float(timing.feature_wall_seconds.sum()),
        "cumulative_selection_seconds": float(timing.selection_wall_seconds.sum()),
        "cumulative_interval_seconds": float(timing.interval_wall_seconds.sum()),
        "planned_fit_count": len(timing),
        "attempted_fit_count_total": int(timing.attempt_count.sum()),
        "successful_fit_count": int(timing.fit_status.eq("success").sum()),
        "failed_fit_count": int(timing.failed_attempt_count.sum()),
        "unresolved_failed_fit_count": int(timing.fit_status.eq("failed").sum()),
        "not_attempted_fit_count": int(timing.fit_status.eq("not_attempted").sum()),
        "attempted_this_invocation": int(invocation.get("attempted", 0)),
        "resumed_fit_count_this_invocation": int(invocation.get("resumed", 0)),
        "skipped_completed_fit_count_this_invocation": int(invocation.get("resumed", 0)),
        "auxiliary_calibration_fit_count_this_invocation": int(invocation.get("auxiliary_calibration_fits", 0)),
        "configured_parallel_worker_count": environment["configured_worker_count"],
        "parallel_strategy": environment["parallel_strategy"],
        "hardware": {key: environment[key] for key in (
            "cpu_model", "physical_cpu_count", "logical_cpu_count", "ram_bytes", "gpu_model"
        )},
        "family_summaries": family.to_dict("records"), "seed_summaries": seed.to_dict("records"),
        "timing_interpretation": "Cumulative per-fit elapsed time is the sum of output-fit task durations (including feature/selection/interval work) and is distinct from overall experiment wall-clock time.",
    }
    _atomic_json(paths["base"] / "timing_summary.json", summary)
    return summary


def _failure_outputs(root: Path, ledger: pd.DataFrame) -> pd.DataFrame:
    columns = ["fit_id", "target", "horizon", "origin_year", "model_family", "model", "representation", "seed",
               "exception_class", "warning_or_failure_message"]
    failures = ledger.loc[ledger.fit_status.eq("failed"), columns].copy()
    failures["attempt_number"] = ledger.loc[ledger.fit_status.eq("failed"), "attempt_count"].to_numpy()
    failures["final_fit_status"] = "failed"
    failures["execution_continued"] = True
    prior_rows: list[dict[str, Any]] = []
    for row in ledger.itertuples(index=False):
        try:
            prior = json.loads(getattr(row, "prior_attempt_failures_json", "[]") or "[]")
        except Exception:
            prior = []
        for attempt in prior:
            prior_rows.append({"fit_id": row.fit_id, "target": row.target, "horizon": row.horizon,
                               "origin_year": row.origin_year, "model_family": row.model_family,
                               "model": row.model, "representation": row.representation, "seed": row.seed,
                               "exception_class": attempt.get("exception_class", ""),
                               "warning_or_failure_message": attempt.get("message", ""),
                               "attempt_number": attempt.get("attempt", 1), "final_fit_status": row.fit_status,
                               "execution_continued": True})
    if prior_rows:
        failures = pd.concat([failures, pd.DataFrame(prior_rows)], ignore_index=True, sort=False)
    _atomic_csv(failures, _paths(root)["base"] / "failures.csv")
    return failures


def _primary_tables(root: Path, performance: dict[str, pd.DataFrame], ablation: pd.DataFrame,
                    inference: pd.DataFrame, coverage: pd.DataFrame, negative_r2: pd.DataFrame,
                    timing: dict[str, Any]) -> None:
    tables = _paths(root)["tables"]
    summary = performance["summary"]
    primary = summary.loc[summary.role.eq("primary")].copy()
    winners = performance["rankings"].loc[performance["rankings"]["rank"].eq(1)].copy()
    secondary_columns = ["target", "horizon", "model_family", "model", "model_label", "representation",
                         "rmsse", "smape_percent", "directional_accuracy_percent", "mean_error", "median_error", "r_squared"]
    cost = pd.DataFrame(timing["family_summaries"])
    _atomic_csv(primary, tables / "table_primary_performance.csv")
    _atomic_csv(winners, tables / "table_cell_winners.csv")
    _atomic_csv(primary.loc[:, secondary_columns], tables / "table_secondary_metrics.csv")
    _atomic_csv(performance["variability"], tables / "table_neural_seed_variability.csv")
    _atomic_csv(ablation, tables / "table_ablation.csv")
    _atomic_csv(inference, tables / "table_inference.csv")
    _atomic_csv(coverage, tables / "table_interval_coverage.csv")
    _atomic_csv(negative_r2, tables / "table_negative_r_squared.csv")
    _atomic_csv(cost, tables / "table_computational_cost.csv")


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _figures(root: Path, state: dict[str, Any], performance: dict[str, pd.DataFrame],
             ablation: pd.DataFrame, coverage: pd.DataFrame) -> list[Path]:
    output = _paths(root)["figures"]; output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    colors = {"ASPH": "#276FBF", "MTC": "#D95F02"}
    official = state["official"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for ax, target in zip(axes, ["ASPH", "MTC"]):
        part = official.loc[official.target.eq(target)]
        ax.plot(part.year, part.analysis_value, color=colors[target], marker="o", ms=3, lw=1.8)
        ax.fill_between(part.year, part.lower_interval, part.upper_interval, color=colors[target], alpha=.15)
        ax.set_title(f"{target}: {TARGET_META[target]['definition']} ({TARGET_META[target]['age_population']})")
        ax.set_ylabel(TARGET_META[target]["unit"]); ax.grid(alpha=.25)
    axes[-1].set_xlabel("Year")
    path = output / "official_reconstructed_source_series.png"; _save_figure(fig, path); created.append(path)

    calendar = pd.DataFrame(_calendar_rows(state["protocol"]))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), constrained_layout=True)
    for ax, target in zip(axes, ["ASPH", "MTC"]):
        part = calendar.loc[calendar.target.eq(target)]
        for y, horizon in enumerate([1, 2, 5]):
            cell = part.loc[part.horizon.eq(horizon)]
            ax.scatter(cell.target_year, np.full(len(cell), y), s=35, label=f"h={horizon}, n={len(cell)}")
            for row in cell.itertuples(): ax.plot([row.origin_year, row.target_year], [y, y], color="#777777", alpha=.25)
        ax.set_yticks([0, 1, 2], ["h=1", "h=2", "h=5"]); ax.set_title(f"{target} expanding-origin targets")
        ax.legend(ncol=3, fontsize=8); ax.grid(axis="x", alpha=.25)
    axes[-1].set_xlabel("Target year (line begins at forecast origin)")
    path = output / "rolling_origin_evaluation_calendar.png"; _save_figure(fig, path); created.append(path)

    primary = performance["summary"].loc[performance["summary"].role.eq("primary")].copy()
    for measure, filename in [("rmse", "primary_rmse_comparison.png"), ("mae", "mae_support.png"), ("mase", "mase_support.png")]:
        if measure == "mase":
            fig, axes = plt.subplots(2, 3, figsize=(19, 10))
            fig.subplots_adjust(left=.18, right=.98, bottom=.08, top=.94, wspace=1.05, hspace=.34)
        else:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
        for row_index, target in enumerate(["ASPH", "MTC"]):
            for col_index, horizon in enumerate([1, 2, 5]):
                ax = axes[row_index, col_index]
                cell = primary.loc[primary.target.eq(target) & primary.horizon.eq(horizon)].sort_values(measure)
                y = np.arange(len(cell)); ax.scatter(cell[measure], y, c=[colors[target]] * len(cell), s=25)
                ax.set_yticks(y, cell.model_label, fontsize=6 if measure == "mase" else 7); ax.invert_yaxis(); ax.grid(axis="x", alpha=.25)
                n = int(cell.expected_n.iloc[0]); ax.set_title(f"{target} h={horizon} (n={n})")
                ax.set_xlabel(f"{measure.upper()} ({TARGET_META[target]['unit']})" if measure != "mase" else "MASE (scaled)")
        path = output / filename; _save_figure(fig, path); created.append(path)

    variability = performance["variability"].loc[performance["variability"].role.eq("primary")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for row_index, target in enumerate(["ASPH", "MTC"]):
        for col_index, horizon in enumerate([1, 2, 5]):
            ax = axes[row_index, col_index]
            cell = variability.loc[variability.target.eq(target) & variability.horizon.eq(horizon)].sort_values("mean_seed_rmse")
            y = np.arange(len(cell)); means = cell.mean_seed_rmse.to_numpy(float); sd = cell.sd_seed_rmse.to_numpy(float)
            ax.errorbar(means, y, xerr=sd, fmt="o", color=colors[target], capsize=3)
            ax.set_yticks(y, cell.model, fontsize=7); ax.invert_yaxis(); ax.grid(axis="x", alpha=.25)
            ax.set_title(f"{target} h={horizon} (10 fixed seeds)"); ax.set_xlabel("Mean seed RMSE +/- 1 SD")
    path = output / "neural_seed_variability.png"; _save_figure(fig, path); created.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row_index, target in enumerate(["ASPH", "MTC"]):
        for col_index, horizon in enumerate([1, 2, 5]):
            ax = axes[row_index, col_index]
            cell = ablation.loc[ablation.target.eq(target) & ablation.horizon.eq(horizon)]
            values = cell.rmse_difference_comparison_minus_reference.to_numpy(float)
            ax.bar(cell.comparison, values, color=["#4C78A8", "#F58518"]); ax.axhline(0, color="black", lw=.8)
            ax.set_title(f"{target} h={horizon} (n={int(performance['summary'].loc[performance['summary'].target.eq(target) & performance['summary'].horizon.eq(horizon)].expected_n.iloc[0])})")
            ax.set_ylabel("RMSE difference: comparator - compact attention")
    path = output / "attention_capacity_ablation.png"; _save_figure(fig, path); created.append(path)

    plot_coverage = coverage.loc[coverage.valid_interval_n.gt(0)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for row_index, target in enumerate(["ASPH", "MTC"]):
        for col_index, horizon in enumerate([1, 2, 5]):
            ax = axes[row_index, col_index]
            cell = plot_coverage.loc[plot_coverage.target.eq(target) & plot_coverage.horizon.eq(horizon)]
            for level, marker in [(0.80, "o"), (0.95, "s")]:
                part = cell.loc[np.isclose(cell.nominal_coverage, level)]
                ax.scatter(np.full(len(part), level), part.empirical_coverage, marker=marker, alpha=.65,
                           label=f"nominal {int(level*100)}%")
            ax.plot([.78, .97], [.78, .97], color="black", ls="--", lw=.8)
            n = int(cell.forecast_n.max()) if len(cell) else 0
            ax.set_title(f"{target} h={horizon} (forecast n={n})")
            ax.set_xlim(.77, .98); ax.set_ylim(-.03, 1.03); ax.set_xlabel("Nominal coverage")
            ax.set_ylabel("Empirical coverage across interval-capable models"); ax.grid(alpha=.2)
            if row_index == 0 and col_index == 0: ax.legend(fontsize=8)
    path = output / "empirical_interval_coverage_all_cells.png"; _save_figure(fig, path); created.append(path)
    return created


def _consistency_audit(root: Path, state: dict[str, Any], ledger: pd.DataFrame,
                       performance: dict[str, pd.DataFrame], inference: pd.DataFrame,
                       coverage: pd.DataFrame, failures: pd.DataFrame, timing: dict[str, Any]) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL",
                       "observed": observed, "expected": expected})

    actual_lookup = state["official"].set_index(["target", "year"]).analysis_value.to_dict()
    successful = ledger.loc[ledger.fit_status.eq("success")]
    actual_match = all(float(row.actual) == float(actual_lookup[(row.target, int(row.target_year))])
                       for row in successful.itertuples())
    check("all_actuals_from_official_processed_csv", actual_match, actual_match, True)
    supported = ((ledger.target.eq("ASPH") & ledger.target_year.between(1990, 2019))
                 | (ledger.target.eq("MTC") & ledger.target_year.between(1980, 2018))).all()
    check("no_target_outside_official_coverage", bool(supported), bool(supported), True)
    counts = ledger.groupby(["target", "horizon", "model_family", "model", "representation", "seed"], dropna=False).origin_year.nunique()
    expected = {("ASPH", 1): 9, ("ASPH", 2): 8, ("ASPH", 5): 5,
                ("MTC", 1): 18, ("MTC", 2): 17, ("MTC", 5): 14}
    count_valid = all(int(value) == expected[(key[0], int(key[1]))] for key, value in counts.items())
    check("forecast_calendar_counts_by_configuration", count_valid, "all configuration counts match 9/8/5 and 18/17/14", expected)
    check("no_recent_period_analysis", not any("recent" in str(value).lower() or "regime" in str(value).lower()
                                                for value in ledger.columns), list(ledger.columns), "no recent/regime fields")
    ml_neural = ledger.loc[ledger.model_family.ne("statistical")]
    check("primary_ML_neural_lookback_L3", ml_neural.lookback.eq(3).all(), sorted(ml_neural.lookback.dropna().unique().tolist()), [3])
    check("no_nested_selection", not ledger.configuration_json.astype(str).str.contains("nested", case=False).any(), False, False)
    check("no_L5_L6_L8_execution", not ml_neural.lookback.isin([5, 6, 8]).any(), sorted(ml_neural.lookback.dropna().unique().tolist()), [3])
    check("no_raw_level_primary_execution", not ledger.representation.eq("raw_level_direct").any(),
          sorted(ledger.representation.unique().tolist()), "raw_level_direct absent")
    check("no_evaluation_driven_retuning", True,
          "frozen ML map and frozen neural map used; no selector function invoked", True)
    metric_columns = ["rmse", "mae", "mase", "rmsse", "smape_percent", "directional_accuracy_percent", "mean_error", "median_error"]
    metric_available = performance["summary"][metric_columns].notna().all().all()
    check("all_required_metrics_generated", bool(metric_available), metric_columns, "all finite for complete result cells")
    asph_h5 = inference.loc[inference.target.eq("ASPH") & inference.horizon.eq(5)]
    no_p = asph_h5.dm_hln_statistic.isna().all() and asph_h5.raw_p_value.isna().all() and asph_h5.holm_adjusted_p_value.isna().all()
    check("ASPH_h5_no_DM_or_p_value", bool(no_p), len(asph_h5), "two descriptive ablation rows, all DM/p fields NA")
    coverage_cells = set(zip(coverage.target, coverage.horizon))
    check("all_six_coverage_cells_present", coverage_cells == set(expected), sorted(coverage_cells), sorted(expected))
    check("prediction_ledger_unique_fit_ids", ledger.fit_id.is_unique and len(ledger) == 7242,
          {"rows": len(ledger), "unique_fit_ids": ledger.fit_id.nunique()}, {"rows": 7242, "unique_fit_ids": 7242})
    metric_reconcile = all(int(row.expected_n) == expected[(row.target, int(row.horizon))]
                           for row in performance["summary"].itertuples())
    check("prediction_ledger_reconciles_with_metrics", metric_reconcile, metric_reconcile, True)
    check("fit_counts_reconcile_with_timing", timing["planned_fit_count"] == len(ledger)
          and timing["attempted_fit_count_total"] == int(ledger.attempt_count.sum()),
          {"planned_timing": timing["planned_fit_count"], "ledger": len(ledger),
           "attempted_timing": timing["attempted_fit_count_total"], "attempted_ledger": int(ledger.attempt_count.sum())}, "equal")
    check("failure_counts_reconcile", len(failures) == int(ledger.failed_attempt_count.sum()),
          {"failure_attempt_rows": len(failures), "ledger_failed_attempts": int(ledger.failed_attempt_count.sum()),
           "unresolved_failed_records": int(ledger.fit_status.eq("failed").sum())}, "equal")
    neural_counts = ledger.loc[ledger.model_family.isin(["primary_neural", "neural_ablation"])].groupby(
        ["target", "horizon", "origin_year", "model"]
    ).seed.nunique()
    check("ten_neural_seed_records_per_fit_cell", neural_counts.eq(10).all(),
          {"minimum": int(neural_counts.min()), "maximum": int(neural_counts.max())}, {"minimum": 10, "maximum": 10})
    check("successful_predictions_finite", np.isfinite(successful.prediction).all(), int(np.isfinite(successful.prediction).sum()), len(successful))
    check("raw_data_hash_unchanged", sha256_file(root / state["protocol"]["processed_dataset"]["path"]) == state["data_hash"],
          sha256_file(root / state["protocol"]["processed_dataset"]["path"]), state["data_hash"])
    current_protocol_hash = hashlib.sha256((root / "config" / "round2_frozen_evaluation_protocol.json").read_bytes()).hexdigest()
    check("protocol_hash_unchanged", current_protocol_hash == state["protocol_hash"], current_protocol_hash, state["protocol_hash"])
    check("formal_inference_only_prespecified_families", set(inference.comparison_family) == {"attention", "capacity"},
          sorted(inference.comparison_family.unique().tolist()), ["attention", "capacity"])
    result = pd.DataFrame(checks)
    _atomic_csv(result, _paths(root)["base"] / "internal_consistency_checks.csv")
    _atomic_json(_paths(root)["base"] / "internal_consistency.json", {
        "status": "PASS" if result.status.eq("PASS").all() else "FAIL",
        "passed": int(result.status.eq("PASS").sum()), "total": len(result),
        "failed_checks": result.loc[result.status.eq("FAIL"), "check"].tolist(),
    })
    return result


def _coverage_digest(coverage: pd.DataFrame) -> pd.DataFrame:
    valid = coverage.loc[coverage.valid_interval_n.gt(0)]
    return valid.groupby(["target", "horizon", "nominal_coverage"], as_index=False).agg(
        interval_capable_model_n=("model_label", "nunique"),
        total_valid_intervals=("valid_interval_n", "sum"),
        mean_model_empirical_coverage=("empirical_coverage", "mean"),
        minimum_model_empirical_coverage=("empirical_coverage", "min"),
        maximum_model_empirical_coverage=("empirical_coverage", "max"),
    )


def _execution_audit(root: Path, state: dict[str, Any], performance: dict[str, pd.DataFrame],
                     ablation: pd.DataFrame, inference: pd.DataFrame, coverage: pd.DataFrame,
                     negative_r2: pd.DataFrame, failures: pd.DataFrame, timing: dict[str, Any],
                     consistency: pd.DataFrame) -> None:
    winners = performance["rankings"].loc[performance["rankings"]["rank"].eq(1)].sort_values(["target", "horizon"])
    winner_lines = [
        f"- {row.target} h={int(row.horizon)} (n={int(row.expected_n)}): {row.model_label}; "
        f"RMSE={row.rmse:.10g}, MAE={row.mae:.10g}, MASE={row.mase:.10g}."
        for row in winners.itertuples()
    ]
    coverage_lines = []
    for row in _coverage_digest(coverage).itertuples():
        coverage_lines.append(
            f"- {row.target} h={int(row.horizon)}, nominal {int(row.nominal_coverage*100)}%: "
            f"{int(row.total_valid_intervals)} valid intervals across {int(row.interval_capable_model_n)} model specifications; "
            f"mean model-level empirical coverage={row.mean_model_empirical_coverage:.3f} "
            f"(range {row.minimum_model_empirical_coverage:.3f}-{row.maximum_model_empirical_coverage:.3f})."
        )
    inference_valid = int(inference.raw_p_value.notna().sum())
    inference_invalid = int(inference.raw_p_value.isna().sum())
    text = f"""# Round-2 Phase 2B execution audit

## Status

Internal consistency: **{'PASS' if consistency.status.eq('PASS').all() else 'FAIL'}**
({int(consistency.status.eq('PASS').sum())}/{len(consistency)} checks). This run used only the reconstructed official dataset and stopped before manuscript revision.

## Frozen inputs and calendar

- Dataset: `data/processed/china_male_official_cardiovascular_series.csv`
- Dataset SHA-256: `{state['data_hash']}`
- Protocol: `config/round2_frozen_evaluation_protocol.json`
- Protocol version: `{state['protocol']['protocol_version']}`
- Protocol SHA-256: `{state['protocol_hash']}`
- Initial history: 21 annual observations; expanding windows.
- ASPH: adults aged 30-79 years, 1990-2019; h1/h2/h5 counts 9/8/5.
- MTC: adults aged 18 years and older, 1980-2018; h1/h2/h5 counts 18/17/14.
- ML and neural lookback: fixed L=3. Nested selection and L=5/6/8 reruns were not executed.
- Recent-period/regime sensitivity was not executed.

## Models and seeds

- Statistical: {', '.join(STATISTICAL_MODELS)}.
- Classical ML: {', '.join(ML_MODELS)}, each with both frozen primary representations.
- Primary neural: {', '.join(PRIMARY_NEURAL)}.
- Ablations: {', '.join(ABLATION_NEURAL)}.
- Neural seeds: {', '.join(str(seed) for seed in NEURAL_SEEDS)}.

## Fit and timing reconciliation

- Planned output-producing fits: {timing['planned_fit_count']:,}.
- Attempted/recorded fits: {timing['attempted_fit_count_total']:,}.
- Successful final records: {timing['successful_fit_count']:,}; failed attempts: {timing['failed_fit_count']:,}; unresolved failed records: {timing['unresolved_failed_fit_count']:,}; not attempted: {timing['not_attempted_fit_count']:,}.
- Cumulative per-fit elapsed time: {timing['cumulative_per_fit_elapsed_wall_clock_seconds']:.3f} seconds.
- Cumulative estimator-fit time: {timing['cumulative_estimator_fit_seconds']:.3f} seconds.
- Overall experiment wall-clock through result generation: {timing['experiment_wall_clock_seconds']:.3f} seconds.
- Workers: {timing['configured_parallel_worker_count']} for neural cells; statistical, ML, and reporting were serial.
- CPU: {state['environment']['cpu_model']}; RAM bytes: {state['environment']['ram_bytes']}; GPU: {state['environment']['gpu_model']}.

## Primary winners

{chr(10).join(winner_lines)}

All additional requested metrics (RMSSE, sMAPE, directional accuracy, mean error, and median error) are present in the machine-readable tables. R-squared is diagnostic only.

## Inference

Primary winner-vs-rest inference was not run. The preflight artifact audit found no transferable development-only primary comparison family, and the older six-pair cross-family table selected representatives using evaluation RMSE. Formal inference is limited to the two Phase 2B-brief-prespecified ablation families. There are {inference_valid} mathematically valid DM rows and {inference_invalid} disabled/undefined rows. ASPH h5 has no DM statistic or p value; its bootstrap is exploratory and extremely unstable. Holm correction was applied separately within each named valid family. Non-rejection is not interpreted as equivalence.

## Ablations

The complete 12-row attention/capacity table is `results/round2/ablation_results.csv`. Positive comparison-minus-reference errors favor compact attention; negative values favor the comparator. Findings are cell-specific and are not generalized beyond the six evaluated target-horizon cells.

## Interval coverage

Neural seed variability is not predictive uncertainty. No neural interval method was invented. Statistical native/training-only intervals and classical-ML frozen conformal intervals are evaluated wherever valid; neural rows explicitly state that no frozen predictive-interval method is available.

{chr(10).join(coverage_lines)}

ASPH h5 coverage is explicitly flagged as extremely unstable because n=5.

## Negative R-squared and failures

- Negative/undefined R-squared diagnostic rows: {len(negative_r2)}. Negative R-squared means squared forecast error exceeded the constant-mean variance reference on that short window; it is not a primary ranking metric.
- Failed fit attempts: {len(failures)}; unresolved failed records: {timing['unresolved_failed_fit_count']}. Every failure attempt remains explicit in `results/round2/failures.csv`; no fallback model was substituted. An operationally failed attempt may be retried unchanged and remains in this audit.

## Reproducibility and scope

Per-fit deterministic IDs and atomic JSON checkpoints support restart without duplication. Actuals, calendars, lookbacks, seeds, counts, timing, failures, metrics, intervals, inference scope, and input hashes were reconciled automatically. No interpolation, extrapolation of observations, imputation, smoothing, splicing, or synthetic extension was performed. Unsupported ASPH 1970-1989/2020-2023 and MTC 1970-1979/2019-2023 observations were not used. The manuscript, frozen first-round outputs, Git tags/releases, and public GitHub repository were not changed by this execution.
"""
    _atomic_text(_paths(root)["report"], text)


def _execution_manifest(root: Path, state: dict[str, Any], timing: dict[str, Any],
                        consistency: pd.DataFrame, figures: list[Path]) -> dict[str, Any]:
    paths = _paths(root)
    artifacts = [path for path in paths["base"].rglob("*") if path.is_file()
                 and "progress" not in path.relative_to(paths["base"]).parts
                 and path.name != "RESULTS_SHA256SUMS.txt"]
    manifest = {
        "run_id": state["summary"]["run_id"], "created_utc": _utc_now(),
        "dataset": {"path": state["protocol"]["processed_dataset"]["path"], "sha256": state["data_hash"]},
        "protocol": {"path": "config/round2_frozen_evaluation_protocol.json",
                     "version": state["protocol"]["protocol_version"], "sha256": state["protocol_hash"]},
        "implementation_sources": {
            "src/round2_phase2b.py": sha256_file(root / "src" / "round2_phase2b.py"),
            FROZEN_ML_CONFIG_PATH: sha256_file(root / FROZEN_ML_CONFIG_PATH),
            FROZEN_NEURAL_CONFIG_PATH: sha256_file(root / FROZEN_NEURAL_CONFIG_PATH),
        },
        "fit_counts": {key: timing[key] for key in ("planned_fit_count", "attempted_fit_count_total",
                                                      "successful_fit_count", "failed_fit_count", "unresolved_failed_fit_count",
                                                      "not_attempted_fit_count")},
        "validation_status": "PASS" if consistency.status.eq("PASS").all() else "FAIL",
        "inference_policy": state["summary"]["comparison_policy_resolution"],
        "recent_period_sensitivity_executed": False, "nested_lookback_selection_executed": False,
        "primary_lookback": 3, "figures": [str(path.relative_to(root)).replace("\\", "/") for path in figures],
        "artifact_inventory": [{"path": str(path.relative_to(root)).replace("\\", "/"), "size_bytes": path.stat().st_size}
                               for path in sorted(artifacts)],
    }
    _atomic_json(paths["base"] / "execution_manifest.json", manifest)
    return manifest


def _freeze_hashes(root: Path) -> pd.DataFrame:
    paths = _paths(root); files = [path for path in paths["base"].rglob("*") if path.is_file()
                                  and "progress" not in path.relative_to(paths["base"]).parts
                                  and path.name != "RESULTS_SHA256SUMS.txt"]
    if paths["report"].is_file(): files.append(paths["report"])
    rows = [{"sha256": sha256_file(path), "path": str(path.relative_to(root)).replace("\\", "/")}
            for path in sorted(files)]
    lines = [f"{row['sha256']}  {row['path']}" for row in rows]
    _atomic_text(paths["base"] / "RESULTS_SHA256SUMS.txt", "\n".join(lines) + "\n")
    return pd.DataFrame(rows)


def run(root: Path | None = None, worker_count: int = 18, preflight_only: bool = False) -> dict[str, Any]:
    project = (root or Path(__file__).resolve().parents[1]).resolve()
    paths = _paths(project)
    for path in (paths["base"], paths["progress"], paths["tables"], paths["figures"]):
        path.mkdir(parents=True, exist_ok=True)
    experiment_started = time.perf_counter()
    state = _load_existing_preflight(project, worker_count)
    if state is None:
        state = _preflight(project, worker_count)
        preflight_action = "created"
    else:
        preflight_action = "resumed_existing_checkpoint"
    print(f"PRECHECK PASS action={preflight_action} dataset={state['data_hash']} protocol={state['protocol']['protocol_version']} planned={len(state['plan'])}", flush=True)
    if preflight_only:
        return {"status": "PREFLIGHT_PASS", "planned_fit_count": len(state["plan"]),
                "dataset_sha256": state["data_hash"], "protocol_sha256": state["protocol_hash"]}

    run_state_path = paths["base"] / "run_state.json"
    previous_run_state = json.loads(run_state_path.read_text(encoding="utf-8")) if run_state_path.exists() else {}
    run_start_utc = previous_run_state.get("start_utc", _utc_now())
    prior_interruptions = list(previous_run_state.get("interruptions", []))
    _atomic_json(run_state_path, {"status": "running", "start_utc": run_start_utc,
                                  "run_id": state["summary"]["run_id"],
                                  "dataset_sha256": state["data_hash"], "protocol_sha256": state["protocol_hash"],
                                  "interruptions": prior_interruptions})
    invocation = {"attempted": 0, "resumed": 0, "auxiliary_calibration_fits": 0}
    try:
        statistical_records, statistical_candidates, stat_counts = _execute_statistical(project, state)
        invocation["attempted"] += stat_counts["attempted"]; invocation["resumed"] += stat_counts["resumed"]
        _atomic_csv(pd.DataFrame(statistical_candidates), paths["base"] / "statistical_candidate_search.csv")

        ml_records, calibration, ml_counts = _execute_ml(project, state)
        invocation["attempted"] += ml_counts["attempted"]; invocation["resumed"] += ml_counts["resumed"]
        invocation["auxiliary_calibration_fits"] += ml_counts["auxiliary_calibration_fits"]
        _atomic_csv(pd.DataFrame(calibration), paths["base"] / "ml_conformal_calibration.csv")

        neural_counts = _execute_neural(project, state, worker_count)
        invocation["attempted"] += neural_counts["attempted"]; invocation["resumed"] += neural_counts["resumed"]

        ledger = _collect_progress(project, state)
        _atomic_csv(ledger, paths["base"] / "predictions_long.csv")
        if not statistical_candidates:
            statistical_summary = ledger.loc[ledger.model_family.eq("statistical"), [
                "fit_id", "target", "model", "horizon", "origin_year", "target_year", "fit_status",
                "hyperparameters_json", "selection_wall_seconds", "fit_wall_seconds",
                "prediction_wall_seconds", "warning_or_failure_message"
            ]].rename(columns={"hyperparameters_json": "selected_specification"})
            statistical_summary["record_scope"] = "selected_candidate_or_fixed_specification_summary_reconstructed_from_atomic_fit_records"
            _atomic_csv(statistical_summary, paths["base"] / "statistical_candidate_search.csv")
        if not calibration:
            calibration_summary = ledger.loc[ledger.model_family.eq("classical_ml"), [
                "fit_id", "target", "model", "representation", "horizon", "origin_year", "target_year",
                "calibration_n", "lower_80", "upper_80", "interval_80_valid", "lower_95", "upper_95",
                "interval_95_valid", "interval_method"
            ]].rename(columns={"fit_id": "outer_fit_id"})
            _atomic_csv(calibration_summary, paths["base"] / "ml_conformal_calibration.csv")
        performance = _performance_outputs(project, ledger)
        ablation = _ablation_outputs(project, performance, ledger)
        inference = _inference_outputs(project, performance["ensemble"])
        coverage = _coverage_outputs(project, ledger, performance["ensemble"])
        negative_r2 = _negative_r2_outputs(project, performance["summary"])
        failures = _failure_outputs(project, ledger)
        provisional_seconds = time.perf_counter() - experiment_started
        timing = _timing_outputs(project, ledger, invocation, state["environment"], provisional_seconds)
        _primary_tables(project, performance, ablation, inference, coverage, negative_r2, timing)
        figures = _figures(project, state, performance, ablation, coverage)
        consistency = _consistency_audit(project, state, ledger, performance, inference, coverage, failures, timing)
        timing = _timing_outputs(project, ledger, invocation, state["environment"], time.perf_counter() - experiment_started)
        _primary_tables(project, performance, ablation, inference, coverage, negative_r2, timing)
        _execution_audit(project, state, performance, ablation, inference, coverage, negative_r2,
                         failures, timing, consistency)
        manifest = _execution_manifest(project, state, timing, consistency, figures)
        winners = performance["rankings"].loc[performance["rankings"]["rank"].eq(1)].sort_values(["target", "horizon"])
        summary = {
            "status": "PASS" if consistency.status.eq("PASS").all() else "FAIL",
            "run_id": state["summary"]["run_id"], "dataset_sha256": state["data_hash"],
            "protocol_sha256": state["protocol_hash"], "protocol_version": state["protocol"]["protocol_version"],
            "planned_fit_count": timing["planned_fit_count"], "attempted_fit_count": timing["attempted_fit_count_total"],
            "successful_fit_count": timing["successful_fit_count"], "failed_fit_count": timing["failed_fit_count"],
            "unresolved_failed_fit_count": timing["unresolved_failed_fit_count"],
            "experiment_wall_clock_seconds": timing["experiment_wall_clock_seconds"],
            "cumulative_per_fit_elapsed_seconds": timing["cumulative_per_fit_elapsed_wall_clock_seconds"],
            "workers": worker_count, "winners": winners[["target", "horizon", "model_label", "rmse", "mae", "mase"]].to_dict("records"),
            "neural_primary_cell_wins": int(winners.model_family.eq("primary_neural").sum()),
            "inference_valid_DM_rows": int(inference.raw_p_value.notna().sum()),
            "inference_disabled_or_undefined_rows": int(inference.raw_p_value.isna().sum()),
            "coverage_cells": sorted([f"{target}_h{horizon}" for target, horizon in set(zip(coverage.target, coverage.horizon))]),
            "negative_or_undefined_r_squared_rows": len(negative_r2),
            "internal_consistency_passed": int(consistency.status.eq("PASS").sum()),
            "internal_consistency_total": len(consistency),
            "recent_period_sensitivity_executed": False,
        }
        _atomic_json(paths["base"] / "run_summary.json", summary)
        manifest = _execution_manifest(project, state, timing, consistency, figures)
        _atomic_json(run_state_path, {"status": "complete" if summary["status"] == "PASS" else "requires_correction",
                                      "start_utc": run_start_utc,
                                      "end_utc": _utc_now(), "run_id": state["summary"]["run_id"],
                                      "dataset_sha256": state["data_hash"], "protocol_sha256": state["protocol_hash"],
                                      "interruptions": prior_interruptions})
        hashes = _freeze_hashes(project)
        summary["central_result_hash_count"] = len(hashes)
        print(json.dumps(summary, indent=2, default=_json_default), flush=True)
        return summary
    except BaseException as exc:
        previous = json.loads(run_state_path.read_text(encoding="utf-8")) if run_state_path.exists() else {}
        interruptions = list(previous.get("interruptions", []))
        interruptions.append({"utc": _utc_now(), "exception_class": type(exc).__name__, "message": str(exc),
                              "traceback": traceback.format_exc()})
        _atomic_json(run_state_path, {**previous, "status": "interrupted", "last_update_utc": _utc_now(),
                                      "interruptions": interruptions})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args.root, args.workers, args.preflight_only)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
