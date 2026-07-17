"""Computational-cost, failure, and runtime-environment tables."""

from __future__ import annotations

import ctypes
import os
import platform
import sys

import numpy as np
import pandas as pd


def aggregate_computational_cost(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate measured per-record timing components by requested grouping."""

    rows = []
    groups = ["analysis_window", "target", "model", "horizon"]
    for keys, group in forecasts.groupby(groups, sort=True):
        total = group["total_seconds"].to_numpy(dtype=float)
        rows.append(
            {
                "analysis_window": keys[0], "target": keys[1], "model": keys[2], "horizon": int(keys[3]),
                "origin_records": len(group),
                "model_selection_total_seconds": float(group["selection_seconds"].sum()),
                "final_fitting_total_seconds": float(group["fit_seconds"].sum()),
                "forecasting_total_seconds": float(group["forecast_seconds"].sum()),
                "total_seconds": float(np.sum(total)),
                "median_seconds_per_origin": float(np.median(total)),
                "mean_seconds_per_origin": float(np.mean(total)),
                "maximum_seconds_per_origin": float(np.max(total)),
            }
        )
    return pd.DataFrame(rows)


def build_failure_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return every failed model forecast without silently removing any origin."""

    columns = [
        "analysis_window", "target", "model", "horizon", "origin_year", "target_year",
        "fit_status", "fit_warning", "selected_specification",
    ]
    return forecasts.loc[~forecasts["fit_status"].eq("success"), columns].reset_index(drop=True)


def runtime_environment() -> pd.DataFrame:
    """Collect lightweight platform, CPU, and memory metadata without extra dependencies."""

    total_ram = np.nan
    if sys.platform.startswith("win"):
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total_ram = status.ullTotalPhys / (1024**3)
    return pd.DataFrame(
        [
            {"field": "operating_system", "value": platform.platform()},
            {"field": "python_version", "value": platform.python_version()},
            {"field": "processor_identifier", "value": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unavailable")},
            {"field": "physical_cpu_count", "value": "unavailable without an additional dependency"},
            {"field": "logical_cpu_count", "value": os.cpu_count()},
            {"field": "total_ram_gib", "value": total_ram},
            {"field": "timing_note", "value": "Observed wall-clock timings are hardware- and workload-specific."},
        ]
    )
