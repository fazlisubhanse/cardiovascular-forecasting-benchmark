"""Leakage-controlled Phase 2B feature engineering."""

from .supervised import DirectDataset, build_direct_dataset, build_sample_manifest
from .transformations import fit_linear_trend

__all__ = ["DirectDataset", "build_direct_dataset", "build_sample_manifest", "fit_linear_trend"]
