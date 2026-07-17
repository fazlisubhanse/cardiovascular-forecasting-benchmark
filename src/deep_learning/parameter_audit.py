"""Exact trainable/nontrainable parameter accounting."""

from __future__ import annotations

from torch import nn


def parameter_counts(model: nn.Module) -> dict[str, int]:
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    nontrainable=sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable_parameter_count":int(trainable),"nontrainable_parameter_count":int(nontrainable),
            "total_parameter_count":int(trainable+nontrainable)}
