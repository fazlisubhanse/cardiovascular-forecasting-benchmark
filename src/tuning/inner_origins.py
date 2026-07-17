"""Leakage-free inner pseudo-origin construction."""

from __future__ import annotations


def valid_inner_origins(first_year: int, outer_origin: int, horizon: int, count: int,
                        minimum_training_years: int) -> list[int]:
    """Return the most recent pseudo origins whose targets lie inside outer history."""

    earliest = first_year + minimum_training_years - 1
    latest = outer_origin - horizon
    origins = list(range(earliest, latest + 1))
    return origins[-count:]
