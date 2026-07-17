"""Manual Winkler interval score tests."""

from src.evaluation.intervals import winkler_interval_score


def test_interval_score_coverage_and_miss_penalty() -> None:
    assert winkler_interval_score(5.0, 4.0, 6.0, 0.2) == 2.0
    assert winkler_interval_score(3.0, 4.0, 6.0, 0.2) == 12.0
    assert winkler_interval_score(7.0, 4.0, 6.0, 0.2) == 12.0
