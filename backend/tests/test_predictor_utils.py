import math

import pytest

from backend.app.api.services.predictor import (
    PredictorService,
    get_days_to_class_thresholds,
    map_days_left_to_class,
)


@pytest.mark.parametrize(
    "days_left,expected",
    [
        (6.2, "unripe"),
        (4.2, "ripe"),
        (1.3, "overripe"),
        (0.2, "rotten"),
        (-1.0, "unknowns"),
    ],
)
def test_map_days_left_to_class_matches_thresholds(days_left: float, expected: str) -> None:
    """Ensure the helper respects the configured thresholds."""
    assert map_days_left_to_class(days_left) == expected


def test_map_days_left_to_class_rounds_before_comparison() -> None:
    """Rounding should shift borderline values to the correct class."""
    thresholds = get_days_to_class_thresholds()

    # Slightly below the unripe threshold should round up to unripe
    almost_unripe = thresholds["unripe_min"] - 0.49
    assert map_days_left_to_class(almost_unripe) == "unripe"

    # Slightly above the ripe max should round down to ripe
    almost_ripe = thresholds["ripe_max"] + 0.49
    assert map_days_left_to_class(almost_ripe) == "ripe"


def test_validate_correction_reports_consistency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for correction validation consistency reporting."""
    predictor = PredictorService()

    # Force deterministic mapping by patching helper to a known class
    monkeypatch.setattr(
        "backend.app.api.services.predictor.map_days_left_to_class",
        lambda value: "ripe",
    )

    validation = predictor.validate_correction(days_left=3.0, user_class="ripe")

    assert validation["suggested_class"] == "ripe"
    assert validation["is_consistent"] is True
    assert math.isclose(validation["confidence_score"], 1.0)

    validation_mismatch = predictor.validate_correction(days_left=3.0, user_class="unripe")
    assert validation_mismatch["suggested_class"] == "ripe"
    assert validation_mismatch["is_consistent"] is False
    assert math.isclose(validation_mismatch["confidence_score"], 0.5)
