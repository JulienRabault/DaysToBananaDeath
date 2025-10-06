import pytest

from backend.app.api.services import predictor as predictor_module
from backend.app.api.services.predictor import PredictorService


def test_predict_image_bytes_returns_deterministic_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit test ensuring mock predictions are reproducible when randomness is patched."""
    predictor = PredictorService()

    monkeypatch.setattr(predictor_module.random, "choice", lambda seq: "ripe")
    monkeypatch.setattr(predictor_module.random, "uniform", lambda a, b: 0.88)

    result = predictor.predict_image_bytes(b"not-an-image-but-mock-does-not-care")

    assert result["predicted_class"] == "ripe"
    assert result["confidence"] == pytest.approx(0.88)
    assert result["predicted_index"] == predictor.class_names.index("ripe")

    # Ensure probabilities sum to 1 and contain the mocked confidence
    total_prob = sum(result["class_probabilities"].values())
    assert total_prob == pytest.approx(1.0)
    assert result["class_probabilities"]["ripe"] == pytest.approx(0.88)

    # Expected days should align with the configured mapping for the class
    class_to_days = predictor_module.get_class_to_days_mapping()
    assert result["expected_days"] == pytest.approx(class_to_days["ripe"])

    assert result["model_info"]["format"] == "mock"
    assert "warning" in result
