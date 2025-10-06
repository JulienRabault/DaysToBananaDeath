import asyncio
from io import BytesIO

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from backend.app.api.endpoints import predict as predict_endpoint


class StubPredictor:
    def __init__(self):
        self.seen_payload = None

    def predict_image_bytes(self, payload: bytes):
        self.seen_payload = payload
        return {
            "predicted_class": "ripe",
            "confidence": 0.9,
            "predicted_index": 1,
            "class_probabilities": {"ripe": 0.9, "unripe": 0.025, "overripe": 0.025, "rotten": 0.025, "unknowns": 0.025},
            "expected_days": 4.0,
            "model_info": {"format": "mock", "img_size": 224, "model_type": "test"},
        }


def test_predict_file_endpoint_returns_augmented_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = StubPredictor()

    monkeypatch.setattr(predict_endpoint, "get_predictor", lambda: stub)

    upload = UploadFile(
        file=BytesIO(b"binary-image"),
        filename="banana.jpg",
        headers=Headers({"content-type": "image/jpeg"}),
    )

    response = asyncio.run(predict_endpoint.predict_from_file(upload))

    assert response["predicted_class"] == "ripe"
    assert response["temp_file_data"]["filename"] == "banana.jpg"
    assert response["temp_file_data"]["content_type"] == "image/jpeg"
    assert response["temp_file_data"]["content"]
    assert response["image_key"].startswith("temp_")
    assert stub.seen_payload == b"binary-image"
