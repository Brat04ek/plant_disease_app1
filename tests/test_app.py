"""API contract checks with substituted artifacts, not model accuracy tests."""
import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    sys.modules.pop("app", None)
    with patch.object(Path, "is_file", return_value=True), patch("joblib.load", return_value=object()):
        module = importlib.import_module("app")
    with TestClient(module.app) as test_client:
        yield test_client
    sys.modules.pop("app", None)


def test_home_and_health(client):
    assert client.get("/").status_code == 200
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/static/app.js").status_code == 200


@pytest.mark.parametrize("payload", [b"", b"not an image"])
def test_invalid_upload(client, payload):
    response = client.post("/predict", files={"file": ("leaf.png", payload, "image/png")})
    assert response.status_code == 400
    assert response.json()["status"] == "error"


def test_upload_limit(client):
    response = client.post("/predict", files={"file": ("leaf.png", b"x" * (10 * 1024 * 1024 + 1))})
    assert response.status_code == 413


def test_blurry_image_skips_inference(client):
    ok, image = cv2.imencode(".png", np.zeros((256, 256, 3), dtype=np.uint8))
    assert ok
    response = client.post("/predict", files={"file": ("leaf.png", image.tobytes(), "image/png")})
    assert response.status_code == 200
    assert response.json()["status"] == "bad_quality"


def test_missing_artifacts_explained():
    sys.modules.pop("app", None)
    with patch.object(Path, "is_file", return_value=False):
        with pytest.raises(RuntimeError, match="Missing trained artifacts"):
            importlib.import_module("app")

@pytest.mark.parametrize("confidence, status", [(0.8, "success"), (0.5, "uncertain")])
def test_prediction_contract(client, monkeypatch, confidence, status):
    module = sys.modules["app"]

    class Classifier:
        def predict_proba(self, features):
            assert features.shape == (1, 122)
            return np.array([[confidence, (1 - confidence) / 2, (1 - confidence) / 2]])

    class Encoder:
        def inverse_transform(self, indices):
            return np.array([f"class_{i}" for i in indices])

    monkeypatch.setattr(module, "classifier", Classifier())
    monkeypatch.setattr(module, "label_encoder", Encoder())
    monkeypatch.setattr(module, "check_image_quality", lambda gray: True)
    monkeypatch.setattr(module, "extract_full_feature_vector", lambda image, model: np.zeros(122))
    ok, image = cv2.imencode(".png", np.zeros((256, 256, 3), dtype=np.uint8))
    assert ok
    response = client.post("/predict", files={"file": ("leaf.png", image.tobytes(), "image/png")})
    data = response.json()
    assert response.status_code == 200
    assert data["status"] == status
    assert data["predicted_class"] == "class_0"
    assert data["confidence"] == confidence
    assert len(data["top_predictions"]) == 3
