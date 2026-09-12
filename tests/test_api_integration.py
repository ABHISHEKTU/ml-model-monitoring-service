"""
Integration tests through the real FastAPI app — covers wiring in main.py
that pure unit tests on drift/ can't reach (prediction drift, multivariate
drift attached to the /monitor/drift response, full request/response cycle).
"""
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
np.random.seed(7)


def _reference_payload(model_id: str, with_predictions: bool = False) -> dict:
    payload = {
        "model_id": model_id,
        "numeric_features": ["age"],
        "categorical_features": ["city"],
        "records": [
            {"age": float(a), "city": "A" if i % 2 == 0 else "B"}
            for i, a in enumerate(np.random.uniform(20, 40, 40))
        ],
    }
    if with_predictions:
        payload["reference_predictions"] = list(np.random.uniform(0.1, 0.3, 40))
    return payload


def test_prediction_drift_appears_and_flags_shift():
    client.post("/reference", json=_reference_payload("pred_test", with_predictions=True))

    resp = client.post("/monitor/drift", json={
        "model_id": "pred_test",
        "records": [{"age": float(a), "city": "A"} for a in np.random.uniform(20, 40, 40)],
        "current_predictions": list(np.random.uniform(0.7, 0.9, 40)),  # shifted high
    })
    body = resp.json()
    assert resp.status_code == 200
    assert "prediction_drift" in body
    assert body["prediction_drift"]["psi"]["drift_detected"] is True


def test_no_prediction_drift_key_when_predictions_not_sent():
    client.post("/reference", json=_reference_payload("no_pred_test", with_predictions=False))

    resp = client.post("/monitor/drift", json={
        "model_id": "no_pred_test",
        "records": [{"age": float(a), "city": "A"} for a in np.random.uniform(20, 40, 40)],
    })
    body = resp.json()
    assert resp.status_code == 200
    assert "prediction_drift" not in body


def test_multivariate_drift_appears_in_full_response():
    client.post("/reference", json=_reference_payload("multi_test"))

    resp = client.post("/monitor/drift", json={
        "model_id": "multi_test",
        "records": [{"age": float(a), "city": "A" if i % 2 == 0 else "B"}
                    for i, a in enumerate(np.random.uniform(20, 40, 40))],
    })
    body = resp.json()
    assert resp.status_code == 200
    assert "multivariate_drift" in body
    assert "auc" in body["multivariate_drift"]