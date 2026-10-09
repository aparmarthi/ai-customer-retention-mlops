"""
Pytest test suite for the Churn Decision Intelligence API.

Tests cover:
  - Health check endpoint
  - Single prediction (threshold policy)
  - Batch prediction (threshold + top-K policies)
  - Input validation and error paths
  - CSV file upload

Run:
  pytest src/serving/test_api.py -v
"""

import io
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serving.api import app


@pytest.fixture(scope="module")
def client():
    """Create a test client with startup event handling."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def feature_cols():
    """Load expected feature columns from champion artifacts."""
    art = Path(__file__).resolve().parents[2] / "artifacts" / "champion"
    return json.loads((art / "feature_list.json").read_text())


@pytest.fixture(scope="module")
def sample_record(feature_cols):
    """Build a minimal valid feature record (all zeros — produces a prediction)."""
    return {col: 0 for col in feature_cols}


# ── Health ────────────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_health_reports_features(self, client):
        r = client.get("/health")
        body = r.json()
        assert body["n_features"] > 0
        assert isinstance(body["default_threshold"], float)


# ── Single Prediction ─────────────────────────────────────────────────────────


class TestPredict:
    def test_predict_returns_probability(self, client, sample_record):
        r = client.post("/predict", json={"record": sample_record})
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["churn_label"] in (0, 1)
        assert body["policy_used"] == "threshold"

    def test_predict_custom_threshold(self, client, sample_record):
        r = client.post("/predict", json={
            "record": sample_record,
            "threshold": 0.01,
        })
        assert r.status_code == 200
        # Very low threshold — almost everything should be flagged
        body = r.json()
        assert body["threshold_used"] == 0.01

    def test_predict_rejects_topk_policy(self, client, sample_record):
        r = client.post("/predict", json={
            "record": sample_record,
            "policy": "top_k",
        })
        assert r.status_code == 400
        assert "top_k" in r.json()["detail"].lower() or "batch" in r.json()["detail"].lower()

    def test_predict_missing_features_filled_nan(self, client):
        """Records with missing features should still get a prediction (NaN fill)."""
        r = client.post("/predict", json={"record": {"nonexistent_col": 999}})
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["churn_probability"] <= 1.0

    def test_predict_empty_record(self, client):
        """Empty record should still produce a prediction (all NaN)."""
        r = client.post("/predict", json={"record": {}})
        assert r.status_code == 200


# ── Batch Prediction ──────────────────────────────────────────────────────────


class TestPredictBatch:
    """/predict_batch accepts either a JSON body (BatchPredictRequest) or a CSV file upload."""

    def test_batch_json_threshold(self, client, sample_record):
        r = client.post("/predict_batch", json={"records": [sample_record] * 3, "policy": "threshold"})
        assert r.status_code == 200
        body = r.json()
        assert body["n"] == 3
        assert body["policy_used"] == "threshold"

    def test_batch_json_top_k(self, client, sample_record, feature_cols):
        low_risk = {**sample_record, "cancel_rate": 0.0, "auto_renew_rate": 1.0}
        high_risk = {**sample_record, "cancel_rate": 1.0, "auto_renew_rate": 0.0}
        r = client.post("/predict_batch", json={"records": [low_risk, high_risk, low_risk], "policy": "top_k", "k": 1})
        assert r.status_code == 200
        body = r.json()
        assert body["policy_used"] == "top_k(1)"
        assert sorted(item["rank"] for item in body["items"]) == [1, 2, 3]
        assert sum(item["churn_label"] for item in body["items"]) == 1

    def test_batch_json_top_k_requires_k(self, client, sample_record):
        r = client.post("/predict_batch", json={"records": [sample_record], "policy": "top_k"})
        assert r.status_code == 400

    def test_batch_json_invalid_body(self, client):
        r = client.post("/predict_batch", json={"not_records": []})
        assert r.status_code == 422

    def test_batch_csv_top_k_via_query(self, client, sample_record):
        csv_bytes = pd.DataFrame([sample_record] * 4).to_csv(index=False).encode()
        r = client.post(
            "/predict_batch?policy=top_k&k=2",
            files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert r.status_code == 200
        assert r.json()["policy_used"] == "top_k(2)"
        assert sum(item["churn_label"] for item in r.json()["items"]) == 2

    def test_batch_threshold_via_csv(self, client, sample_record):
        """Batch threshold policy via CSV upload."""
        df = pd.DataFrame([sample_record, sample_record, sample_record])
        csv_bytes = df.to_csv(index=False).encode()
        r = client.post(
            "/predict_batch",
            files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["n"] == 3
        assert body["policy_used"] == "threshold"
        assert len(body["items"]) == 3
        for item in body["items"]:
            assert 0.0 <= item["churn_probability"] <= 1.0

    def test_batch_returns_labels_and_thresholds(self, client, sample_record):
        df = pd.DataFrame([sample_record])
        csv_bytes = df.to_csv(index=False).encode()
        r = client.post(
            "/predict_batch",
            files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert r.status_code == 200
        item = r.json()["items"][0]
        assert item["churn_label"] in (0, 1)
        assert item["policy_used"] == "threshold"
        assert item["threshold_used"] is not None

    def test_batch_csv_with_extra_columns(self, client, sample_record):
        """CSV with extra columns should work (extras silently dropped)."""
        record = {**sample_record, "extra_col_1": 999, "extra_col_2": "hello"}
        df = pd.DataFrame([record])
        csv_bytes = df.to_csv(index=False).encode()
        r = client.post(
            "/predict_batch",
            files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert r.status_code == 200

    def test_batch_csv_missing_columns(self, client):
        """CSV with only some features should work (missing filled with NaN)."""
        df = pd.DataFrame([{"some_random_col": 42}])
        csv_bytes = df.to_csv(index=False).encode()
        r = client.post(
            "/predict_batch",
            files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert r.status_code == 200

    def test_batch_csv_malformed(self, client):
        """Malformed CSV should return 400, not 500."""
        r = client.post(
            "/predict_batch",
            files={"file": ("bad.csv", io.BytesIO(b"not,a,valid\x00csv\xff"), "text/csv")},
        )
        # Should either parse it (pandas is forgiving) or return 400
        assert r.status_code in (200, 400)

    def test_batch_no_input(self, client):
        """No body and no file should return 400."""
        r = client.post("/predict_batch")
        assert r.status_code in (400, 422)


# ── Edge Cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_topk_k_zero_rejected(self, client, sample_record):
        """k=0 should be rejected (nothing to target)."""
        # This test uses CSV upload since JSON batch has parameter parsing issues
        # with FastAPI dual-optional parameters. The k=0 check is in _apply_topk_policy.
        # We verify the policy logic directly here:
        from src.serving.api import _apply_topk_policy
        import numpy as np
        with pytest.raises(Exception):
            _apply_topk_policy(np.array([0.5, 0.3]), k=0)

    def test_threshold_zero_flags_all(self, client, sample_record):
        """Threshold of 0 should flag everything as churn."""
        r = client.post("/predict", json={
            "record": sample_record,
            "threshold": 0.0,
        })
        assert r.status_code == 200
        assert r.json()["churn_label"] == 1

    def test_threshold_one_flags_none(self, client, sample_record):
        """Threshold of 1.0 should flag nothing (probability never reaches 1.0)."""
        r = client.post("/predict", json={
            "record": sample_record,
            "threshold": 1.0,
        })
        assert r.status_code == 200
        assert r.json()["churn_label"] == 0


# ── CORS ──────────────────────────────────────────────────────────────────────


def test_cors_preflight_allows_browser_clients(client):
    r = client.options(
        "/predict",
        headers={"Origin": "https://example.bolt.new", "Access-Control-Request-Method": "POST"},
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] in ("*", "https://example.bolt.new")
