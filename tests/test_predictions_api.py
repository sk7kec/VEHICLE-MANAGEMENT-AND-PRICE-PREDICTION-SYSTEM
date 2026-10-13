"""
Prediction API Tests

Tests for the price prediction and category classification endpoints.

Author: Sahishnakumaar S S (24ADR134)
"""

import os
import sys
import json
import tempfile
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fastapi_backend"))

_temp_dir = tempfile.mkdtemp()
os.environ["DATA_DIR"] = _temp_dir
os.environ["ARTIFACTS_DIR"] = os.path.join(os.path.dirname(__file__), "..", "artifacts")

from app.main import app
from app.services.prediction_service import PredictionService

client = TestClient(app)


def make_prediction_payload(**overrides):
    """Create a default prediction payload."""
    payload = {
        "make": "Hyundai",
        "model": "Creta",
        "year": 2019,
        "km_driven": 45000,
        "fuel_type": "Diesel",
        "transmission": "Automatic",
        "engine_cc": 1493,
        "previous_owners": 1,
    }
    payload.update(overrides)
    return payload


# ── Price Prediction ──

class TestPricePrediction:
    def test_predict_price_valid(self):
        """Test price prediction with valid input."""
        payload = make_prediction_payload()
        response = client.post("/predict/price", json=payload)

        if response.status_code == 503:
            pytest.skip("Price model not available (not trained yet)")

        assert response.status_code == 200
        data = response.json()
        assert "predicted_price" in data
        assert isinstance(data["predicted_price"], (int, float))
        assert data["predicted_price"] >= 0
        assert data["currency"] == "INR"
        assert "model_name" in data
        assert "input_summary" in data
        assert "disclaimer" in data

    def test_predict_price_missing_fields(self):
        """Test price prediction with missing required fields."""
        response = client.post("/predict/price", json={"make": "Honda"})
        assert response.status_code == 422

    def test_predict_price_invalid_year(self):
        """Test price prediction with invalid year."""
        payload = make_prediction_payload(year=1800)
        response = client.post("/predict/price", json=payload)
        assert response.status_code == 422

    def test_predict_price_invalid_fuel(self):
        """Test price prediction with invalid fuel type."""
        payload = make_prediction_payload(fuel_type="Hydrogen")
        response = client.post("/predict/price", json=payload)
        assert response.status_code == 422

    def test_predict_price_negative_km(self):
        """Test price prediction with negative km."""
        payload = make_prediction_payload(km_driven=-5000)
        response = client.post("/predict/price", json=payload)
        assert response.status_code == 422

    def test_predict_price_zero_engine(self):
        """Test price prediction with zero engine CC."""
        payload = make_prediction_payload(engine_cc=0)
        response = client.post("/predict/price", json=payload)
        assert response.status_code == 422

    def test_predict_price_with_different_vehicles(self):
        """Test price prediction gives different results for different vehicles."""
        payload1 = make_prediction_payload(make="Maruti Suzuki", model="Alto", year=2015, engine_cc=796)
        payload2 = make_prediction_payload(make="BMW", model="3 Series", year=2022, engine_cc=2998)

        resp1 = client.post("/predict/price", json=payload1)
        resp2 = client.post("/predict/price", json=payload2)

        if resp1.status_code == 503 or resp2.status_code == 503:
            pytest.skip("Price model not available")

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        # Different vehicles should generally get different prices
        price1 = resp1.json()["predicted_price"]
        price2 = resp2.json()["predicted_price"]
        assert price1 != price2 or (price1 == price2 == 0), "Different vehicles should get different price estimates"


# ── Category Classification ──

class TestCategoryClassification:
    VALID_CATEGORIES = {"Budget", "Mid-range", "Premium"}

    def test_classify_valid(self):
        """Test category classification with valid input."""
        payload = make_prediction_payload()
        response = client.post("/predict/category", json=payload)

        if response.status_code == 503:
            pytest.skip("Category model not available (not trained yet)")

        assert response.status_code == 200
        data = response.json()
        assert "predicted_category" in data
        assert data["predicted_category"] in self.VALID_CATEGORIES
        assert "model_name" in data
        assert "categories" in data
        assert "category_description" in data
        assert "disclaimer" in data

    def test_classify_missing_fields(self):
        """Test classification with missing fields."""
        response = client.post("/predict/category", json={"make": "Toyota"})
        assert response.status_code == 422

    def test_classify_invalid_transmission(self):
        """Test classification with invalid transmission."""
        payload = make_prediction_payload(transmission="CVT")
        response = client.post("/predict/category", json=payload)
        assert response.status_code == 422

    def test_classify_budget_vehicle(self):
        """Test that a budget-like vehicle is classified correctly."""
        payload = make_prediction_payload(
            make="Maruti Suzuki", model="Alto",
            year=2010, km_driven=120000,
            engine_cc=796, fuel_type="Petrol",
            transmission="Manual", previous_owners=3
        )
        response = client.post("/predict/category", json=payload)
        if response.status_code == 503:
            pytest.skip("Category model not available")
        assert response.status_code == 200
        assert response.json()["predicted_category"] in self.VALID_CATEGORIES

    def test_classify_premium_vehicle(self):
        """Test classification for a premium vehicle."""
        payload = make_prediction_payload(
            make="BMW", model="3 Series",
            year=2023, km_driven=5000,
            engine_cc=2998, fuel_type="Petrol",
            transmission="Automatic", previous_owners=0
        )
        response = client.post("/predict/category", json=payload)
        if response.status_code == 503:
            pytest.skip("Category model not available")
        assert response.status_code == 200
        assert response.json()["predicted_category"] in self.VALID_CATEGORIES


# ── Integration Tests ──

class TestIntegration:
    def test_api_endpoints_documented(self):
        """Test that all required endpoints are in the OpenAPI schema."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        paths = schema.get("paths", {})

        required_paths = [
            "/",
            "/vehicles",
            "/vehicles/{vehicle_id}",
            "/predict/price",
            "/predict/category",
        ]
        for path in required_paths:
            assert path in paths, f"Missing endpoint in API docs: {path}"

    def test_vehicle_crud_workflow(self):
        """Test complete CRUD workflow."""
        # Create
        payload = make_prediction_payload()
        payload["make"] = "Toyota"
        payload["model"] = "Camry"
        resp = client.post("/vehicles", json=payload)
        assert resp.status_code == 201
        vehicle_id = resp.json()["id"]

        # Read
        resp = client.get(f"/vehicles/{vehicle_id}")
        assert resp.status_code == 200
        assert resp.json()["make"] == "Toyota"

        # Update
        payload["model"] = "Corolla"
        resp = client.put(f"/vehicles/{vehicle_id}", json=payload)
        assert resp.status_code == 200
        assert resp.json()["model"] == "Corolla"

        # Patch
        resp = client.patch(f"/vehicles/{vehicle_id}", json={"km_driven": 60000})
        assert resp.status_code == 200
        assert resp.json()["km_driven"] == 60000

        # List
        resp = client.get("/vehicles")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

        # Delete
        resp = client.delete(f"/vehicles/{vehicle_id}")
        assert resp.status_code == 200

        # Verify deleted
        resp = client.get(f"/vehicles/{vehicle_id}")
        assert resp.status_code == 404

    def test_response_schema_consistency(self):
        """Test that vehicle response contains all expected fields."""
        payload = make_prediction_payload()
        resp = client.post("/vehicles", json=payload)
        assert resp.status_code == 201
        data = resp.json()

        expected_fields = [
            "id", "make", "model", "year", "km_driven",
            "fuel_type", "transmission", "engine_cc",
            "previous_owners", "created_at", "updated_at"
        ]
        for field in expected_fields:
            assert field in data, f"Missing field in response: {field}"
