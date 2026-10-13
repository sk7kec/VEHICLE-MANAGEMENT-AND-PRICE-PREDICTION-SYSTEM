"""
Vehicle API Tests

Tests for the FastAPI vehicle CRUD endpoints.

Author: Sahishnakumaar S S (24ADR134)
"""

import os
import sys
import json
import tempfile
import pytest
from fastapi.testclient import TestClient

# Ensure project is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fastapi_backend"))

# Set up temporary storage before importing the app
_temp_dir = tempfile.mkdtemp()
os.environ["DATA_DIR"] = _temp_dir
os.environ["ARTIFACTS_DIR"] = os.path.join(os.path.dirname(__file__), "..", "artifacts")

from app.main import app, repository

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_storage():
    """Reset vehicle storage before each test."""
    # Write directly to the repository's actual file path
    storage_path = repository._filepath
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    with open(storage_path, "w") as f:
        json.dump([], f)
    yield


def make_vehicle_payload(**overrides):
    """Create a default vehicle payload with optional overrides."""
    payload = {
        "make": "Maruti Suzuki",
        "model": "Swift",
        "year": 2020,
        "km_driven": 35000,
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "engine_cc": 1197,
        "previous_owners": 1,
    }
    payload.update(overrides)
    return payload


# ── Health Check ──

class TestHealthCheck:
    def test_api_status(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "endpoints" in data
        assert data["project"] == "Vehicle Management and Price Prediction System"

    def test_docs_available(self):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "info" in schema


# ── Create Vehicle ──

class TestCreateVehicle:
    def test_create_valid_vehicle(self):
        payload = make_vehicle_payload()
        response = client.post("/vehicles", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["make"] == "Maruti Suzuki"
        assert data["model"] == "Swift"
        assert data["year"] == 2020
        assert data["km_driven"] == 35000
        assert data["fuel_type"] == "Petrol"
        assert data["transmission"] == "Manual"
        assert data["engine_cc"] == 1197
        assert "created_at" in data
        assert "updated_at" in data

    def test_create_with_missing_required_field(self):
        payload = {"make": "Honda"}  # Missing most required fields
        response = client.post("/vehicles", json=payload)
        assert response.status_code == 422

    def test_create_with_invalid_year(self):
        payload = make_vehicle_payload(year=1800)
        response = client.post("/vehicles", json=payload)
        assert response.status_code == 422

    def test_create_with_negative_km(self):
        payload = make_vehicle_payload(km_driven=-100)
        response = client.post("/vehicles", json=payload)
        assert response.status_code == 422

    def test_create_with_invalid_fuel_type(self):
        payload = make_vehicle_payload(fuel_type="Nuclear")
        response = client.post("/vehicles", json=payload)
        assert response.status_code == 422

    def test_create_with_invalid_transmission(self):
        payload = make_vehicle_payload(transmission="CVT")
        response = client.post("/vehicles", json=payload)
        assert response.status_code == 422

    def test_create_with_invalid_engine_cc(self):
        payload = make_vehicle_payload(engine_cc=0)
        response = client.post("/vehicles", json=payload)
        assert response.status_code == 422

    def test_create_with_empty_make(self):
        payload = make_vehicle_payload(make="")
        response = client.post("/vehicles", json=payload)
        assert response.status_code == 422

    def test_create_with_empty_model(self):
        payload = make_vehicle_payload(model="   ")
        response = client.post("/vehicles", json=payload)
        assert response.status_code == 422

    def test_create_with_default_owners(self):
        payload = make_vehicle_payload()
        del payload["previous_owners"]
        response = client.post("/vehicles", json=payload)
        assert response.status_code == 201
        assert response.json()["previous_owners"] == 0


# ── List Vehicles ──

class TestListVehicles:
    def test_list_empty(self):
        response = client.get("/vehicles")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["vehicles"] == []

    def test_list_with_vehicles(self):
        # Create two vehicles
        client.post("/vehicles", json=make_vehicle_payload())
        client.post("/vehicles", json=make_vehicle_payload(model="Baleno"))
        response = client.get("/vehicles")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["vehicles"]) == 2


# ── Get Vehicle by ID ──

class TestGetVehicle:
    def test_get_existing(self):
        create_resp = client.post("/vehicles", json=make_vehicle_payload())
        vehicle_id = create_resp.json()["id"]
        response = client.get(f"/vehicles/{vehicle_id}")
        assert response.status_code == 200
        assert response.json()["id"] == vehicle_id

    def test_get_missing(self):
        response = client.get("/vehicles/NONEXISTENT")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# ── Update Vehicle (PUT) ──

class TestUpdateVehicle:
    def test_update_existing(self):
        create_resp = client.post("/vehicles", json=make_vehicle_payload())
        vehicle_id = create_resp.json()["id"]

        update_payload = make_vehicle_payload(model="Dzire", year=2022, km_driven=15000)
        response = client.put(f"/vehicles/{vehicle_id}", json=update_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "Dzire"
        assert data["year"] == 2022
        assert data["km_driven"] == 15000
        assert data["id"] == vehicle_id

    def test_update_missing(self):
        response = client.put("/vehicles/NONEXISTENT", json=make_vehicle_payload())
        assert response.status_code == 404

    def test_update_with_invalid_data(self):
        create_resp = client.post("/vehicles", json=make_vehicle_payload())
        vehicle_id = create_resp.json()["id"]
        response = client.put(f"/vehicles/{vehicle_id}", json={"make": "Test"})
        assert response.status_code == 422


# ── Patch Vehicle (PATCH) ──

class TestPatchVehicle:
    def test_patch_single_field(self):
        create_resp = client.post("/vehicles", json=make_vehicle_payload())
        vehicle_id = create_resp.json()["id"]

        response = client.patch(f"/vehicles/{vehicle_id}", json={"km_driven": 50000})
        assert response.status_code == 200
        data = response.json()
        assert data["km_driven"] == 50000
        assert data["make"] == "Maruti Suzuki"  # unchanged

    def test_patch_multiple_fields(self):
        create_resp = client.post("/vehicles", json=make_vehicle_payload())
        vehicle_id = create_resp.json()["id"]

        response = client.patch(f"/vehicles/{vehicle_id}", json={
            "fuel_type": "Diesel",
            "transmission": "Automatic"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["fuel_type"] == "Diesel"
        assert data["transmission"] == "Automatic"

    def test_patch_missing(self):
        response = client.patch("/vehicles/NONEXISTENT", json={"km_driven": 50000})
        assert response.status_code == 404

    def test_patch_invalid_fuel(self):
        create_resp = client.post("/vehicles", json=make_vehicle_payload())
        vehicle_id = create_resp.json()["id"]
        response = client.patch(f"/vehicles/{vehicle_id}", json={"fuel_type": "Steam"})
        assert response.status_code == 422


# ── Delete Vehicle ──

class TestDeleteVehicle:
    def test_delete_existing(self):
        create_resp = client.post("/vehicles", json=make_vehicle_payload())
        vehicle_id = create_resp.json()["id"]

        response = client.delete(f"/vehicles/{vehicle_id}")
        assert response.status_code == 200
        assert response.json()["id"] == vehicle_id

        # Verify it's gone
        get_resp = client.get(f"/vehicles/{vehicle_id}")
        assert get_resp.status_code == 404

    def test_delete_missing(self):
        response = client.delete("/vehicles/NONEXISTENT")
        assert response.status_code == 404


# ── Storage Edge Cases ──

class TestStorageEdgeCases:
    def test_malformed_storage_file(self):
        """Test recovery from corrupted storage file."""
        storage_path = repository._filepath
        with open(storage_path, "w") as f:
            f.write("{corrupted data!!!")

        # The API should still work (recovery mechanism)
        response = client.get("/vehicles")
        assert response.status_code == 200

    def test_empty_storage_file(self):
        """Test handling of empty storage file."""
        storage_path = repository._filepath
        with open(storage_path, "w") as f:
            f.write("")

        response = client.get("/vehicles")
        assert response.status_code == 200
        assert response.json()["total"] == 0

    def test_persistence(self):
        """Test that created vehicles persist."""
        payload = make_vehicle_payload()
        create_resp = client.post("/vehicles", json=payload)
        vehicle_id = create_resp.json()["id"]

        # Read directly from file
        storage_path = repository._filepath
        with open(storage_path, "r") as f:
            records = json.load(f)

        found = any(r["id"] == vehicle_id for r in records)
        assert found, "Vehicle should be persisted in JSON file"

