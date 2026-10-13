"""
API Routes — All FastAPI endpoint definitions

Author: Sahishnakumaar S S (24ADR134)
"""

import logging
from fastapi import APIRouter, HTTPException, status

from ..schemas import (
    VehicleCreate, VehicleUpdate, VehiclePatch,
    VehicleResponse, VehicleListResponse, DeleteResponse,
    PricePredictionRequest, PricePredictionResponse,
    CategoryPredictionRequest, CategoryPredictionResponse,
    StatusResponse, ErrorResponse,
)
from ..services.vehicle_service import VehicleService
from ..services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter()

# Service instances (set during app startup)
vehicle_service: VehicleService = None
prediction_service: PredictionService = None


def init_services(vs: VehicleService, ps: PredictionService):
    """Initialize route-level service references."""
    global vehicle_service, prediction_service
    vehicle_service = vs
    prediction_service = ps


# ── Status ───────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=StatusResponse,
    tags=["Status"],
    summary="API Health Check",
    description="Returns the current status and available endpoints of the API."
)
def get_status():
    """API health/status endpoint."""
    return StatusResponse(
        status="running",
        endpoints={
            "vehicles": "/vehicles",
            "vehicle_by_id": "/vehicles/{id}",
            "predict_price": "/predict/price",
            "predict_category": "/predict/category",
            "docs": "/docs",
        }
    )


# ── Vehicle CRUD ─────────────────────────────────────────────────

@router.get(
    "/vehicles",
    response_model=VehicleListResponse,
    tags=["Vehicles"],
    summary="List All Vehicles",
    description="Retrieve all vehicle records from storage."
)
def list_vehicles():
    """Get all vehicle records."""
    vehicles = vehicle_service.get_all_vehicles()
    return VehicleListResponse(total=len(vehicles), vehicles=vehicles)


@router.get(
    "/vehicles/{vehicle_id}",
    response_model=VehicleResponse,
    tags=["Vehicles"],
    summary="Get Vehicle by ID",
    description="Retrieve a single vehicle record by its unique ID.",
    responses={404: {"model": ErrorResponse, "description": "Vehicle not found"}}
)
def get_vehicle(vehicle_id: str):
    """Get a vehicle record by ID."""
    vehicle = vehicle_service.get_vehicle(vehicle_id)
    if vehicle is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle with ID '{vehicle_id}' not found"
        )
    return vehicle


@router.post(
    "/vehicles",
    response_model=VehicleResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Vehicles"],
    summary="Create Vehicle",
    description="Create a new vehicle record. A unique ID is assigned automatically."
)
def create_vehicle(vehicle: VehicleCreate):
    """Create a new vehicle record."""
    vehicle_data = vehicle.model_dump()
    # Convert enum values to strings for storage
    vehicle_data["fuel_type"] = vehicle_data["fuel_type"].value if hasattr(vehicle_data["fuel_type"], "value") else vehicle_data["fuel_type"]
    vehicle_data["transmission"] = vehicle_data["transmission"].value if hasattr(vehicle_data["transmission"], "value") else vehicle_data["transmission"]
    record = vehicle_service.create_vehicle(vehicle_data)
    return record


@router.put(
    "/vehicles/{vehicle_id}",
    response_model=VehicleResponse,
    tags=["Vehicles"],
    summary="Replace Vehicle",
    description="Replace all fields of an existing vehicle record.",
    responses={404: {"model": ErrorResponse, "description": "Vehicle not found"}}
)
def update_vehicle(vehicle_id: str, vehicle: VehicleUpdate):
    """Replace a vehicle record (PUT)."""
    vehicle_data = vehicle.model_dump()
    vehicle_data["fuel_type"] = vehicle_data["fuel_type"].value if hasattr(vehicle_data["fuel_type"], "value") else vehicle_data["fuel_type"]
    vehicle_data["transmission"] = vehicle_data["transmission"].value if hasattr(vehicle_data["transmission"], "value") else vehicle_data["transmission"]
    record = vehicle_service.update_vehicle(vehicle_id, vehicle_data)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle with ID '{vehicle_id}' not found"
        )
    return record


@router.patch(
    "/vehicles/{vehicle_id}",
    response_model=VehicleResponse,
    tags=["Vehicles"],
    summary="Partially Update Vehicle",
    description="Update selected fields of an existing vehicle record.",
    responses={404: {"model": ErrorResponse, "description": "Vehicle not found"}}
)
def patch_vehicle(vehicle_id: str, vehicle: VehiclePatch):
    """Partially update a vehicle record (PATCH)."""
    patch_data = vehicle.model_dump(exclude_unset=True)
    # Convert enum values
    if "fuel_type" in patch_data and patch_data["fuel_type"] is not None:
        patch_data["fuel_type"] = patch_data["fuel_type"].value if hasattr(patch_data["fuel_type"], "value") else patch_data["fuel_type"]
    if "transmission" in patch_data and patch_data["transmission"] is not None:
        patch_data["transmission"] = patch_data["transmission"].value if hasattr(patch_data["transmission"], "value") else patch_data["transmission"]

    # Check vehicle exists first
    existing = vehicle_service.get_vehicle(vehicle_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle with ID '{vehicle_id}' not found"
        )

    record = vehicle_service.patch_vehicle(vehicle_id, patch_data)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle with ID '{vehicle_id}' not found"
        )
    return record


@router.delete(
    "/vehicles/{vehicle_id}",
    response_model=DeleteResponse,
    tags=["Vehicles"],
    summary="Delete Vehicle",
    description="Delete a vehicle record by ID.",
    responses={404: {"model": ErrorResponse, "description": "Vehicle not found"}}
)
def delete_vehicle(vehicle_id: str):
    """Delete a vehicle record."""
    success = vehicle_service.delete_vehicle(vehicle_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle with ID '{vehicle_id}' not found"
        )
    return DeleteResponse(message="Vehicle deleted successfully", id=vehicle_id)


# ── Predictions ──────────────────────────────────────────────────

@router.post(
    "/predict/price",
    response_model=PricePredictionResponse,
    tags=["Predictions"],
    summary="Predict Vehicle Price",
    description="Predict the estimated resale price of a vehicle using the trained RandomForestRegressor model.",
    responses={
        503: {"model": ErrorResponse, "description": "Model not available"},
        422: {"model": ErrorResponse, "description": "Invalid input"},
    }
)
def predict_price(request: PricePredictionRequest):
    """Predict estimated vehicle resale price."""
    features = request.model_dump()
    features["fuel_type"] = features["fuel_type"].value if hasattr(features["fuel_type"], "value") else features["fuel_type"]
    features["transmission"] = features["transmission"].value if hasattr(features["transmission"], "value") else features["transmission"]

    try:
        result = prediction_service.predict_price(features)
        return PricePredictionResponse(**result)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in price prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during price prediction"
        )


@router.post(
    "/predict/category",
    response_model=CategoryPredictionResponse,
    tags=["Predictions"],
    summary="Classify Vehicle Category",
    description="Classify a vehicle into a category (Budget/Mid-range/Premium) using the trained RandomForestClassifier model.",
    responses={
        503: {"model": ErrorResponse, "description": "Model not available"},
        422: {"model": ErrorResponse, "description": "Invalid input"},
    }
)
def predict_category(request: CategoryPredictionRequest):
    """Classify vehicle into a price-band category."""
    features = request.model_dump()
    features["fuel_type"] = features["fuel_type"].value if hasattr(features["fuel_type"], "value") else features["fuel_type"]
    features["transmission"] = features["transmission"].value if hasattr(features["transmission"], "value") else features["transmission"]

    try:
        result = prediction_service.predict_category(features)
        return CategoryPredictionResponse(**result)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in category prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during category classification"
        )
