"""
Pydantic Schemas for the Vehicle Management API

Defines request/response models with validation.

Author: Sahishnakumaar S S (24ADR134)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class FuelType(str, Enum):
    PETROL = "Petrol"
    DIESEL = "Diesel"
    CNG = "CNG"
    ELECTRIC = "Electric"


class TransmissionType(str, Enum):
    MANUAL = "Manual"
    AUTOMATIC = "Automatic"


# ── Vehicle Schemas ──────────────────────────────────────────────

class VehicleCreate(BaseModel):
    """Schema for creating a new vehicle record."""
    make: str = Field(
        ..., min_length=1, max_length=100,
        description="Vehicle manufacturer/brand",
        examples=["Maruti Suzuki"]
    )
    model: str = Field(
        ..., min_length=1, max_length=100,
        description="Vehicle model name",
        examples=["Swift"]
    )
    year: int = Field(
        ..., ge=1900, le=2030,
        description="Manufacturing year",
        examples=[2020]
    )
    km_driven: int = Field(
        ..., ge=0, le=2000000,
        description="Kilometres driven",
        examples=[35000]
    )
    fuel_type: FuelType = Field(
        ..., description="Fuel type",
        examples=["Petrol"]
    )
    transmission: TransmissionType = Field(
        ..., description="Transmission type",
        examples=["Manual"]
    )
    engine_cc: int = Field(
        ..., ge=50, le=10000,
        description="Engine capacity in CC",
        examples=[1197]
    )
    previous_owners: Optional[int] = Field(
        default=0, ge=0, le=20,
        description="Number of previous owners",
        examples=[1]
    )

    @field_validator("make", "model")
    @classmethod
    def strip_whitespace(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Field must not be empty or whitespace only")
        return v


class VehicleUpdate(BaseModel):
    """Schema for full vehicle update (PUT)."""
    make: str = Field(..., min_length=1, max_length=100)
    model: str = Field(..., min_length=1, max_length=100)
    year: int = Field(..., ge=1900, le=2030)
    km_driven: int = Field(..., ge=0, le=2000000)
    fuel_type: FuelType
    transmission: TransmissionType
    engine_cc: int = Field(..., ge=50, le=10000)
    previous_owners: Optional[int] = Field(default=0, ge=0, le=20)

    @field_validator("make", "model")
    @classmethod
    def strip_whitespace(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Field must not be empty or whitespace only")
        return v


class VehiclePatch(BaseModel):
    """Schema for partial vehicle update (PATCH)."""
    make: Optional[str] = Field(default=None, min_length=1, max_length=100)
    model: Optional[str] = Field(default=None, min_length=1, max_length=100)
    year: Optional[int] = Field(default=None, ge=1900, le=2030)
    km_driven: Optional[int] = Field(default=None, ge=0, le=2000000)
    fuel_type: Optional[FuelType] = None
    transmission: Optional[TransmissionType] = None
    engine_cc: Optional[int] = Field(default=None, ge=50, le=10000)
    previous_owners: Optional[int] = Field(default=None, ge=0, le=20)

    @field_validator("make", "model")
    @classmethod
    def strip_whitespace(cls, v):
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Field must not be empty or whitespace only")
        return v


class VehicleResponse(BaseModel):
    """Schema for a vehicle record in API responses."""
    id: str = Field(..., description="Unique vehicle ID")
    make: str
    model: str
    year: int
    km_driven: int
    fuel_type: str
    transmission: str
    engine_cc: int
    previous_owners: int
    created_at: str
    updated_at: str


class VehicleListResponse(BaseModel):
    """Schema for listing all vehicles."""
    total: int = Field(..., description="Total number of vehicle records")
    vehicles: List[VehicleResponse]


class DeleteResponse(BaseModel):
    """Schema for successful deletion."""
    message: str
    id: str


# ── Prediction Schemas ───────────────────────────────────────────

class PricePredictionRequest(BaseModel):
    """Schema for price prediction input."""
    make: str = Field(
        ..., min_length=1, max_length=100,
        description="Vehicle manufacturer/brand",
        examples=["Hyundai"]
    )
    model: str = Field(
        ..., min_length=1, max_length=100,
        description="Vehicle model name",
        examples=["Creta"]
    )
    year: int = Field(
        ..., ge=1900, le=2030,
        description="Manufacturing year",
        examples=[2019]
    )
    km_driven: int = Field(
        ..., ge=0, le=2000000,
        description="Kilometres driven",
        examples=[45000]
    )
    fuel_type: FuelType = Field(
        ..., description="Fuel type",
        examples=["Diesel"]
    )
    transmission: TransmissionType = Field(
        ..., description="Transmission type",
        examples=["Automatic"]
    )
    engine_cc: int = Field(
        ..., ge=50, le=10000,
        description="Engine capacity in CC",
        examples=[1493]
    )
    previous_owners: Optional[int] = Field(
        default=0, ge=0, le=20,
        description="Number of previous owners",
        examples=[1]
    )

    @field_validator("make", "model")
    @classmethod
    def strip_whitespace(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Field must not be empty or whitespace only")
        return v


class PricePredictionResponse(BaseModel):
    """Schema for price prediction output."""
    predicted_price: float = Field(..., description="Estimated resale price in INR")
    currency: str = Field(default="INR", description="Currency code")
    model_name: str = Field(..., description="ML model used")
    input_summary: Dict[str, Any] = Field(..., description="Input features summary")
    disclaimer: str = Field(
        default="This is an estimated price based on a machine learning model trained on real market data. "
                "It is not a guaranteed sale price or certified valuation."
    )


class CategoryPredictionRequest(BaseModel):
    """Schema for category classification input."""
    make: str = Field(..., min_length=1, max_length=100, examples=["Tata"])
    model: str = Field(..., min_length=1, max_length=100, examples=["Nexon"])
    year: int = Field(..., ge=1900, le=2030, examples=[2021])
    km_driven: int = Field(..., ge=0, le=2000000, examples=[25000])
    fuel_type: FuelType = Field(..., examples=["Petrol"])
    transmission: TransmissionType = Field(..., examples=["Manual"])
    engine_cc: int = Field(..., ge=50, le=10000, examples=[1199])
    previous_owners: Optional[int] = Field(default=0, ge=0, le=20, examples=[0])

    @field_validator("make", "model")
    @classmethod
    def strip_whitespace(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Field must not be empty or whitespace only")
        return v


class CategoryPredictionResponse(BaseModel):
    """Schema for category classification output."""
    predicted_category: str = Field(..., description="Predicted vehicle category")
    model_name: str = Field(..., description="ML model used")
    categories: List[str] = Field(..., description="All possible categories")
    category_description: str = Field(..., description="What this category represents")
    input_summary: Dict[str, Any] = Field(..., description="Input features summary")
    disclaimer: str = Field(
        default="Category is based on price bands derived from real market data. "
                "It represents a price segment, not a quality or condition assessment."
    )


# ── Status Schemas ───────────────────────────────────────────────

class StatusResponse(BaseModel):
    """Schema for API status/health response."""
    status: str = "running"
    project: str = "Vehicle Management and Price Prediction System"
    version: str = "1.0.0"
    author: str = "Sahishnakumaar S S (24ADR134)"
    endpoints: Dict[str, str]


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    detail: str
