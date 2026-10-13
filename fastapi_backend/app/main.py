"""
FastAPI Backend — Main Application Entry Point

Vehicle Management and Price Prediction System

Author: Sahishnakumaar S S (24ADR134)
"""

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    API_TITLE, API_DESCRIPTION, API_VERSION,
    VEHICLES_JSON_PATH, PRICE_MODEL_PATH, CATEGORY_MODEL_PATH,
    PRICE_METADATA_PATH, CATEGORY_METADATA_PATH, HOST, PORT,
)
from .storage.vehicle_repository import VehicleRepository
from .services.vehicle_service import VehicleService
from .services.prediction_service import PredictionService
from .api.routes import router, init_services

# ── Logging Configuration ────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── FastAPI App ───────────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS (allow Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Service Initialization ────────────────────────────────────────

logger.info("Initializing services...")

# Vehicle storage
repository = VehicleRepository(VEHICLES_JSON_PATH)
vehicle_service = VehicleService(repository)
logger.info(f"Vehicle storage initialized: {VEHICLES_JSON_PATH}")

# ML prediction service
prediction_service = PredictionService(
    price_model_path=PRICE_MODEL_PATH,
    category_model_path=CATEGORY_MODEL_PATH,
    price_metadata_path=PRICE_METADATA_PATH,
    category_metadata_path=CATEGORY_METADATA_PATH,
)
logger.info(f"Price model available: {prediction_service.price_model_available}")
logger.info(f"Category model available: {prediction_service.category_model_available}")

# Inject services into routes
init_services(vehicle_service, prediction_service)

# Include router
app.include_router(router)

logger.info("FastAPI application initialized successfully")


# ── Startup / Shutdown Events ─────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    logger.info(f"Swagger UI: http://{HOST}:{PORT}/docs")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
