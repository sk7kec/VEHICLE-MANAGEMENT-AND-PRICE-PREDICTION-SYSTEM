"""
Backend Configuration

Centralized configuration for the FastAPI backend.

Author: Sahishnakumaar S S (24ADR134)
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

# Storage
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
VEHICLES_JSON_PATH = os.path.join(DATA_DIR, "vehicles.json")

# Model artifacts
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.join(PROJECT_ROOT, "artifacts"))
PRICE_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "vehicle_price_random_forest.pkl")
CATEGORY_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "vehicle_category_classifier.pkl")
PRICE_METADATA_PATH = os.path.join(ARTIFACTS_DIR, "price_model_metadata.json")
CATEGORY_METADATA_PATH = os.path.join(ARTIFACTS_DIR, "category_model_metadata.json")

# API settings
API_TITLE = "Vehicle Management and Price Prediction API"
API_DESCRIPTION = """
## Vehicle Management and Price Prediction System

A RESTful API for managing vehicle records and predicting used vehicle resale prices
and categories using machine learning.

### Modules
- **Vehicle Management**: Full CRUD operations for vehicle records
- **Price Prediction**: Estimated resale price using RandomForestRegressor
- **Category Classification**: Vehicle category (Budget/Mid-range/Premium) using RandomForestClassifier

### Author
- **Name**: Sahishnakumaar S S
- **Roll Number**: 24ADR134
"""
API_VERSION = "1.0.0"

# Server
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
