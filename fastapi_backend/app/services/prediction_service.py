"""
Prediction Service — ML model loading and inference.

Loads trained model pipelines and provides prediction methods.

Author: Sahishnakumaar S S (24ADR134)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class PredictionService:
    """Handles ML model loading and inference for price and category predictions."""

    def __init__(self, price_model_path: str, category_model_path: str,
                 price_metadata_path: str = None, category_metadata_path: str = None):
        self._price_model_path = price_model_path
        self._category_model_path = category_model_path
        self._price_metadata_path = price_metadata_path
        self._category_metadata_path = category_metadata_path

        self._price_model = None
        self._category_model = None
        self._price_metadata = None
        self._category_metadata = None

        self._load_models()

    def _load_models(self):
        """Load model artifacts at startup."""
        # Price model
        if os.path.exists(self._price_model_path):
            try:
                self._price_model = joblib.load(self._price_model_path)
                logger.info(f"Price model loaded from: {self._price_model_path}")
            except Exception as e:
                logger.error(f"Failed to load price model: {e}")
        else:
            logger.warning(f"Price model not found: {self._price_model_path}")

        # Category model
        if os.path.exists(self._category_model_path):
            try:
                self._category_model = joblib.load(self._category_model_path)
                logger.info(f"Category model loaded from: {self._category_model_path}")
            except Exception as e:
                logger.error(f"Failed to load category model: {e}")
        else:
            logger.warning(f"Category model not found: {self._category_model_path}")

        # Price metadata
        if self._price_metadata_path and os.path.exists(self._price_metadata_path):
            try:
                with open(self._price_metadata_path, "r", encoding="utf-8") as f:
                    self._price_metadata = json.load(f)
                logger.info("Price model metadata loaded")
            except Exception as e:
                logger.error(f"Failed to load price metadata: {e}")

        # Category metadata
        if self._category_metadata_path and os.path.exists(self._category_metadata_path):
            try:
                with open(self._category_metadata_path, "r", encoding="utf-8") as f:
                    self._category_metadata = json.load(f)
                logger.info("Category model metadata loaded")
            except Exception as e:
                logger.error(f"Failed to load category metadata: {e}")

    @property
    def price_model_available(self) -> bool:
        return self._price_model is not None

    @property
    def category_model_available(self) -> bool:
        return self._category_model is not None

    def predict_price(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict vehicle resale price."""
        if not self.price_model_available:
            raise RuntimeError(
                "Price prediction model is not available. "
                "Please run the training script first: python ml_training/train_price_model.py"
            )

        # Build features DataFrame
        input_df = pd.DataFrame([{
            "make": str(features["make"]).strip(),
            "model": str(features["model"]).strip(),
            "year": int(features["year"]),
            "km_driven": int(features["km_driven"]),
            "fuel_type": str(features["fuel_type"]).strip(),
            "transmission": str(features["transmission"]).strip(),
            "engine_cc": int(features["engine_cc"]),
            "previous_owners": int(features.get("previous_owners", 0)),
        }])

        try:
            prediction = self._price_model.predict(input_df)
            predicted_price = float(prediction[0])

            # Ensure prediction is finite and positive
            if not np.isfinite(predicted_price):
                raise ValueError("Model returned a non-finite prediction")
            predicted_price = max(0, round(predicted_price, 2))

        except Exception as e:
            logger.error(f"Price prediction failed: {e}")
            raise RuntimeError(f"Price prediction failed: {str(e)}")

        return {
            "predicted_price": predicted_price,
            "currency": "INR",
            "model_name": self._price_metadata.get("model_name", "vehicle_price_random_forest") if self._price_metadata else "vehicle_price_random_forest",
            "input_summary": {
                "make": features["make"],
                "model": features["model"],
                "year": features["year"],
                "km_driven": features["km_driven"],
                "fuel_type": features["fuel_type"],
                "transmission": features["transmission"],
                "engine_cc": features["engine_cc"],
                "previous_owners": features.get("previous_owners", 0),
            },
            "disclaimer": (
                "This is an estimated price based on a machine learning model trained on real market data. "
                "It is not a guaranteed sale price or certified valuation."
            )
        }

    def predict_category(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict vehicle category (Budget/Mid-range/Premium)."""
        if not self.category_model_available:
            raise RuntimeError(
                "Category classification model is not available. "
                "Please run the training script first: python ml_training/train_category_model.py"
            )

        input_df = pd.DataFrame([{
            "make": str(features["make"]).strip(),
            "model": str(features["model"]).strip(),
            "year": int(features["year"]),
            "km_driven": int(features["km_driven"]),
            "fuel_type": str(features["fuel_type"]).strip(),
            "transmission": str(features["transmission"]).strip(),
            "engine_cc": int(features["engine_cc"]),
            "previous_owners": int(features.get("previous_owners", 0)),
        }])

        try:
            prediction = self._category_model.predict(input_df)
            predicted_category = str(prediction[0])
        except Exception as e:
            logger.error(f"Category prediction failed: {e}")
            raise RuntimeError(f"Category prediction failed: {str(e)}")

        categories = ["Budget", "Mid-range", "Premium"]
        if self._category_metadata:
            categories = self._category_metadata.get("categories", categories)

        category_descriptions = {
            "Budget": "Vehicles with estimated resale price below ₹3,00,000",
            "Mid-range": "Vehicles with estimated resale price between ₹3,00,000 and ₹8,00,000",
            "Premium": "Vehicles with estimated resale price above ₹8,00,000",
        }

        return {
            "predicted_category": predicted_category,
            "model_name": self._category_metadata.get("model_name", "vehicle_category_classifier") if self._category_metadata else "vehicle_category_classifier",
            "categories": categories,
            "category_description": category_descriptions.get(
                predicted_category,
                f"Vehicle classified as: {predicted_category}"
            ),
            "input_summary": {
                "make": features["make"],
                "model": features["model"],
                "year": features["year"],
                "km_driven": features["km_driven"],
                "fuel_type": features["fuel_type"],
                "transmission": features["transmission"],
                "engine_cc": features["engine_cc"],
                "previous_owners": features.get("previous_owners", 0),
            },
            "disclaimer": (
                "Category is based on price bands derived from real market data. "
                "It represents a price segment, not a quality or condition assessment."
            )
        }
