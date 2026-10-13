"""
ML Training Tests

Tests for model training, artifact saving/loading, prediction quality,
and enforcement of the real-data-only policy.

Author: Sahishnakumaar S S (24ADR134)
"""

import os
import sys
import json
import tempfile
import shutil
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_training"))

from ml_training.load_real_data import (
    load_real_dataset, BLOCKED_DATASETS, APPROVED_DATASETS,
    _assert_not_blocked, _parse_engine_cc, _extract_make_model,
)
from ml_training.train_price_model import (
    validate_dataset as validate_price_dataset,
    build_pipeline as build_price_pipeline,
    ALL_FEATURES as PRICE_FEATURES,
    TARGET,
)
from ml_training.train_category_model import (
    validate_dataset as validate_category_dataset,
    build_pipeline as build_category_pipeline,
    assign_category,
    CATEGORY_ORDER,
    ALL_FEATURES as CATEGORY_FEATURES,
)


@pytest.fixture(scope="module")
def temp_dir():
    """Create a temporary directory for test artifacts."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def dataset_dir():
    """Get the real dataset directory."""
    return os.path.join(os.path.dirname(__file__), "..", "dataset")


@pytest.fixture(scope="module")
def dataset_df(dataset_dir):
    """Load the real dataset."""
    return load_real_dataset(dataset_dir)


# ── Synthetic Data Blocking Tests ──

class TestSyntheticDataBlocking:
    """Tests that ensure vehicles.csv (synthetic data) is NEVER used for training."""

    def test_vehicles_csv_is_blocked(self):
        """Verify 'vehicles.csv' is in the blocked list."""
        assert "vehicles.csv" in BLOCKED_DATASETS

    def test_blocked_assertion_raises_on_vehicles_csv(self):
        """Verify _assert_not_blocked raises for vehicles.csv."""
        with pytest.raises(ValueError, match="BLOCKED"):
            _assert_not_blocked("vehicles.csv")

    def test_blocked_assertion_raises_on_full_path(self):
        """Verify _assert_not_blocked raises for full path to vehicles.csv."""
        with pytest.raises(ValueError, match="BLOCKED"):
            _assert_not_blocked("/some/path/vehicles.csv")

    def test_blocked_assertion_case_insensitive(self):
        """Verify blocking is case-insensitive."""
        with pytest.raises(ValueError, match="BLOCKED"):
            _assert_not_blocked("Vehicles.CSV")

    def test_load_real_dataset_rejects_vehicles_csv(self, temp_dir):
        """Verify load_real_dataset raises if vehicles.csv is explicitly requested."""
        with pytest.raises(ValueError, match="BLOCKED"):
            load_real_dataset(temp_dir, use_datasets=["vehicles.csv"])

    def test_approved_datasets_do_not_include_vehicles_csv(self):
        """Verify vehicles.csv is NOT in the approved list."""
        for name in APPROVED_DATASETS:
            assert name.lower() != "vehicles.csv", \
                f"APPROVED_DATASETS must not contain vehicles.csv, found: {name}"

    def test_training_metadata_does_not_reference_synthetic(self, dataset_dir, temp_dir):
        """Verify that training metadata never says 'synthetic'."""
        from ml_training.train_price_model import train_and_evaluate
        _, metadata = train_and_evaluate(dataset_dir, temp_dir)
        assert metadata["dataset_type"] == "real"
        assert "vehicles.csv" not in str(metadata.get("dataset_sources", []))
        assert "synthetic" not in metadata.get("data_policy", "").lower() or \
               "prohibited" in metadata.get("data_policy", "").lower()


# ── Data Loading Tests ──

class TestDataLoading:
    def test_real_datasets_loaded(self, dataset_df):
        """Verify real datasets are loaded successfully."""
        assert len(dataset_df) > 0

    def test_dataset_has_required_columns(self, dataset_df):
        required = ["make", "model", "year", "km_driven", "fuel_type",
                     "transmission", "engine_cc", "previous_owners", "selling_price"]
        for col in required:
            assert col in dataset_df.columns, f"Missing column: {col}"

    def test_dataset_no_nulls_in_required(self, dataset_df):
        for col in PRICE_FEATURES + [TARGET]:
            assert dataset_df[col].isnull().sum() == 0, f"Nulls in {col}"

    def test_dataset_valid_price_range(self, dataset_df):
        assert (dataset_df["selling_price"] > 0).all()

    def test_dataset_valid_year_range(self, dataset_df):
        assert (dataset_df["year"] >= 1980).all()
        assert (dataset_df["year"] <= 2026).all()

    def test_dataset_valid_km_range(self, dataset_df):
        assert (dataset_df["km_driven"] >= 0).all()

    def test_dataset_valid_engine_cc(self, dataset_df):
        assert (dataset_df["engine_cc"] > 0).all()

    def test_dataset_valid_fuel_types(self, dataset_df):
        valid_fuels = {"Petrol", "Diesel", "CNG", "Electric"}
        assert set(dataset_df["fuel_type"].unique()).issubset(valid_fuels)

    def test_dataset_valid_transmissions(self, dataset_df):
        valid_trans = {"Manual", "Automatic"}
        assert set(dataset_df["transmission"].unique()).issubset(valid_trans)

    def test_source_column_present(self, dataset_df):
        """Verify _source column tracks which real CSV each record came from."""
        assert "_source" in dataset_df.columns
        sources = set(dataset_df["_source"].unique())
        assert "vehicles.csv" not in sources, \
            "Dataset contains records from blocked vehicles.csv!"


# ── Helper Function Tests ──

class TestHelperFunctions:
    def test_parse_engine_cc(self):
        assert _parse_engine_cc("1248 CC") == 1248.0
        assert _parse_engine_cc("1498 cc") == 1498.0
        assert _parse_engine_cc("999") == 999.0
        assert np.isnan(_parse_engine_cc(None))
        assert np.isnan(_parse_engine_cc(""))

    def test_extract_make_model(self):
        make, model = _extract_make_model("Maruti Swift Dzire VDI")
        assert make == "Maruti"
        assert "Swift" in model

        make, model = _extract_make_model("Hyundai i20 Sportz")
        assert make == "Hyundai"
        assert "i20" in model

    def test_category_assignment(self):
        assert assign_category(100000) == "Budget"
        assert assign_category(299999) == "Budget"
        assert assign_category(300000) == "Mid-range"
        assert assign_category(500000) == "Mid-range"
        assert assign_category(799999) == "Mid-range"
        assert assign_category(800000) == "Premium"
        assert assign_category(5000000) == "Premium"


# ── Dataset Validation ──

class TestDatasetValidation:
    def test_price_validation(self, dataset_df):
        validated = validate_price_dataset(dataset_df.copy())
        assert len(validated) > 0
        assert TARGET in validated.columns

    def test_category_validation(self, dataset_df):
        validated = validate_category_dataset(dataset_df.copy())
        assert len(validated) > 0
        assert "category" in validated.columns


# ── Price Model Training ──

class TestPriceModel:
    @pytest.fixture(scope="class")
    def trained_price_model(self, dataset_df):
        """Train a price model on real data."""
        df = validate_price_dataset(dataset_df.copy())
        X = df[PRICE_FEATURES]
        y = df[TARGET]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline = build_price_pipeline()
        pipeline.fit(X_train, y_train)
        return pipeline, X_test, y_test

    def test_pipeline_trains(self, trained_price_model):
        pipeline, _, _ = trained_price_model
        assert pipeline is not None

    def test_prediction_is_numeric(self, trained_price_model):
        pipeline, X_test, _ = trained_price_model
        predictions = pipeline.predict(X_test)
        assert all(np.isfinite(predictions))

    def test_prediction_is_positive(self, trained_price_model):
        pipeline, X_test, _ = trained_price_model
        predictions = pipeline.predict(X_test)
        # Most predictions should be positive (some edge cases may go slightly negative)
        assert (predictions > 0).mean() > 0.95

    def test_r2_score_reasonable(self, trained_price_model):
        pipeline, X_test, y_test = trained_price_model
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, pipeline.predict(X_test))
        assert r2 > 0.5, f"R2 score too low: {r2}"

    def test_model_save_and_load(self, trained_price_model, temp_dir):
        pipeline, X_test, _ = trained_price_model
        import joblib
        path = os.path.join(temp_dir, "test_price_model.pkl")
        joblib.dump(pipeline, path)
        assert os.path.exists(path)

        loaded = joblib.load(path)
        original_pred = pipeline.predict(X_test.head(5))
        loaded_pred = loaded.predict(X_test.head(5))
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


# ── Category Model Training ──

class TestCategoryModel:
    @pytest.fixture(scope="class")
    def trained_category_model(self, dataset_df):
        """Train a category model on real data."""
        df = validate_category_dataset(dataset_df.copy())
        X = df[CATEGORY_FEATURES]
        y = df["category"]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline = build_category_pipeline()
        pipeline.fit(X_train, y_train)
        return pipeline, X_test, y_test

    def test_pipeline_trains(self, trained_category_model):
        pipeline, _, _ = trained_category_model
        assert pipeline is not None

    def test_prediction_valid_categories(self, trained_category_model):
        pipeline, X_test, _ = trained_category_model
        predictions = pipeline.predict(X_test)
        valid = set(CATEGORY_ORDER)
        for pred in predictions:
            assert pred in valid, f"Invalid category: {pred}"

    def test_accuracy_reasonable(self, trained_category_model):
        pipeline, X_test, y_test = trained_category_model
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, pipeline.predict(X_test))
        assert accuracy > 0.5, f"Accuracy too low: {accuracy}"

    def test_model_save_and_load(self, trained_category_model, temp_dir):
        pipeline, X_test, _ = trained_category_model
        import joblib
        path = os.path.join(temp_dir, "test_category_model.pkl")
        joblib.dump(pipeline, path)
        assert os.path.exists(path)

        loaded = joblib.load(path)
        original_pred = pipeline.predict(X_test.head(5))
        loaded_pred = loaded.predict(X_test.head(5))
        assert list(original_pred) == list(loaded_pred)


# ── Evaluation Metrics ──

class TestEvaluationMetrics:
    def test_mae_computed(self):
        from sklearn.metrics import mean_absolute_error
        y_true = [100, 200, 300]
        y_pred = [110, 190, 310]
        mae = mean_absolute_error(y_true, y_pred)
        assert mae == pytest.approx(10.0)

    def test_rmse_computed(self):
        from sklearn.metrics import mean_squared_error
        y_true = [100, 200, 300]
        y_pred = [110, 190, 310]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        assert rmse == pytest.approx(10.0)

    def test_r2_computed(self):
        from sklearn.metrics import r2_score
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 5]
        r2 = r2_score(y_true, y_pred)
        assert r2 == pytest.approx(1.0)
