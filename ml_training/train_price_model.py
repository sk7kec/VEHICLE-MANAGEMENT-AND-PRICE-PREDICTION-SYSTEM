"""
Vehicle Price Prediction Model Training Script

Trains a RandomForestRegressor pipeline for predicting used vehicle resale prices.
Uses ONLY real CSV datasets from the dataset/ directory.
The synthetic 'vehicles.csv' is STRICTLY PROHIBITED.

Author: Sahishnakumaar S S (24ADR134)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings("ignore")

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_real_data import load_real_dataset, get_dataset_dir, BLOCKED_DATASETS

SEED = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = ["year", "km_driven", "engine_cc", "previous_owners"]
CATEGORICAL_FEATURES = ["make", "model", "fuel_type", "transmission"]
TARGET = "selling_price"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def validate_dataset(df):
    """Validate the dataset schema and data quality."""
    required_columns = ALL_FEATURES + [TARGET]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check for empty dataset
    if len(df) == 0:
        raise ValueError("Dataset is empty")

    # Enforce: reject if dataset came from a blocked source
    if "_source" in df.columns:
        blocked_sources = {b for b in BLOCKED_DATASETS}
        actual_sources = set(df["_source"].unique())
        overlap = actual_sources & blocked_sources
        if overlap:
            raise ValueError(
                f"BLOCKED: Dataset contains records from prohibited source(s): {overlap}. "
                f"Only real CSV datasets are permitted for training."
            )

    # Check target column
    if df[TARGET].isnull().any():
        null_count = df[TARGET].isnull().sum()
        print(f"WARNING: {null_count} null values in target column, dropping them")
        df = df.dropna(subset=[TARGET])

    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        raise ValueError(f"Target column '{TARGET}' must be numeric")

    if (df[TARGET] <= 0).any():
        bad_count = (df[TARGET] <= 0).sum()
        print(f"WARNING: {bad_count} non-positive target values, removing them")
        df = df[df[TARGET] > 0]

    # Check numeric features
    for col in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"WARNING: Converting {col} to numeric")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Check categorical features
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        print(f"Removed {before - len(df)} duplicate records")

    # Drop rows with missing feature values
    before = len(df)
    df = df.dropna(subset=ALL_FEATURES)
    if len(df) < before:
        print(f"Removed {before - len(df)} rows with missing feature values")

    # Validate ranges
    if "year" in df.columns:
        invalid_years = (df["year"] < 1980) | (df["year"] > 2026)
        if invalid_years.any():
            print(f"WARNING: {invalid_years.sum()} records with year outside 1980-2026")
            df = df[~invalid_years]

    if "km_driven" in df.columns:
        invalid_km = df["km_driven"] < 0
        if invalid_km.any():
            print(f"WARNING: {invalid_km.sum()} records with negative km_driven")
            df = df[~invalid_km]

    if "engine_cc" in df.columns:
        invalid_cc = df["engine_cc"] <= 0
        if invalid_cc.any():
            print(f"WARNING: {invalid_cc.sum()} records with invalid engine_cc")
            df = df[~invalid_cc]

    print(f"Validated dataset: {len(df)} records")
    return df.reset_index(drop=True)


def build_pipeline():
    """Build the preprocessing + model pipeline."""
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1
        ))
    ])

    return pipeline


def train_and_evaluate(dataset_dir, artifacts_dir):
    """Train the price prediction model using real datasets and save artifacts."""
    print("=" * 60)
    print("VEHICLE PRICE PREDICTION MODEL TRAINING")
    print("(Using REAL datasets only — vehicles.csv is BLOCKED)")
    print("=" * 60)

    # Load real datasets
    print(f"\nLoading real datasets from: {dataset_dir}")
    df = load_real_dataset(dataset_dir)

    # Track which real files were used
    sources = df["_source"].value_counts().to_dict() if "_source" in df.columns else {}
    source_files = list(sources.keys())
    print(f"Real dataset sources: {sources}")

    # Validate
    df = validate_dataset(df)

    # Features and target
    X = df[ALL_FEATURES]
    y = df[TARGET]

    print(f"\nFeatures: {ALL_FEATURES}")
    print(f"Target: {TARGET}")
    print(f"Target stats: mean={y.mean():.0f}, std={y.std():.0f}, min={y.min():.0f}, max={y.max():.0f}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # Build and train
    pipeline = build_pipeline()
    print("\nTraining RandomForestRegressor pipeline...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Evaluation on Test Set ---")
    print(f"MAE:  INR {mae:,.0f}")
    print(f"RMSE: INR {rmse:,.0f}")
    print(f"R2:   {r2:.4f}")

    # Save model
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "vehicle_price_random_forest.pkl")
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save metadata
    metadata = {
        "model_type": "RandomForestRegressor",
        "model_name": "vehicle_price_random_forest",
        "training_date": datetime.now().isoformat(),
        "dataset_sources": source_files,
        "dataset_type": "real",
        "total_records": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "random_seed": SEED,
        "test_split": TEST_SIZE,
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        },
        "evaluation_metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2_score": round(r2, 4),
        },
        "data_policy": "Trained exclusively on real CSV datasets. Synthetic 'vehicles.csv' is prohibited.",
    }

    metadata_path = os.path.join(artifacts_dir, "price_model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    print("\n" + "=" * 60)
    print("PRICE MODEL TRAINING COMPLETE")
    print("=" * 60)

    return pipeline, metadata


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_root, "dataset")
    artifacts_dir = os.path.join(project_root, "artifacts")

    # Verify vehicles.csv is NOT being used
    print("DATA POLICY: Only real CSV datasets are permitted for training.")
    print("             'vehicles.csv' (synthetic) is STRICTLY BLOCKED.\n")

    train_and_evaluate(dataset_dir, artifacts_dir)
