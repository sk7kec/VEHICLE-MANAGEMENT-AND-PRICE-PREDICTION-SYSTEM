"""
Vehicle Category Classification Model Training Script

Trains a RandomForestClassifier pipeline for classifying vehicles into
price-band categories: Budget, Mid-range, Premium.

Category definitions (price-band based):
- Budget:    selling_price < 3,00,000 INR
- Mid-range: 3,00,000 <= selling_price < 8,00,000 INR
- Premium:   selling_price >= 8,00,000 INR

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import joblib

warnings.filterwarnings("ignore")

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_real_data import load_real_dataset, get_dataset_dir, BLOCKED_DATASETS

SEED = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = ["year", "km_driven", "engine_cc", "previous_owners"]
CATEGORICAL_FEATURES = ["make", "model", "fuel_type", "transmission"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORY_BOUNDARIES = {
    "Budget": (0, 300000),
    "Mid-range": (300000, 800000),
    "Premium": (800000, float("inf")),
}

CATEGORY_ORDER = ["Budget", "Mid-range", "Premium"]


def assign_category(price):
    """Assign a price-band category based on selling price."""
    if price < 300000:
        return "Budget"
    elif price < 800000:
        return "Mid-range"
    else:
        return "Premium"


def validate_dataset(df):
    """Validate the dataset for classification training."""
    required_columns = ALL_FEATURES + ["selling_price"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

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

    # Clean
    df = df.dropna(subset=ALL_FEATURES + ["selling_price"])
    df = df[df["selling_price"] > 0]
    df = df.drop_duplicates()

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()

    df = df.dropna(subset=ALL_FEATURES)

    # Assign categories
    df["category"] = df["selling_price"].apply(assign_category)

    print(f"Validated dataset: {len(df)} records")
    print(f"Category distribution:\n{df['category'].value_counts().to_string()}")

    return df.reset_index(drop=True)


def build_pipeline():
    """Build the preprocessing + classifier pipeline."""
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
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    return pipeline


def train_and_evaluate(dataset_dir, artifacts_dir):
    """Train the category classification model using real datasets and save artifacts."""
    print("=" * 60)
    print("VEHICLE CATEGORY CLASSIFICATION MODEL TRAINING")
    print("(Using REAL datasets only -- vehicles.csv is BLOCKED)")
    print("=" * 60)

    # Load real datasets
    print(f"\nLoading real datasets from: {dataset_dir}")
    df = load_real_dataset(dataset_dir)

    # Track which real files were used
    sources = df["_source"].value_counts().to_dict() if "_source" in df.columns else {}
    source_files = list(sources.keys())
    print(f"Real dataset sources: {sources}")

    # Validate and assign categories
    df = validate_dataset(df)

    # Features and target
    X = df[ALL_FEATURES]
    y = df["category"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # Build and train
    pipeline = build_pipeline()
    print("\nTraining RandomForestClassifier pipeline...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n--- Evaluation on Test Set ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print(f"\nClassification Report:")
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=CATEGORY_ORDER)
    print(f"Confusion Matrix (rows=actual, cols=predicted):")
    print(f"Labels: {CATEGORY_ORDER}")
    print(cm)

    # Save model
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "vehicle_category_classifier.pkl")
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")

    # Per-class metrics
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Save metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "model_name": "vehicle_category_classifier",
        "training_date": datetime.now().isoformat(),
        "dataset_sources": source_files,
        "dataset_type": "real",
        "total_records": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": "category (derived from selling_price)",
        "categories": CATEGORY_ORDER,
        "category_boundaries": {
            "Budget": "selling_price < INR 3,00,000",
            "Mid-range": "INR 3,00,000 <= selling_price < INR 8,00,000",
            "Premium": "selling_price >= INR 8,00,000",
        },
        "labeling_methodology": "Categories derived from selling price bands in real datasets. These represent price segments, not vehicle quality or condition.",
        "random_seed": SEED,
        "test_split": TEST_SIZE,
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
        },
        "evaluation_metrics": {
            "accuracy": round(accuracy, 4),
            "precision_weighted": round(precision, 4),
            "recall_weighted": round(recall, 4),
            "f1_score_weighted": round(f1, 4),
            "per_class": {
                cat: {
                    "precision": round(report_dict.get(cat, {}).get("precision", 0), 4),
                    "recall": round(report_dict.get(cat, {}).get("recall", 0), 4),
                    "f1_score": round(report_dict.get(cat, {}).get("f1-score", 0), 4),
                    "support": int(report_dict.get(cat, {}).get("support", 0)),
                }
                for cat in CATEGORY_ORDER
            },
            "confusion_matrix": cm.tolist(),
        },
        "data_policy": "Trained exclusively on real CSV datasets. Synthetic 'vehicles.csv' is prohibited.",
    }

    metadata_path = os.path.join(artifacts_dir, "category_model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    print("\n" + "=" * 60)
    print("CATEGORY MODEL TRAINING COMPLETE")
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
