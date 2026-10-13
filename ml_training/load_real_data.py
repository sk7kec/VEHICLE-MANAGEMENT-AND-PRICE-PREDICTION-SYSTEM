"""
Real Dataset Loader — Loads and normalizes real CSV vehicle datasets.

POLICY: Only the four approved real CSV files may be used for ML training.
         'vehicles.csv' is synthetic and is STRICTLY PROHIBITED.

Supported datasets:
  1. Car details v3.csv — primary (8128 rows, rich feature set)
  2. CAR DETAILS FROM CAR DEKHO.csv — secondary (4340 rows)
  3. car details v4.csv — supplementary (2059 rows)
  4. car data.csv — supplementary (301 rows, prices in Lakhs)

After loading, all datasets are normalized to the following schema:
  make, model, year, km_driven, fuel_type, transmission, engine_cc,
  previous_owners, selling_price

"""

import os
import re
import pandas as pd
import numpy as np

# ── Blocked Datasets ─────────────────────────────────────────────

BLOCKED_DATASETS = {"vehicles.csv"}

# ── Approved Real Datasets (in preference order) ─────────────────

APPROVED_DATASETS = [
    "Car details v3.csv",
    "CAR DETAILS FROM CAR DEKHO.csv",
    "car details v4.csv",
    "car data.csv",
]

# ── Normalized Schema ────────────────────────────────────────────

NORMALIZED_COLUMNS = [
    "make", "model", "year", "km_driven", "fuel_type",
    "transmission", "engine_cc", "previous_owners", "selling_price",
]

# ── Owner Mapping ────────────────────────────────────────────────

OWNER_MAP = {
    "First Owner": 0, "First": 0,
    "Second Owner": 1, "Second": 1,
    "Third Owner": 2, "Third": 2,
    "Fourth & Above Owner": 3, "Fourth": 3, "4 or More": 3,
    "Test Drive Car": 0, "UnRegistered Car": 0,
}

# ── Fuel Type Normalization ──────────────────────────────────────

FUEL_MAP = {
    "Petrol": "Petrol",
    "Diesel": "Diesel",
    "CNG": "CNG",
    "LPG": "CNG",          # Map LPG → CNG (both are gaseous fuels)
    "Electric": "Electric",
    "Petrol + CNG": "CNG",
    "Petrol + LPG": "CNG",
    "CNG + CNG": "CNG",
    "Hybrid": "Petrol",
}


def _assert_not_blocked(filepath: str):
    """Raise an error if the file is a blocked synthetic dataset."""
    basename = os.path.basename(filepath).strip()
    if basename.lower() in {b.lower() for b in BLOCKED_DATASETS}:
        raise ValueError(
            f"BLOCKED: '{basename}' is a synthetic dataset and MUST NOT be "
            f"used for ML training. Only real CSV datasets are permitted."
        )


def _parse_engine_cc(value) -> float:
    """Extract numeric engine CC from strings like '1248 CC', '1248 cc', etc."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower().replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)", s)
    if match:
        return float(match.group(1))
    return np.nan


def _extract_make_model(name: str):
    """Extract make and model from a combined name string.

    Examples:
        'Maruti Swift Dzire VDI'      → ('Maruti', 'Swift Dzire VDI')
        'Hyundai i20 Sportz Diesel'   → ('Hyundai', 'i20 Sportz Diesel')
        'Maruti Suzuki Swift VXI'     → ('Maruti Suzuki', 'Swift VXI')
    """
    if pd.isna(name):
        return "Unknown", "Unknown"

    name = str(name).strip()
    if not name:
        return "Unknown", "Unknown"

    # Known two-word makes
    two_word_makes = [
        "Maruti Suzuki", "Land Rover", "Aston Martin",
        "Rolls Royce", "Force Motors", "Tata Motors",
    ]
    for make in two_word_makes:
        if name.lower().startswith(make.lower()):
            model_part = name[len(make):].strip()
            return make, model_part if model_part else "Unknown"

    # Single-word make
    parts = name.split(None, 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], "Unknown"


def _map_owner(value) -> int:
    """Convert owner string or numeric to integer count of previous owners."""
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip()
    # Try direct numeric
    try:
        return int(float(s))
    except (ValueError, TypeError):
        pass
    return OWNER_MAP.get(s, 0)


# ── Dataset-Specific Loaders ─────────────────────────────────────

def _load_car_details_v3(filepath: str) -> pd.DataFrame:
    """Load and normalize 'Car details v3.csv'.

    Original columns: name, year, selling_price, km_driven, fuel,
                      seller_type, transmission, owner, mileage,
                      engine, max_power, torque, seats
    """
    print(f"  Loading: {os.path.basename(filepath)} ...", end=" ")
    df = pd.read_csv(filepath)
    print(f"{len(df)} rows")

    # Extract make and model from 'name'
    make_model = df["name"].apply(_extract_make_model)
    df["make"] = make_model.apply(lambda x: x[0])
    df["model"] = make_model.apply(lambda x: x[1])

    # Parse engine CC
    df["engine_cc"] = df["engine"].apply(_parse_engine_cc)

    # Map owner
    df["previous_owners"] = df["owner"].apply(_map_owner)

    # Normalize fuel type
    df["fuel_type"] = df["fuel"].map(FUEL_MAP).fillna("Petrol")

    # Rename columns to match schema
    df = df.rename(columns={"transmission": "transmission"})  # already correct

    # Select and validate
    result = df[NORMALIZED_COLUMNS].copy()
    result["selling_price"] = pd.to_numeric(result["selling_price"], errors="coerce")
    result["year"] = pd.to_numeric(result["year"], errors="coerce")
    result["km_driven"] = pd.to_numeric(result["km_driven"], errors="coerce")
    result["engine_cc"] = pd.to_numeric(result["engine_cc"], errors="coerce")
    result["_source"] = "Car details v3.csv"

    return result


def _load_car_dekho(filepath: str) -> pd.DataFrame:
    """Load and normalize 'CAR DETAILS FROM CAR DEKHO.csv'.

    Original columns: name, year, selling_price, km_driven, fuel,
                      seller_type, transmission, owner
    NOTE: No engine column → will be NaN (dropped later or imputed).
    """
    print(f"  Loading: {os.path.basename(filepath)} ...", end=" ")
    df = pd.read_csv(filepath)
    print(f"{len(df)} rows")

    make_model = df["name"].apply(_extract_make_model)
    df["make"] = make_model.apply(lambda x: x[0])
    df["model"] = make_model.apply(lambda x: x[1])

    df["engine_cc"] = np.nan  # Not available in this dataset
    df["previous_owners"] = df["owner"].apply(_map_owner)
    df["fuel_type"] = df["fuel"].map(FUEL_MAP).fillna("Petrol")

    result = df[NORMALIZED_COLUMNS].copy()
    result["selling_price"] = pd.to_numeric(result["selling_price"], errors="coerce")
    result["year"] = pd.to_numeric(result["year"], errors="coerce")
    result["km_driven"] = pd.to_numeric(result["km_driven"], errors="coerce")
    result["_source"] = "CAR DETAILS FROM CAR DEKHO.csv"

    return result


def _load_car_details_v4(filepath: str) -> pd.DataFrame:
    """Load and normalize 'car details v4.csv'.

    Original columns: Make, Model, Price, Year, Kilometer, Fuel Type,
                      Transmission, Location, Color, Owner, Seller Type,
                      Engine, Max Power, Max Torque, Drivetrain, Length,
                      Width, Height, Seating Capacity, Fuel Tank Capacity
    """
    print(f"  Loading: {os.path.basename(filepath)} ...", end=" ")
    df = pd.read_csv(filepath)
    print(f"{len(df)} rows")

    df["engine_cc"] = df["Engine"].apply(_parse_engine_cc)
    df["previous_owners"] = df["Owner"].apply(_map_owner)
    df["fuel_type"] = df["Fuel Type"].map(FUEL_MAP).fillna("Petrol")

    result = pd.DataFrame({
        "make": df["Make"],
        "model": df["Model"],
        "year": pd.to_numeric(df["Year"], errors="coerce"),
        "km_driven": pd.to_numeric(df["Kilometer"], errors="coerce"),
        "fuel_type": df["fuel_type"],
        "transmission": df["Transmission"],
        "engine_cc": df["engine_cc"],
        "previous_owners": df["previous_owners"],
        "selling_price": pd.to_numeric(df["Price"], errors="coerce"),
    })
    result["_source"] = "car details v4.csv"

    return result


def _load_car_data(filepath: str) -> pd.DataFrame:
    """Load and normalize 'car data.csv'.

    Original columns: Car_Name, Year, Selling_Price, Present_Price,
                      Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner
    NOTE: Selling_Price and Present_Price are in LAKHS. Convert to INR.
          No engine_cc column.
    """
    print(f"  Loading: {os.path.basename(filepath)} ...", end=" ")
    df = pd.read_csv(filepath)
    print(f"{len(df)} rows")

    make_model = df["Car_Name"].apply(_extract_make_model)
    df["make"] = make_model.apply(lambda x: x[0])
    df["model"] = make_model.apply(lambda x: x[1])

    result = pd.DataFrame({
        "make": df["make"],
        "model": df["model"],
        "year": pd.to_numeric(df["Year"], errors="coerce"),
        "km_driven": pd.to_numeric(df["Kms_Driven"], errors="coerce"),
        "fuel_type": df["Fuel_Type"],  # Already: Petrol, Diesel, CNG
        "transmission": df["Transmission"],  # Already: Manual, Automatic
        "engine_cc": np.nan,  # Not available
        "previous_owners": pd.to_numeric(df["Owner"], errors="coerce").fillna(0).astype(int),
        "selling_price": pd.to_numeric(df["Selling_Price"], errors="coerce") * 100000,  # Lakhs → INR
    })
    result["_source"] = "car data.csv"

    return result


# ── Dataset Loader Registry ──────────────────────────────────────

_LOADERS = {
    "Car details v3.csv": _load_car_details_v3,
    "CAR DETAILS FROM CAR DEKHO.csv": _load_car_dekho,
    "car details v4.csv": _load_car_details_v4,
    "car data.csv": _load_car_data,
}


# ── Public API ───────────────────────────────────────────────────

def load_real_dataset(dataset_dir: str, use_datasets: list = None) -> pd.DataFrame:
    """Load and combine real CSV datasets into a normalized DataFrame.

    Args:
        dataset_dir: Path to the dataset/ directory.
        use_datasets: Optional list of specific dataset filenames to use.
                      Defaults to all approved datasets.

    Returns:
        Combined, cleaned DataFrame with normalized schema.

    Raises:
        ValueError: If vehicles.csv is requested, or if no usable data is found.
        FileNotFoundError: If no approved datasets are found in dataset_dir.
    """
    print("=" * 60)
    print("LOADING REAL VEHICLE DATASETS")
    print("=" * 60)

    targets = use_datasets or APPROVED_DATASETS

    # Enforce blocked list
    for t in targets:
        _assert_not_blocked(t)

    # Also verify no blocked file is in the directory listing
    all_files = os.listdir(dataset_dir) if os.path.isdir(dataset_dir) else []
    print(f"\nDataset directory: {dataset_dir}")
    print(f"Files found: {all_files}")

    dfs = []
    for name in targets:
        filepath = os.path.join(dataset_dir, name)
        if not os.path.exists(filepath):
            print(f"  SKIPPED (not found): {name}")
            continue

        _assert_not_blocked(filepath)

        loader = _LOADERS.get(name)
        if loader is None:
            print(f"  SKIPPED (no loader): {name}")
            continue

        try:
            df = loader(filepath)
            dfs.append(df)
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
            continue

    if not dfs:
        raise FileNotFoundError(
            f"No approved real datasets found in '{dataset_dir}'. "
            f"Expected one or more of: {APPROVED_DATASETS}. "
            f"The synthetic 'vehicles.csv' is NOT permitted for training."
        )

    # Combine
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined raw records: {len(combined)}")

    # ── Clean and validate ────────────────────────────────────────

    # Drop rows with missing selling_price or non-positive
    before = len(combined)
    combined = combined.dropna(subset=["selling_price"])
    combined = combined[combined["selling_price"] > 0]
    if len(combined) < before:
        print(f"Dropped {before - len(combined)} rows with invalid selling_price")

    # Drop rows with missing critical features
    critical = ["make", "model", "year", "km_driven", "fuel_type", "transmission"]
    before = len(combined)
    combined = combined.dropna(subset=critical)
    if len(combined) < before:
        print(f"Dropped {before - len(combined)} rows with missing critical features")

    # Drop rows missing engine_cc (can't use for models that require it)
    before = len(combined)
    combined = combined.dropna(subset=["engine_cc"])
    combined = combined[combined["engine_cc"] > 0]
    if len(combined) < before:
        print(f"Dropped {before - len(combined)} rows with missing/invalid engine_cc")

    # Validate year range
    combined = combined[(combined["year"] >= 1980) & (combined["year"] <= 2026)]

    # Validate km_driven
    combined = combined[combined["km_driven"] >= 0]

    # Normalize fuel types to allowed set
    allowed_fuels = {"Petrol", "Diesel", "CNG", "Electric"}
    combined = combined[combined["fuel_type"].isin(allowed_fuels)]

    # Normalize transmission
    combined["transmission"] = combined["transmission"].str.strip()
    combined = combined[combined["transmission"].isin({"Manual", "Automatic"})]

    # Clean string columns
    combined["make"] = combined["make"].str.strip()
    combined["model"] = combined["model"].str.strip()

    # Ensure correct types
    combined["year"] = combined["year"].astype(int)
    combined["km_driven"] = combined["km_driven"].astype(int)
    combined["engine_cc"] = combined["engine_cc"].astype(int)
    combined["previous_owners"] = combined["previous_owners"].fillna(0).astype(int)
    combined["selling_price"] = combined["selling_price"].astype(float)

    # Drop duplicates
    dedup_cols = [c for c in NORMALIZED_COLUMNS]
    before = len(combined)
    combined = combined.drop_duplicates(subset=dedup_cols)
    if len(combined) < before:
        print(f"Dropped {before - len(combined)} duplicate records")

    # Reset index
    combined = combined.reset_index(drop=True)

    print(f"\nFinal cleaned dataset: {len(combined)} records")
    if len(combined) == 0:
        raise ValueError(
            "No usable records remain after cleaning. "
            "Check that real CSV files have valid data."
        )

    # Print summary stats
    print(f"\nDataset Summary:")
    print(f"  Sources: {combined['_source'].value_counts().to_dict()}")
    print(f"  Makes: {combined['make'].nunique()} unique")
    print(f"  Year range: {combined['year'].min()} - {combined['year'].max()}")
    print(f"  Price range: INR {combined['selling_price'].min():,.0f} - INR {combined['selling_price'].max():,.0f}")
    print(f"  Mean price: INR {combined['selling_price'].mean():,.0f}")
    print(f"  Fuel types: {sorted(combined['fuel_type'].unique())}")
    print(f"  Transmissions: {sorted(combined['transmission'].unique())}")

    print("=" * 60)

    return combined


def get_dataset_dir() -> str:
    """Get the default dataset directory path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, "dataset")


if __name__ == "__main__":
    # Quick test: load and display summary
    dataset_dir = get_dataset_dir()
    df = load_real_dataset(dataset_dir)
    print(f"\nLoaded {len(df)} records")
    print(df.head())
    print(df.dtypes)
