# Vehicle Dataset Documentation

## Datasets

This directory contains **four real vehicle market datasets** and one synthetic dataset:

### Real Datasets (Approved for ML Training)

| File | Records | Source | Key Features |
|------|---------|--------|-------------|
| `Car details v3.csv` | 8,128 | Real market data | name, year, selling_price, km_driven, fuel, transmission, engine, owner |
| `CAR DETAILS FROM CAR DEKHO.csv` | 4,340 | CarDekho listings | name, year, selling_price, km_driven, fuel, transmission, owner |
| `car details v4.csv` | 2,059 | Real market data | Make, Model, Price, Year, Kilometer, Fuel Type, Transmission, Engine, Owner |
| `car data.csv` | 301 | Real market data | Car_Name, Year, Selling_Price (Lakhs), Kms_Driven, Fuel_Type, Transmission |

### Synthetic Dataset (BLOCKED from ML Training)

| File | Status |
|------|--------|
| `vehicles.csv` | **STRICTLY PROHIBITED** for ML training, validation, testing, or evaluation |

## Data Policy

- **Only the four real CSV datasets** are permitted for ML model training and evaluation.
- `vehicles.csv` is synthetically generated and is **explicitly blocked** by `ml_training/load_real_data.py`.
- The `load_real_data.py` module automatically loads, normalizes, and combines the real datasets.
- If `vehicles.csv` is passed to the loader, a `ValueError` is raised immediately.
- Automated tests in `tests/test_ml_training.py` verify this restriction.

## Normalized Schema

After loading through `load_real_data.py`, all datasets are normalized to:

| Column | Type | Description | Example |
|---|---|---|---|
| make | string | Vehicle manufacturer/brand | Maruti Suzuki |
| model | string | Vehicle model name | Swift |
| year | integer | Manufacturing year | 2018 |
| km_driven | integer | Kilometres driven | 45000 |
| fuel_type | string | Fuel type (Petrol/Diesel/CNG/Electric) | Petrol |
| transmission | string | Transmission type (Manual/Automatic) | Manual |
| engine_cc | integer | Engine displacement in CC | 1197 |
| previous_owners | integer | Number of previous owners (0-4) | 1 |
| selling_price | float | Resale selling price in INR | 485000.0 |

## Category Labels (Derived)

Vehicle categories are derived from selling price bands:
- **Budget**: selling_price < INR 3,00,000
- **Mid-range**: INR 3,00,000 <= selling_price < INR 8,00,000
- **Premium**: selling_price >= INR 8,00,000

These categories represent **price bands**, not assessments of vehicle quality or condition.

## Data Loading

```bash
cd vehicle_management_prediction
python -m ml_training.load_real_data
```

This loads, normalizes, and validates all real CSV datasets.

## Notes on Data Normalization

- `Car details v3.csv` and `CAR DETAILS FROM CAR DEKHO.csv`: Vehicle name is split into make and model.
- `car data.csv`: Prices are converted from Lakhs to INR (multiplied by 100,000).
- `car details v4.csv`: Engine CC is parsed from strings like "1248 cc".
- Owner strings ("First Owner", "Second Owner", etc.) are mapped to integers (0, 1, 2, ...).
- Rows with missing engine_cc are excluded (affects CAR DEKHO and car data datasets).
- Fuel types are normalized to: Petrol, Diesel, CNG, Electric.
