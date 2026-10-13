<<<<<<< HEAD
# Vehicle Management and Price Prediction System

## 👤 Student Details

- **Name:** Sahishnakumaar S S
- **Roll Number:** 24ADR134

---

## 📋 Introduction

A full-stack Python application that combines vehicle record management with machine learning–powered price prediction and category classification. The system provides:

- Complete CRUD operations for vehicle records via a REST API
- Used-vehicle resale price estimation using a trained RandomForestRegressor
- Vehicle category classification (Budget/Mid-range/Premium) using a trained RandomForestClassifier
- An interactive Streamlit web interface
- A FastAPI backend with automated Swagger documentation
- Docker-based deployment with Docker Compose

---

## 🎯 Objectives

1. Implement vehicle record management with persistent JSON storage
2. Train and deploy a regression model for resale price prediction
3. Train and deploy a classification model for vehicle categorization
4. Build a user-friendly web interface with Streamlit
5. Expose all functionality through a RESTful API
6. Containerize the application with Docker

---

## ✨ Features & Modules

| Module | Description |
|---|---|
| **Vehicle Management** | Full CRUD (Create, Read, Update, Patch, Delete) with Pydantic validation |
| **Price Prediction** | RandomForestRegressor trained on vehicle features to estimate resale price |
| **Category Classification** | RandomForestClassifier to classify vehicles into Budget/Mid-range/Premium |
| **FastAPI Backend** | REST API with Swagger UI, structured error handling, and logging |
| **Streamlit Frontend** | Interactive dashboard, forms, predictions, and about page |
| **Docker Deployment** | Multi-service Docker Compose configuration with persistent storage |

---

## 🛠️ Technology Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.11+ |
| Backend API | FastAPI | 0.115.x |
| ASGI Server | Uvicorn | 0.30.x |
| Frontend | Streamlit | 1.38.x |
| Data Validation | Pydantic | 2.9.x |
| Data Processing | Pandas | 2.2.x |
| Numerical Computing | NumPy | 1.26.x |
| Machine Learning | Scikit-learn | 1.5.x |
| Model Serialization | Joblib | 1.4.x |
| HTTP Client | Requests | 2.32.x |
| Testing | Pytest | 8.x |
| Containerization | Docker & Docker Compose | Latest |

---

## 🏗️ Architecture

```
User → Streamlit Frontend (Port 8501) → FastAPI Backend (Port 8000) → Storage / ML Models → Response
```

- **Streamlit** communicates with **FastAPI** via HTTP requests using the `requests` library
- **FastAPI** handles all business logic, storage operations, and ML inference
- Vehicle records are persisted in a local **JSON file** with thread-safe operations
- ML models are trained offline and loaded at backend startup from **Joblib artifacts**
- The frontend **never** directly accesses storage or model files

---

## 📁 Project Structure

```
vehicle_management_prediction/
├── fastapi_backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI application entry point
│   │   ├── config.py                  # Configuration and paths
│   │   ├── schemas.py                 # Pydantic request/response schemas
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py              # All API endpoint definitions
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── vehicle_service.py     # Vehicle CRUD business logic
│   │   │   └── prediction_service.py  # ML model loading & inference
│   │   └── storage/
│   │       ├── __init__.py
│   │       └── vehicle_repository.py  # JSON file storage with thread safety
│   └── requirements.txt
├── streamlit_frontend/
│   ├── app.py                         # Streamlit application (all sections)
│   └── requirements.txt
├── ml_training/
│   ├── __init__.py
│   ├── load_real_data.py              # Real CSV dataset loader & normalizer
│   ├── generate_dataset.py            # Synthetic dataset generator (legacy)
│   ├── train_price_model.py           # Price regression training (real data)
│   ├── train_category_model.py        # Category classification training (real data)
│   └── evaluate_models.py             # Evaluation report generator
├── dataset/
│   ├── README.md                      # Dataset documentation & policy
│   ├── Car details v3.csv             # Real dataset (8128 records)
│   ├── CAR DETAILS FROM CAR DEKHO.csv # Real dataset (4340 records)
│   ├── car details v4.csv             # Real dataset (2059 records)
│   ├── car data.csv                   # Real dataset (301 records)
│   └── vehicles.csv                   # Synthetic — BLOCKED from training
├── artifacts/
│   ├── vehicle_price_random_forest.pkl      # Trained price model
│   ├── vehicle_category_classifier.pkl      # Trained category model
│   ├── price_model_metadata.json            # Price model training metadata
│   ├── category_model_metadata.json         # Category model training metadata
│   └── evaluation_report.md                 # Model evaluation report
├── data/
│   └── vehicles.json                  # Vehicle records storage (runtime)
├── tests/
│   ├── test_vehicles_api.py           # Vehicle CRUD API tests
│   ├── test_predictions_api.py        # Prediction endpoint tests
│   └── test_ml_training.py            # ML pipeline & data policy tests
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── .env.example
└── README.md
```

---

## 📊 Dataset

### Source & Provenance

This project uses **real vehicle market datasets** from Indian used-vehicle marketplaces. Four CSV files provide authentic vehicle listings with actual selling prices.

### Data Policy

> **IMPORTANT:** Only the four real CSV datasets are permitted for ML training. The synthetic `vehicles.csv` is explicitly blocked by `ml_training/load_real_data.py` and will raise an error if used. Automated tests enforce this restriction.

### Approved Real Datasets

| File | Records | Key Columns |
|------|---------|-------------|
| `Car details v3.csv` | 8,128 | name, year, selling_price, km_driven, fuel, transmission, engine, owner |
| `CAR DETAILS FROM CAR DEKHO.csv` | 4,340 | name, year, selling_price, km_driven, fuel, transmission, owner |
| `car details v4.csv` | 2,059 | Make, Model, Price, Year, Kilometer, Fuel Type, Transmission, Engine |
| `car data.csv` | 301 | Car_Name, Year, Selling_Price (Lakhs), Kms_Driven, Fuel_Type |

### Normalized Schema (After Loading)

| Column | Type | Description |
|---|---|---|
| make | string | Vehicle manufacturer (e.g., Maruti Suzuki, Hyundai) |
| model | string | Vehicle model (e.g., Swift, Creta) |
| year | integer | Manufacturing year |
| km_driven | integer | Kilometres driven |
| fuel_type | string | Petrol / Diesel / CNG / Electric |
| transmission | string | Manual / Automatic |
| engine_cc | integer | Engine displacement in CC |
| previous_owners | integer | Number of previous owners (0–4) |
| selling_price | float | Resale selling price in INR (target) |

### Dataset Loading

```bash
cd vehicle_management_prediction
python -m ml_training.load_real_data
```

This loads, normalizes, and validates all real CSV datasets (~8,692 usable records).

---

## 🤖 Model Training

### Price Prediction Model

```bash
python ml_training/train_price_model.py
```

- **Algorithm:** RandomForestRegressor (200 trees, max_depth=20)
- **Features:** make, model, year, km_driven, fuel_type, transmission, engine_cc, previous_owners
- **Target:** selling_price (INR)
- **Output:** `artifacts/vehicle_price_random_forest.pkl`

### Category Classification Model

```bash
python ml_training/train_category_model.py
```

- **Algorithm:** RandomForestClassifier (200 trees, max_depth=15, balanced class weights)
- **Categories:** Budget (<₹3L), Mid-range (₹3L–₹8L), Premium (≥₹8L)
- **Output:** `artifacts/vehicle_category_classifier.pkl`

### Evaluation Report

```bash
python ml_training/evaluate_models.py
```

Generates `artifacts/evaluation_report.md` with metrics from both models.

### Methodology

1. Real CSV datasets are loaded and normalized via `ml_training/load_real_data.py`
2. `vehicles.csv` is automatically blocked (raises `ValueError` if attempted)
3. Dataset is validated for schema compliance, missing values, and invalid records
4. Data is split into 80% training / 20% testing (stratified for classification)
5. Numeric features are scaled with StandardScaler
6. Categorical features are encoded with OneHotEncoder (handle_unknown="ignore")
7. Preprocessing and model are combined in a Scikit-learn Pipeline
8. **No data leakage**: preprocessing is fitted only on training data
9. Random seed 42 ensures reproducible results

> **Note:** Models are trained on real market data from Indian used-vehicle listings.

---

## 🚀 Local Setup & Execution

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
cd vehicle_management_prediction
pip install -r fastapi_backend/requirements.txt
pip install -r streamlit_frontend/requirements.txt
pip install pytest httpx
```

### Step 2: Generate Dataset

```bash
python ml_training/generate_dataset.py
```
.\venv\Scripts\activate

### Step 3: Train Models

```bash
python ml_training/train_price_model.py
python ml_training/train_category_model.py
python ml_training/evaluate_models.py
```

### Step 4: Start Backend

```bash
cd fastapi_backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: **http://localhost:8000**
Swagger UI: **http://localhost:8000/docs**

### Step 5: Start Frontend (New Terminal)

```bash
cd streamlit_frontend
streamlit run app.py --server.port 8501
```

The frontend will be available at: **http://localhost:8501**

---

## 🐳 Docker Setup

### Build and Run

```bash
cd vehicle_management_prediction
docker-compose up --build
```

### Services

| Service | URL | Port |
|---|---|---|
| Backend API | http://localhost:8000 | 8000 |
| Swagger UI | http://localhost:8000/docs | 8000 |
| Frontend | http://localhost:8501 | 8501 |

### Stop Services

```bash
docker-compose down
```

### Persist Data

Vehicle records are stored in a Docker volume (`vehicle_data`) and persist across container restarts.

---

## 📡 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API health/status check |
| GET | `/vehicles` | List all vehicles |
| GET | `/vehicles/{id}` | Get vehicle by ID |
| POST | `/vehicles` | Create a new vehicle |
| PUT | `/vehicles/{id}` | Replace vehicle (full update) |
| PATCH | `/vehicles/{id}` | Partial vehicle update |
| DELETE | `/vehicles/{id}` | Delete a vehicle |
| POST | `/predict/price` | Predict vehicle resale price |
| POST | `/predict/category` | Classify vehicle category |

### Example: Create Vehicle

```bash
curl -X POST http://localhost:8000/vehicles \
  -H "Content-Type: application/json" \
  -d '{
    "make": "Hyundai",
    "model": "Creta",
    "year": 2021,
    "km_driven": 25000,
    "fuel_type": "Diesel",
    "transmission": "Automatic",
    "engine_cc": 1493,
    "previous_owners": 0
  }'
```

**Response (201):**
```json
{
  "id": "VH-A1B2C3D4",
  "make": "Hyundai",
  "model": "Creta",
  "year": 2021,
  "km_driven": 25000,
  "fuel_type": "Diesel",
  "transmission": "Automatic",
  "engine_cc": 1493,
  "previous_owners": 0,
  "created_at": "2024-01-01T12:00:00+00:00",
  "updated_at": "2024-01-01T12:00:00+00:00"
}
```

### Example: Predict Price

```bash
curl -X POST http://localhost:8000/predict/price \
  -H "Content-Type: application/json" \
  -d '{
    "make": "Hyundai",
    "model": "Creta",
    "year": 2019,
    "km_driven": 45000,
    "fuel_type": "Diesel",
    "transmission": "Automatic",
    "engine_cc": 1493,
    "previous_owners": 1
  }'
```

**Response (200):**
```json
{
  "predicted_price": 892500.00,
  "currency": "INR",
  "model_name": "vehicle_price_random_forest",
  "input_summary": { ... },
  "disclaimer": "This is an estimated price..."
}
```

### Example: Classify Category

```bash
curl -X POST http://localhost:8000/predict/category \
  -H "Content-Type: application/json" \
  -d '{
    "make": "Maruti Suzuki",
    "model": "Alto",
    "year": 2012,
    "km_driven": 95000,
    "fuel_type": "Petrol",
    "transmission": "Manual",
    "engine_cc": 796,
    "previous_owners": 2
  }'
```

**Response (200):**
```json
{
  "predicted_category": "Budget",
  "model_name": "vehicle_category_classifier",
  "categories": ["Budget", "Mid-range", "Premium"],
  "category_description": "Vehicles with estimated resale price below ₹3,00,000",
  "input_summary": { ... },
  "disclaimer": "Category is based on price bands..."
}
```

---

## 🧪 Running Tests

```bash
cd vehicle_management_prediction
pytest tests/ -v
```

Tests cover:
- API health and OpenAPI schema
- Vehicle CRUD operations (create, read, update, patch, delete)
- Input validation and error handling
- ML model training and artifact persistence
- Prediction quality (R², accuracy thresholds)
- Integration workflow (complete CRUD cycle)
- Storage edge cases (corrupted files, empty storage)

---

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| "Cannot connect to backend" | Ensure FastAPI is running on port 8000 |
| "Model not available" | Run the training scripts first |
| Import errors | Verify you're in the correct directory |
| Docker build fails | Ensure artifacts/ directory has trained models |
| Port already in use | Change PORT in .env or use `--port` flag |
| Corrupted vehicles.json | Delete the file; it will be recreated |

---

## ⚠️ Limitations & Future Improvements

### Current Limitations

1. **Market Coverage**: Models are trained on ~8,692 real records from Indian market. Predictions for uncommon makes/models may be less accurate.
2. **No Authentication**: The API has no user authentication or authorization.
3. **JSON Storage**: Suitable for small-scale use but lacks multi-process transactional guarantees.
4. **No Real-time Updates**: Models are not retrained on new vehicle records.
5. **Price Currency**: All prices are in INR. No multi-currency support.

### Future Improvements

1. Expand real dataset coverage with more market sources
2. Add user authentication (JWT tokens)
3. Migrate to SQLite or PostgreSQL for better data integrity
4. Implement model retraining pipeline
5. Add more ML models (XGBoost, LightGBM) for comparison
6. Deploy to cloud (AWS, GCP, Azure)
7. Add pagination for vehicle listing
8. Implement caching for prediction responses

---

## 📄 License

This project is developed for academic purposes as part of the coursework by Sahishnakumaar S S (24ADR134).

---

*Vehicle Management and Price Prediction System © 2024 — Sahishnakumaar S S (24ADR134)*
=======
# VEHICLE-MANAGEMENT-AND-PRICE-PREDICTION-SYSTEM
>>>>>>> origin/main
