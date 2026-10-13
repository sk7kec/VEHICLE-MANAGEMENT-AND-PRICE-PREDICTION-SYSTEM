# Model Evaluation Report

**Project:** Vehicle Management and Price Prediction System
**Author:** Sahishnakumaar S S (24ADR134)

## 1. Price Prediction Model (Regression)

- **Model Type:** RandomForestRegressor
- **Training Date:** 2026-10-13T10:52:53.415595
- **Dataset:** Car details v3.csv, car details v4.csv (real)
- **Total Records:** 8692
- **Train/Test Split:** 6953 / 1739
- **Random Seed:** 42

### Evaluation Metrics

| Metric | Value |
|---|---|
| MAE | ₹147,491.38 |
| RMSE | ₹447,658.98 |
| R² Score | 0.8802 |

### Hyperparameters

- **n_estimators:** 200
- **max_depth:** 20
- **min_samples_split:** 5
- **min_samples_leaf:** 2

> **Disclaimer:** 

## 2. Category Classification Model

- **Model Type:** RandomForestClassifier
- **Training Date:** 2026-10-13T10:53:16.208305
- **Dataset:** Car details v3.csv, car details v4.csv (real)
- **Categories:** Budget, Mid-range, Premium
- **Total Records:** 8692
- **Train/Test Split:** 6953 / 1739

### Category Definitions

- **Budget:** selling_price < INR 3,00,000
- **Mid-range:** INR 3,00,000 <= selling_price < INR 8,00,000
- **Premium:** selling_price >= INR 8,00,000

*Categories derived from selling price bands in real datasets. These represent price segments, not vehicle quality or condition.*

### Overall Metrics

| Metric | Value |
|---|---|
| Accuracy | 0.7637 |
| Precision (weighted) | 0.7775 |
| Recall (weighted) | 0.7637 |
| F1-Score (weighted) | 0.7618 |

### Per-Class Metrics

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Budget | 0.7163 | 0.8938 | 0.7953 | 452 |
| Mid-range | 0.8423 | 0.6685 | 0.7454 | 887 |
| Premium | 0.7028 | 0.8275 | 0.76 | 400 |

> **Disclaimer:** 
