# 🏠 End-to-End House Price Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?style=flat&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red?style=flat)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

A full machine learning pipeline for predicting residential home sale prices using the **Ames Housing Dataset**. The project covers everything from raw data cleaning to model deployment-ready predictions, with a focus on clean code, reproducibility, and domain-aware decision making.

---

## 📌 Problem Statement

Predict the final sale price of homes in Ames, Iowa based on 79 explanatory features describing almost every aspect of residential homes — from lot size and neighborhood to garage quality and year of renovation.

**Type:** Supervised Regression  
**Evaluation Metric:** RMSE on log-transformed SalePrice  
**Dataset:** [Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

---

## 🔍 Project Highlights

- **Domain-aware missing value handling** — distinguishing between true missing data and structural absences (e.g., no garage, no pool)
- **Neighborhood-based LotFrontage imputation** — smarter than global median
- **Custom `FeatureEngineer` transformer** — sklearn-compatible, prevents data leakage
- **6 engineered features** — `TotalSF`, `TotalBath`, `HouseAge`, `IsRemodeled`, `YearsSinceRemod`, `HasMisc`
- **3-strategy encoding pipeline** — Ordinal, Target, and One-Hot encoding applied appropriately per column type
- **Two model comparison** — XGBoost vs LightGBM with `RandomizedSearchCV` + 5-fold CV
- **Full Pipeline architecture** — from raw input to final prediction in one `pipeline.predict()` call

---

## 🗂️ Repository Structure
...
house-price-prediction/
│
├── notebook/
│   └── End-to-End_House_Price_Prediction.ipynb
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── data_description.txt
│
├── outputs/
│   └── submission.csv
│
├── docs/
│   └── House_Price_Prediction_Insights.md
│
├── requirements.txt
├── .gitignore
└── README.md
...

> **Note:** The `data/` folder is excluded from git by default. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place it in a `data/` directory.

---

## ⚙️ Pipeline Architecture

```
Raw Data
   │
   ▼
FeatureEngineer (Custom Transformer)
   ├── Missing value imputation (domain-aware)
   ├── Type corrections (MSSubClass → str)
   └── New feature creation (TotalSF, HouseAge, ...)
   │
   ▼
ColumnTransformer
   ├── OrdinalEncoder    → 19 quality/condition columns
   ├── TargetEncoder     → 15 high-cardinality nominals
   └── OneHotEncoder     →  9 low-cardinality nominals
   │
   ▼
Model (XGBoost / LightGBM)
   │
   ▼
log1p target → expm1 predictions → submission.csv
```

---

## 📊 Results

| Model | CV RMSE (log scale) | MAE (log scale) | R² |
|-------|--------------------|-----------------|----|
| XGBoost | ~0.0.117 | ~0.080 | ~0.0.911 |
| LightGBM | ~0.122 | ~0.0.083 | ~0.903 |

> Metrics are on **log-transformed** SalePrice. The best model was selected based on 5-fold cross-validated RMSE and retrained on the full training set for final submission.

---

## 🧪 Key Steps

| Step | Description |
|------|-------------|
| **1. EDA** | Distribution analysis, correlation heatmap, neighborhood/quality breakdowns |
| **2. Missing Values** | 20+ columns handled with domain knowledge — not blind imputation |
| **3. Outlier Removal** | 2 anomalous large-but-cheap homes removed (known Ames dataset issue) |
| **4. Feature Engineering** | 6 new composite features created inside a leakage-safe transformer |
| **5. Encoding** | Three strategies applied based on feature cardinality and ordering |
| **6. Modeling** | XGBoost & LightGBM tuned with RandomizedSearchCV + 5-fold CV |
| **7. Evaluation** | RMSE, MAE, R², True vs Predicted plot, Residual analysis |
| **8. Submission** | Final model retrained on full data, predictions reverse-transformed |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/AdhamEssam7/house-price-prediction.git
cd house-price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the data
Download `train.csv` and `test.csv` from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place them in a `data/` folder.

### 4. Run the notebook
```bash
jupyter notebook End-to-End_House_Price_Prediction.ipynb
```

---

## 📦 Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Main libraries:

- `pandas`, `numpy` — data manipulation
- `scikit-learn` — preprocessing, pipelines, cross-validation
- `xgboost`, `lightgbm` — gradient boosting models
- `matplotlib`, `seaborn`, `plotly` — visualization

---

## 💡 What I Learned

- Missing values in structured datasets often carry **semantic meaning** — blindly filling them loses information
- Embedding imputation inside a **sklearn Pipeline** is essential to prevent data leakage during cross-validation
- **Target encoding** outperforms one-hot encoding for high-cardinality features like `Neighborhood`
- `log1p` transforming the target before training and `expm1` reversing it after prediction is a simple but high-impact technique
- Feature engineering (especially `TotalSF`) contributed more to model performance than hyperparameter tuning

---

## 📁 Dataset

This project uses the **Ames Housing Dataset** from the Kaggle competition:  
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

The dataset is not included in this repository. Please download it directly from Kaggle.

---

## 👤 Author

**Eng : Adham Essam**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/adham-essam-529492293/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/adhamessam7)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/AdhamEssam7/)

---

*If you found this project useful, please consider giving it a ⭐*
