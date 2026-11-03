# Indian House Price Prediction System

A functional mini project built using **Python**, **MongoDB (PyMongo)**, **Scikit-learn**, and **Streamlit** to predict the estimated price of Indian residential properties.

---

## 📌 Project Overview
This system predicts residential property prices in Indian cities based on key parameters:
- **City** (e.g., Chennai, Pune, Mumbai, Delhi, Bengaluru)
- **Locality**
- **Property Area** (Square Feet)
- **Number of Bedrooms** (BHK)
- **Number of Bathrooms**
- **Property Type** (Apartment, Independent House, Villa)
- **Furnishing Status** (Furnished, Semi-furnished, Unfurnished)
- **Parking Availability** (Yes, No)

Output is presented in both **Indian Lakhs (₹ L)** and **Indian Rupees (₹)**.

---

## 🗄️ MongoDB Architecture & PyMongo Integration

The system connects directly to MongoDB (`mongodb://localhost:27017/`) using **PyMongo** and manages three primary collections in database `indian_house_db`:

| Collection | Purpose | Key PyMongo Operations |
|---|---|---|
| `house_data` | Stores original 1,000 Kaggle property listings | `insert_many()`, `find()`, `update_one()`, `delete_one()`, `aggregate()`, `count_documents()` |
| `model_metrics` | Stores performance results for trained regression models | `insert_one()`, `find()`, `delete_many()` |
| `predictions` | Stores user predictions with input parameters & timestamp | `insert_one()`, `find()`, `delete_many()` |

### Duplicate Import Prevention
Before loading, the system queries `db.house_data.count_documents({})`. If records exist, duplicate imports are prevented unless the user explicitly triggers a re-sync.

### MongoDB Demonstrations Included:
- **`insert_many()`**: Bulk import of exactly 1,000 Kaggle records.
- **`find()`**: Search, pagination (20 records/page), and filtering by city, price range, and BHK.
- **`update_one()`**: Updates prices or attributes using the `$set` operator.
- **`delete_one()`**: Deletes documents by property ID.
- **Aggregation Pipelines**: Utilizes `$match`, `$group`, `$avg`, and `$sort` to compute average prices by city and property types.

---

## 🤖 Machine Learning Model: Random Forest Regressor

The project uses **Random Forest Regressor** (Ensemble Bagging algorithm with 100 decision trees) from **Scikit-learn**:
- **Why Random Forest?** Real estate pricing has non-linear relationships and high-cardinality interactions (e.g. area, locality, bedrooms, furnishing). Random Forest excels on tabular real estate data by combining predictions from multiple de-correlated decision trees, minimizing variance without overfitting.

### Preprocessing & Training Pipeline
- **Dataset Split:** 80% Training (800 records) and 20% Testing (200 records).
- **Categorical Features:** One-Hot Encoded via `ColumnTransformer` with `handle_unknown='ignore'` to handle unseen cities or localities safely.
- **Numerical Features:** Scaled using `StandardScaler`.
- **Hyperparameters:** `n_estimators=100`, `max_depth=14`, `min_samples_split=4`.
- **Model Persistence:** The trained model pipeline is serialized using **Joblib** to `saved_models/random_forest_model.joblib`.

### Regression Evaluation Metrics
- **$R^2$ Score (Coefficient of Determination):** Proportion of variance in property prices explained by features.
- **MAE (Mean Absolute Error):** Average magnitude of prediction errors in Lakhs.
- **MSE (Mean Squared Error):** Average squared difference, penalizing larger deviations.
- **RMSE (Root Mean Squared Error):** Square root of MSE, returned to Lakhs scale.

---

## 🎓 College Viva Concept: Regression vs Classification Metrics

> **Viva Question:** Can we use Accuracy, Precision, Recall, or F1-Score for House Price Prediction?
>
> **Answer:** **No.** 
> - **House price is a continuous numerical value**, not a discrete categorical label.
> - **Accuracy** checks for exact matches ($y_{pred} == y_{actual}$). If actual price is ₹ 50.00 Lakhs and model predicts ₹ 50.01 Lakhs, categorical accuracy would consider it 0% (wrong), despite being 99.98% accurate in reality.
> - **Precision, Recall, and F1-Score** require a Confusion Matrix (True Positives, False Positives, False Negatives), which only exists for discrete classes (e.g. Fraud / Not Fraud).
> - Therefore, continuous problems must be evaluated using **$R^2$, MAE, MSE, and RMSE**.

---

## 🚀 Setup and Execution Instructions

### 1. Prerequisites
- Python 3.10+ installed
- MongoDB Community Server running locally on port `27017`

### 2. Install Dependencies
Open terminal / command prompt in this directory and run:
```bash
pip install -r requirements.txt
```

### 3. Verify MongoDB Service is Running
Ensure MongoDB service is running:
- **Windows:** Run `services.msc` and check `MongoDB Server (MongoDB)` is **Running**, or run:
  ```powershell
  net start MongoDB
  ```

### 4. Run the Streamlit Application
Launch the interactive web application using Python's module launcher:
```bash
python -m streamlit run app.py
```
*(Or `streamlit run app.py` if Python's Scripts folder is in your system PATH).*
Open your browser at `http://localhost:8501`.

---

## 📁 Project Directory Structure
```
HousePrediction/
├── app.py                         # Streamlit multi-section application
├── load_dataset.py                # Standalone script for loading, cleaning & MongoDB ingestion
├── db.py                          # PyMongo connection & CRUD/aggregation operations
├── model.py                       # ML preprocessing, training, evaluation, & Joblib saving
├── indian_urban_house_prices.csv  # Kaggle Indian Urban House Price dataset
├── requirements.txt               # Required Python packages
├── README.md                      # Documentation and viva preparation
└── saved_models/                  # Joblib persisted models
    ├── best_model.joblib
    ├── linear_regression.joblib
    ├── decision_tree_regressor.joblib
    └── random_forest_regressor.joblib
```

