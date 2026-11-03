Here is a shorter and simple `README.md` for your project.

# Indian House Price Prediction System

## Project Overview

The Indian House Price Prediction System is a machine learning-based application that predicts house prices in different Indian cities based on property features such as location, area, bedrooms, and bathrooms.

The project uses Python, MongoDB, and machine learning to analyze housing data, predict prices, and store housing records and prediction history.

## Technologies Used

- **Language:** Python
- **Database:** MongoDB
- **Connectivity:** PyMongo
- **Machine Learning:** Scikit-learn
- **Algorithms:** Linear Regression, Decision Tree Regressor, Random Forest Regressor
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Interface:** Streamlit
- **Model Saving:** Joblib

## Dataset

[Indian Urban House Price Prediction – Kaggle](https://www.kaggle.com/competitions/indian-urban-house-price-prediction/data)

The project uses 1,000 valid and unique records from the Kaggle training dataset, subject to availability.

## Key Features

- Store and manage housing records in MongoDB.
- Perform CRUD operations and aggregation.
- Train and compare three regression models.
- Evaluate models using R², MAE, MSE, and RMSE.
- Predict house prices using property details.
- Store prediction history in MongoDB.

## Installation

```
git clone <your-repository-url>
cd Indian-House-Price-Prediction
pip install -r requirements.txt
```

Make sure MongoDB is running, then start the application:

```
streamlit run app.py
```

## Objective

To develop a simple house price prediction system by integrating machine learning algorithms with MongoDB for data storage and management.

Add a project folder structureExplain the prediction workflow briefly
