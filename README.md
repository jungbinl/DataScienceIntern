# California Property Close Price Prediction (IDX Exchange)

This repository contains a machine learning project dedicated to predicting the final sales price (`ClosePrice`) of single-family residential properties in California. Leveraging historical data sourced from the California Regional Multiple Listing Service (CRMLS), the goal is to develop a robust regression model capable of estimating a property's market value based on its structural and geographical characteristics at the time of query.

This project is highly self-directed and serves as part of the IDX Exchange Data Science Summer 2026 Intern.

## Dataset Description
The dataset consists of historical real estate transactions from CRMLS, pre-filtered based on strict project constraints:
* **PropertyType:** `"Residential"`
* **PropertySubType:** `"SingleFamilyResidence"`
* **Data Duration:** Minimum of 6 months of historical data.

## 🛠️ Pipeline & Task Description

### 1. Data Exploration (EDA)
* Analyzed the overall dataset structure, data types, and distributions of key variables.
* Identified core features influencing `ClosePrice`, such as living area (square footage), number of bedrooms/bathrooms, lot size, and location characteristics.

### 2. Data Preprocessing & Engineering
* **Filtering:** Narrowed down the dataset strictly to `Residential` and `SingleFamilyResidence`.
* **Missing Values:** Handled nulls via strategic imputation (e.g., median for skewed numerical values, mode for categorical variables).
* **Categorical Encoding:** Converted categorical variables using techniques like One-Hot Encoding or Target Encoding.
* **Feature Scaling:** Applied scaling (StandardScaler / MinMaxScaler) to numerical inputs where appropriate for specific models.
* **Data Split:** Partitioned the preprocessed data into training and testing sets to guarantee unbiased model evaluation.

### 3. Model Selection & Training
We explored and trained multiple regression architectures to compare baseline performance against advanced algorithms:
* Linear Regression (Baseline)
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Models (e.g., XGBoost, LightGBM, or CatBoost)
* *Hyperparameter tuning was conducted to optimize model performance and prevent overfitting.*

### 4. Model Evaluation
Models were rigorously evaluated on the testing dataset using key regression metrics:
* **R-squared ($R^2$)**
* **Root Mean Squared Error (RMSE)**
* **Mean Absolute Error (MAE)**
