"""
============================================================
HOUSE PRICE PREDICTION USING LINEAR REGRESSION
============================================================

Dataset:
    California Housing Dataset

Goal:
    Predict house prices using Linear Regression.

Steps:
    1. Load dataset
    2. Explore dataset
    3. Visualize data
    4. Split data
    5. Train model
    6. Predict values
    7. Evaluate model
    8. Visualize results
    9. Predict a new sample

Required Packages:
    pip install pandas matplotlib seaborn scikit-learn
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("\nLoading California Housing Dataset...\n")

housing = fetch_california_housing(as_frame=True)

# Complete dataframe
df = housing.frame

# Display basic information
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Rows and Columns:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

# ==========================================================
# DATA EXPLORATION
# ==========================================================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================================
# VISUALIZATION 1
# FEATURE CORRELATION HEATMAP
# ==========================================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# ==========================================================
# VISUALIZATION 2
# MEDIAN INCOME VS HOUSE VALUE
# ==========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["MedInc"],
    df["MedHouseVal"],
    alpha=0.3
)

plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Median Income vs House Value")

plt.grid(True)

plt.show()

# ==========================================================
# PREPARE FEATURES AND TARGET
# ==========================================================

# Input features
X = df.drop("MedHouseVal", axis=1)

# Target variable
y = df["MedHouseVal"]

print("\nFeature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================================================
# SPLIT DATA
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================================
# CREATE MODEL
# ==========================================================

model = LinearRegression()

# ==========================================================
# TRAIN MODEL
# ==========================================================

print("\nTraining Linear Regression Model...")

model.fit(X_train, y_train)

print("Training Completed!")

# ==========================================================
# MODEL COEFFICIENTS
# ==========================================================

print("\n" + "=" * 60)
print("MODEL COEFFICIENTS")
print("=" * 60)

print("Intercept:")
print(model.intercept_)

print("\nFeature Contributions:")

for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature:15} -> {coefficient:.6f}")

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred = model.predict(X_test)

# ==========================================================
# MODEL EVALUATION
# ==========================================================

mse = mean_squared_error(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Mean Absolute Error : {mae:.4f}")
print(f"Mean Squared Error  : {mse:.4f}")
print(f"R² Score            : {r2:.4f}")

# ==========================================================
# VISUALIZATION 3
# ACTUAL VS PREDICTED VALUES
# ==========================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

# Perfect prediction line
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    linewidth=2
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted")

plt.grid(True)

plt.show()

# ==========================================================
# VISUALIZATION 4
# RESIDUAL ERRORS
# ==========================================================

# Residual = Actual - Predicted
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.5
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted Values")
plt.ylabel("Residual Errors")
plt.title("Residual Plot")

plt.grid(True)

plt.show()

# ==========================================================
# PREDICT NEW HOUSE
# ==========================================================

print("\n" + "=" * 60)
print("NEW HOUSE PREDICTION")
print("=" * 60)

# Example house
new_house = pd.DataFrame({
    "MedInc": [8.5],
    "HouseAge": [20],
    "AveRooms": [6.0],
    "AveBedrms": [1.1],
    "Population": [1200],
    "AveOccup": [3.0],
    "Latitude": [34.2],
    "Longitude": [-118.4]
})

predicted_value = model.predict(new_house)

print(
    f"\nPredicted House Value: "
    f"${predicted_value[0] * 100000:,.2f}"
)

print("\nProgram Completed Successfully.")