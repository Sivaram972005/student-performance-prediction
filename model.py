# ============================================
# Student Performance Prediction
# Random Forest Regression
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================
# 1. Load Dataset
# ============================================

file_path = "../data/student_performance.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# ============================================
# 2. Data Cleaning
# ============================================

# Remove duplicate records
df = df.drop_duplicates()

# Fill missing numerical values using median
numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================
# 3. Exploratory Data Analysis
# ============================================

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())


# --------------------------------------------
# Study Time vs Final Score
# --------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["StudyTime"],
    df["FinalScore"]
)

plt.xlabel("Study Time")
plt.ylabel("Final Score")
plt.title("Study Time vs Final Score")

plt.grid(True)
plt.show()


# --------------------------------------------
# Attendance vs Final Score
# --------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Attendance"],
    df["FinalScore"]
)

plt.xlabel("Attendance")
plt.ylabel("Final Score")
plt.title("Attendance vs Final Score")

plt.grid(True)
plt.show()


# --------------------------------------------
# Previous Score vs Final Score
# --------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["PreviousScore"],
    df["FinalScore"]
)

plt.xlabel("Previous Score")
plt.ylabel("Final Score")
plt.title("Previous Score vs Final Score")

plt.grid(True)
plt.show()


# ============================================
# 4. Correlation Analysis
# ============================================

correlation = df[
    [
        "StudyTime",
        "Attendance",
        "PreviousScore",
        "FinalScore"
    ]
].corr()

print("\n========== CORRELATION ==========")
print(correlation)


# ============================================
# 5. Feature Selection
# ============================================

X = df[
    [
        "StudyTime",
        "Attendance",
        "PreviousScore"
    ]
]

y = df["FinalScore"]


# ============================================
# 6. Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n========== DATA SPLIT ==========")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# 7. Random Forest Model
# ============================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ============================================
# 8. Train Model
# ============================================

model.fit(
    X_train,
    y_train
)

print("\nRandom Forest model trained successfully.")


# ============================================
# 9. Prediction
# ============================================

y_pred = model.predict(X_test)


print("\n========== PREDICTIONS ==========")

results = pd.DataFrame({
    "Actual Score": y_test.values,
    "Predicted Score": np.round(y_pred, 2)
})

print(results)


# ============================================
# 10. Model Evaluation
# ============================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n========== MODEL EVALUATION ==========")

print("MAE  :", round(mae, 2))
print("MSE  :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("R2   :", round(r2, 2))


# ============================================
# 11. Feature Importance
# ============================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)


# ============================================
# Feature Importance Visualization
# ============================================

plt.figure(figsize=(8, 5))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# ============================================
# 12. Actual vs Predicted
# ============================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Actual vs Predicted Student Performance")

plt.grid(True)
plt.show()


# ============================================
# 13. Predict New Student
# ============================================

print("\n========== NEW STUDENT PREDICTION ==========")

new_student = pd.DataFrame({
    "StudyTime": [6],
    "Attendance": [90],
    "PreviousScore": [82]
})

predicted_score = model.predict(
    new_student
)

print(
    "Predicted Final Score:",
    round(predicted_score[0], 2)
)
