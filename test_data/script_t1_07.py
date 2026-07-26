# logistic_regression.py
"""
Logistic Regression Module
Demonstrates train/test split, binary classification model training, accuracy evaluation, and classification reporting.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "logistic_data.csv")

df = pd.read_csv(DATASET_PATH)

# Features and Target
X = df[['Hours_Studied', 'Attendance']]
Y = df['Pass']

# Train / Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.19, random_state=42
)

# Model Training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Predictions & Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)

print("--- Test Set Predictions ---")
print(f"Predictions: {predictions}")
print(f"Accuracy Score: {accuracy * 100:.2f}%\n")

print("--- Classification Report ---")
print(classification_report(Y_test, predictions))