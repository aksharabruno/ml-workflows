# ================================
# Task 4 - Logistic Regression
# Breast Cancer Classification
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

# --------------------------------
# Load Dataset
# --------------------------------

df = pd.read_csv("data.csv")
print("\nFirst 5 Rows\n")
print(df.head())

print("\nDataset Shape:", df.shape)

# --------------------------------
# Drop ID Column
# --------------------------------

if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
    # Drop empty column if it exists
if 'Unnamed: 32' in df.columns:
    df.drop('Unnamed: 32', axis=1, inplace=True)

# --------------------------------
# Convert Target Column
# M = 1
# B = 0
# --------------------------------

df['diagnosis'] = df['diagnosis'].map({
    'M': 1,
    'B': 0
})
# Remove rows with missing values (if any)
df.dropna(inplace=True)

# --------------------------------
# Features and Target
# --------------------------------

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# --------------------------------
# Train Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------
# Feature Scaling
# --------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------
# Train Logistic Regression
# --------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# --------------------------------
# Prediction
# --------------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

# --------------------------------
# Accuracy
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy")

print(round(accuracy*100,2), "%")

# --------------------------------
# Precision
# --------------------------------

precision = precision_score(y_test, y_pred)

print("\nPrecision")

print(round(precision,4))

# --------------------------------
# Recall
# --------------------------------

recall = recall_score(y_test, y_pred)

print("\nRecall")

print(round(recall,4))

# --------------------------------
# ROC AUC
# --------------------------------

roc_auc = roc_auc_score(y_test, y_prob)

print("\nROC AUC Score")

print(round(roc_auc,4))

# --------------------------------
# Classification Report
# --------------------------------

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# --------------------------------
# Confusion Matrix
# --------------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign","Malignant"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()

# --------------------------------
# ROC Curve
# --------------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7,5))

plt.plot(fpr,tpr,label="ROC Curve")

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# --------------------------------
# Threshold Example
# --------------------------------

threshold = 0.60

custom_prediction = (y_prob >= threshold).astype(int)

print("\nPrediction using Threshold =", threshold)

print(custom_prediction[:20])

# --------------------------------
# Sigmoid Function
# --------------------------------

x = np.linspace(-10,10,200)

sigmoid = 1/(1+np.exp(-x))

plt.figure(figsize=(7,5))

plt.plot(x,sigmoid)

plt.title("Sigmoid Function")

plt.xlabel("x")

plt.ylabel("Sigmoid(x)")

plt.grid(True)

plt.show()