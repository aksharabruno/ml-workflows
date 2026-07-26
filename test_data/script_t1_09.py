
# Wine Quality Prediction - Logistic Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# 1. Load Datasets
red_wine = pd.read_csv("C:\\Users\\donsh\\Downloads\\wine+quality\\winequality-red.csv", sep=";")
white_wine = pd.read_csv("C:\\Users\\donsh\\Downloads\\wine+quality\\winequality-white.csv", sep=";")

# Add wine type feature
red_wine["wine_type"] = 0   # Red
white_wine["wine_type"] = 1 # White

# Combine datasets
df = pd.concat([red_wine, white_wine], axis=0)

print("Dataset Shape:", df.shape)
print(df.head())

# 2. Define Features & Target
X = df.drop("quality", axis=1)
y = df["quality"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Logistic Regression Model
log_model = LogisticRegression(
    max_iter=5000,
    solver="lbfgs"
)

log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)

# 6. Model Evaluation
print("\n📌 Logistic Regression Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Wine Quality")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 8. Multiclass ROC-AUC (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=sorted(y.unique()))
y_prob = log_model.predict_proba(X_test_scaled)

roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr")

print("ROC-AUC Score (OvR):", roc_auc)