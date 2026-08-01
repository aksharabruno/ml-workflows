#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse


# In[19]:


import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

TARGET = "type"

TEXT_COLUMNS = [
    "title",
    "director",
    "cast",
    "country",
    "description"
]

DROP_COLUMNS = [
    "id",
    "date_added",
    "duration",
    "listed_in",
    "platform",
    "rating"
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Movie/TV Show classification model"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training CSV file"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["logistic_regression", "linear_svm"],
        help="Model to train"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()



args = parse_args()


def load_data(path):
    df = pd.read_csv(path)
    print(df.head())
    print(df.info())
    return df

def perform_eda(df):
    os.makedirs("outputs", exist_ok=True)

    print("\nMissing Values\n")
    print(df.isnull().sum())

    df.isnull().sum().to_csv("outputs/missing_values.csv")

    plt.figure(figsize=(5,4))
    df[TARGET].value_counts().plot(kind="bar")
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig("outputs/target_distribution.png")
    plt.close()

    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] > 1:
        corr = numeric.corr()
        plt.figure(figsize=(6,5))
        plt.imshow(corr)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.tight_layout()
        plt.savefig("outputs/correlation.png")
        plt.close()

def build_models():

    return {
        "logistic_regression": Pipeline([
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000
                )
            )
        ]),

        "linear_svm": Pipeline([
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "classifier",
                LinearSVC()
            )
        ])
    }

def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print(f"\n{name}")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print(classification_report(y_test, pred))

    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, pred))
    disp.plot()
    plt.tight_layout()
    plt.savefig(f"outputs/{name}_cm.png")
    plt.close()

    return acc
