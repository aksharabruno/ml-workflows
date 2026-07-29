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

def preprocess(df, seed):

    # Remove rows where target is missing
    df = df.dropna(subset=["type"]).copy()

    # Features we want to use
    feature_columns = [
        "title",
        "description",
        "director",
        "country",
        "release_year"
    ]

    X = df[feature_columns].copy()

    # Fill missing text/categorical values
    text_columns = [
        "title",
        "description",
        "director",
        "country"
    ]

    for col in text_columns:
        X[col] = X[col].fillna("Unknown")

    # Handle missing numerical value
    X["release_year"] = X["release_year"].fillna(
        X["release_year"].median()
    )

    # Target
    y = df["type"]

    # Encode Movie / TV Show -> 0 / 1
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split BEFORE fitting TF-IDF / OneHotEncoder / Scaler
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, label_encoder

def build_preprocessor():

    preprocessor = ColumnTransformer(
        transformers=[

            # Text features
            (
                "description_tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=5000,
                    ngram_range=(1, 2)
                ),
                "description"
            ),

            (
                "title_tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=2000
                ),
                "title"
            ),

            # Categorical features
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                [
                    "director",
                    "country"
                ]
            ),

            # Numerical feature
            (
                "numerical",
                StandardScaler(),
                ["release_year"]
            )
        ]
    )

    return preprocessor


def main():

    
    # Parse command-line arguments
    

    args = parse_args()

    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)

    print("Data path :", args.data_path)
    print("Model     :", args.model)
    print("Seed      :", args.seed)


    
    # Load data
    

    df = load_data(
        args.data_path
    )


    
    # EDA
    

    perform_eda(df)


    
    # Preprocessing + split
    

    X_train, X_test, y_train, y_test, label_encoder = preprocess(
        df,
        args.seed
    )

    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples :", X_test.shape[0])


    
    # Build models
    

    models = build_models()


    
    # Select requested model
    

    model = models[args.model]


    
    # Train
    

    print(f"\nTraining: {args.model}")

    model.fit(
        X_train,
        y_train
    )


    
    # Evaluate
    

    score = evaluate(
        args.model,
        model,
        X_test,
        y_test
    )


    
    # Save model
    

    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = f"models/{args.model}.pkl"

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        label_encoder,
        "models/label_encoder.pkl"
    )

    print("\nTraining completed.")
    print("Model saved to:", model_path)
    print("Accuracy:", score)
if __name__ == "__main__":
    main()


# In[ ]:



