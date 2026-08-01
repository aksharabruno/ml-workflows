"""Train a model to predict TMDB vote_average (regression or classification)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import build_feature_frame

TaskName = Literal["regression", "classification"]
LabelScheme = Literal["exact", "tier3"]

NUMERIC_COLS = [
    "popularity",
    "log_pop",
    "log_vote_count",
    "votes_per_log_pop",
    "air_year",
    "overview_len",
    "name_len",
    "overview_words",
]


def _tracking_uri_default() -> str:
    return (Path.cwd() / "mlruns").resolve().as_uri()


def _use_mlflow_flag(use_mlflow: bool | None) -> bool:
    if use_mlflow is not None:
        return use_mlflow
    return os.environ.get("MLFLOW_DISABLE", "").lower() not in ("1", "true", "yes")



do_mlflow = _use_mlflow_flag(use_mlflow)
if do_mlflow:
    import mlflow
    from mlflow import sklearn as mlflow_sklearn

    tracking = os.environ.get("MLFLOW_TRACKING_URI") or _tracking_uri_default()
    experiment = os.environ.get("MLFLOW_EXPERIMENT", "tmdb-tv-rating")
    mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment(experiment)


p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, default=Path("data/tv_shows.csv"))
p.add_argument("--out", type=Path, default=Path("models"))
p.add_argument("--test-size", type=float, default=0.2)
p.add_argument(
    "--task",
    choices=("regression", "classification"),
    default="classification",
    help="classification reports accuracy; regression reports MAE/R2.",
)
p.add_argument(
    "--label-scheme",
    choices=("exact", "tier3"),
    default="exact",
    help="exact=0-10 stars (hard). tier3=3 quantile buckets on train labels (easier accuracy).",
)
p.add_argument(
    "--no-text-features",
    action="store_true",
    help="Disable TF-IDF+SVD on title+overview (faster, usually lower accuracy).",
)
p.add_argument(
    "--tune",
    action="store_true",
    help="Randomized search for best holdout accuracy (classification only; slower).",
)
p.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
args = p.parse_args()
tune = args.tune and args.task == "classification"

def build_regression_pipeline(*, text_features: bool = True) -> Pipeline:
    reg = HistGradientBoostingRegressor(
        max_depth=12,
        learning_rate=0.06,
        max_iter=600,
        min_samples_leaf=3,
        l2_regularization=0.04,
        random_state=42,
    )
    return Pipeline([("prep", build_preprocessor(text_features=text_features)), ("model", reg)])

def build_classifier_pipeline(*, text_features: bool = True) -> Pipeline:
    clf = HistGradientBoostingClassifier(
        max_depth=14,
        learning_rate=0.07,
        max_iter=800,
        min_samples_leaf=2,
        l2_regularization=0.02,
        random_state=42,
    )
    return Pipeline([("prep", build_preprocessor(text_features=text_features)), ("model", clf)])

def _quantile_labels(y_reference: np.ndarray, y_to_label: np.ndarray, q: int = 3) -> np.ndarray:
    s_ref = pd.Series(y_reference)
    try:
        _, bins = pd.qcut(s_ref, q=q, retbins=True, duplicates="drop")
    except ValueError:
        bins = np.linspace(float(np.min(y_reference)), float(np.max(y_reference)), q + 1)
    bins = np.asarray(bins, dtype=float)
    if len(np.unique(bins)) < 2:
        return np.zeros(len(y_to_label), dtype=np.int64)
    out = pd.cut(pd.Series(y_to_label), bins=bins, include_lowest=True, labels=False)
    return out.fillna(0).astype(np.int64).to_numpy()

def _tune_classifier(base: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    param_distributions = {
        "model__max_depth": [10, 12, 14, 16, 18, 22],
        "model__learning_rate": [0.04, 0.06, 0.08, 0.1, 0.12],
        "model__max_iter": [600, 800, 1000, 1200],
        "model__min_samples_leaf": [1, 2, 3, 4],
        "model__l2_regularization": [0.0, 0.01, 0.03, 0.06, 0.1],
    }
    search = RandomizedSearchCV(
        clone(base),
        param_distributions=param_distributions,
        n_iter=32,
        cv=3,
        random_state=42,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

def _metrics_dict_fixed(
    task: TaskName, pipe: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray, n_train: int
) -> dict[str, Any]:
    if task == "classification":
        pred = pipe.predict(X_test)
        err = np.abs(pred.astype(np.float64) - y_test.astype(np.float64))
        return {
            "task": "classification",
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "pct_within_1_star": float(np.mean(err <= 1)),
            "pct_within_2_stars": float(np.mean(err <= 2)),
            "n_train": n_train,
            "n_test": int(len(y_test)),
        }
    pred = pipe.predict(X_test)
    return {
        "task": "regression",
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "n_train": n_train,
        "n_test": int(len(y_test)),
    }

def _class_to_mean_vote(y_cls: np.ndarray, y_cont: np.ndarray) -> dict[str, float]:
    d = pd.DataFrame({"c": y_cls, "y": y_cont})
    means = d.groupby("c")["y"].mean()
    return {str(int(i)): float(v) for i, v in means.items()}
