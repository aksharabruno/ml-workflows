from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.evaluate import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)



tracking_uri = (PROJECT_ROOT / "mlruns").resolve().as_uri()
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)


def _build_pipeline(model_name: str) -> Pipeline:
    model = _build_model(model_name)
    return Pipeline([("scaler", StandardScaler()), ("model", model)])

def _save_confusion_matrix(model_name: str, y_true: pd.Series, y_pred: pd.Series) -> Path:
    plot_path = MODELS_DIR / f"{model_name}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = [0, 1]
    cm = pd.crosstab(y_true, y_pred, rownames=["Actual"], colnames=["Predicted"])
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    ax.matshow(cm.to_numpy(), cmap="Blues")
    ax.set_title(f"{model_name} confusion matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.iloc[i, j]), ha="center", va="center", color="black")
    plt.close(fig)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    return plot_path
