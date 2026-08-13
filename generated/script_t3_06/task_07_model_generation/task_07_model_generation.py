from dependency import *  # noqa: F401,F403


def model_generation_7(model_name, pipeline):
    trusted_types = []
    if model_name == "xgboost":
        trusted_types = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
    if model_name == "catboost":
        trusted_types = ["catboost.core.CatBoostClassifier"]
    mlflow.sklearn.log_model(pipeline, name=model_name, skops_trusted_types=trusted_types)

