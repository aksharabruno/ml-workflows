from dependency import *  # noqa: F401,F403


def model_generation_5(model_name, pipeline):
    mlflow.log_params({
        "model": model_name,
        "dataset_version": dataset_version,
        "random_state": 42,
        "train_test_split": "60/20/20",
        "cv": 3,
        "seed": 42,
    })
    if model_name == "random_forest":
        mlflow.log_params({"best_n_estimators": getattr(pipeline.named_steps["model"], "n_estimators", None)})
        mlflow.log_params({"best_max_depth": getattr(pipeline.named_steps["model"], "max_depth", None)})
