from dependency import *  # noqa: F401,F403


def model_generation_12(model_path, pipe):
    reg_name = os.environ.get("MLFLOW_MODEL_NAME")
    log_model_kwargs = {"artifact_path": "sklearn-model"}
    if reg_name and not str(tracking).lower().startswith("file:"):
        log_model_kwargs["registered_model_name"] = reg_name
    mlflow_sklearn.log_model(pipe, **log_model_kwargs)

    mlflow.log_artifact(str(model_path))
