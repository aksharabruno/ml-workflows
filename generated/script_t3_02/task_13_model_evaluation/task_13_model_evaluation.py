from dependency import *  # noqa: F401,F403


def model_evaluation_13():
    mlflow.log_artifact(str(out_dir / "metrics.json"))
