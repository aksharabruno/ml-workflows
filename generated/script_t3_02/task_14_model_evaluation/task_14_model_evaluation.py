from dependency import *  # noqa: F401,F403


def model_evaluation_14():
    mlflow.log_artifact(str(out_dir / "metrics.json"))
