from dependency import *  # noqa: F401,F403


def model_generation_15():
    mlflow.log_artifact(str(out_dir / "training_meta.json"))

