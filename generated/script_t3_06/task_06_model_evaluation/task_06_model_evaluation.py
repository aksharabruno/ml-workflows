from dependency import *  # noqa: F401,F403


def model_evaluation_6(X_test, model_name, pipeline, test_metrics, val_metrics, y_test):
    mlflow.log_metrics({**{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"}, **{f"test_{k}": v for k, v in test_metrics.items() if k != "confusion_matrix"}})

    plot_path = _save_confusion_matrix(model_name, y_test, pipeline.predict(X_test))
    mlflow.log_artifact(str(plot_path))

