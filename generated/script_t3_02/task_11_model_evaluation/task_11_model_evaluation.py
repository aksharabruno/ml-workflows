from dependency import *  # noqa: F401,F403


def model_evaluation_11(df, metrics):
    with mlflow.start_run():
        mlflow.log_params(
            {
                "test_size": test_size,
                "data_path": str(data_path.resolve()),
                "n_rows_raw": int(len(df)),
                "task": task,
                "tune": tune,
                "label_scheme": label_scheme,
                "text_features": text_features,
            }
        )
        float_metrics = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float, np.integer, np.floating))
        }
        mlflow.log_metrics(float_metrics)
        mlflow.log_param("task", task)
        mlflow.log_param("label_scheme", label_scheme)
        mlflow.log_param("text_features", str(text_features))

