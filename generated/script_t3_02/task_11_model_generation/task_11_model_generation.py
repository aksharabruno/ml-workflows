from dependency import *  # noqa: F401,F403


def model_generation_11(df):
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
