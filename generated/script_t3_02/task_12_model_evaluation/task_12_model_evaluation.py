from dependency import *  # noqa: F401,F403


def model_evaluation_12(metrics):
    float_metrics = {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float, np.integer, np.floating))
    }
    mlflow.log_metrics(float_metrics)
