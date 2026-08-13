from dependency import *  # noqa: F401,F403


def model_evaluation_8(model_name, pipeline, results, test_metrics, val_metrics):
    results[model_name] = {"val": val_metrics, "test": test_metrics}
    if test_metrics["f1"] > best_score:
        best_score = test_metrics["f1"]
        best_model_name = model_name
        best_pipeline = pipeline
        best_metrics = test_metrics

    return best_model_name, best_pipeline, best_score
