from dependency import *  # noqa: F401,F403


def model_evaluation_4(X_test, X_val, pipeline, y_test, y_val):
    val_metrics = evaluate_model(pipeline, X_val, y_val)
    test_metrics = evaluate_model(pipeline, X_test, y_test)

    return test_metrics, val_metrics
