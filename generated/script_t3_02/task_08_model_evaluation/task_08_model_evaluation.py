from dependency import *  # noqa: F401,F403


def model_evaluation_8(X_test, X_train, pipe, y_test):
    metrics = _metrics_dict_fixed(task, pipe, X_test, y_test, n_train=len(X_train))
    metrics["label_scheme"] = label_scheme if task == "classification" else "n/a"
    metrics["text_features"] = text_features

    return metrics
