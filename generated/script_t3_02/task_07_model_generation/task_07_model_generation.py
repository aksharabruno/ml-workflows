from dependency import *  # noqa: F401,F403


def model_generation_7(X_train, y_train):
    if task == "classification":
        base = build_classifier_pipeline(text_features=text_features)
        if tune:
            pipe = _tune_classifier(base, X_train, y_train)
        else:
            pipe = clone(base)
            pipe.fit(X_train, y_train)
    else:
        pipe = build_regression_pipeline(text_features=text_features)
        pipe.fit(X_train, y_train)

    return pipe
