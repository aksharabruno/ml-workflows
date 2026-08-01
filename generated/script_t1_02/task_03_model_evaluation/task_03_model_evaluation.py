from dependency import *  # noqa: F401,F403


def model_evaluation_3(X_test, bst):
    # make predictions
    preds = bst.predict(X_test)
