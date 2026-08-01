from dependency import *  # noqa: F401,F403


def model_evaluation_6(X_test, model, y_test):
    # Evaluate


    score = evaluate(
        args.model,
        model,
        X_test,
        y_test
    )



