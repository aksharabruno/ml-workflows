from dependency import *  # noqa: F401,F403


def model_generation_2(X_train, y_train):
    # --------------------------------
    # Train Logistic Regression
    # --------------------------------

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model
