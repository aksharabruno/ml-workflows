from dependency import *  # noqa: F401,F403


def model_generation_4(X_train, Y_train):
    # Model Training
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    return model
