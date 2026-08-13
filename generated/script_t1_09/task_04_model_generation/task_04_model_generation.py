from dependency import *  # noqa: F401,F403


def model_generation_4(X_train_scaled, y_train):
    # 5. Logistic Regression Model
    log_model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs"
    )

    log_model.fit(X_train_scaled, y_train)

    return log_model
