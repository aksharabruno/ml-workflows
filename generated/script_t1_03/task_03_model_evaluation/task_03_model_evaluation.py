from dependency import *  # noqa: F401,F403


def model_evaluation_3(X_test, gbm, y_test):
    print("Starting predicting...")
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
    print(f"The RMSE of prediction is: {rmse_test}")
