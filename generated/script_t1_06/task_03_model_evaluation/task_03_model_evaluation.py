from dependency import *  # noqa: F401,F403


def model_evaluation_3(X_test, model, y_test):
    # 4. التوقع y = mx + c
    y_pred = model.predict(X_test)

    # 5. حساب metrics (التقييم)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mae, r2
