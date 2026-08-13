from dependency import *  # noqa: F401,F403


def data_preparation_4():
    # جلب وتجهيز البيانات باستخدام كود الطالب 1
    df = get_data()
    X, y = preprocess_data(df)

    # تدريب وتقييم الموديل
    model, X_test, y_test, y_pred, mae, r2 = train_and_evaluate(X, y)

    return X, mae, model, r2
