from dependency import *  # noqa: F401,F403


def data_preparation_1():
        # 1. تقسيم البيانات (80% تدريب و 20% اختبار)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_test, X_train, y_test, y_train
