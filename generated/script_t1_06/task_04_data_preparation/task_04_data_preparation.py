from dependency import *  # noqa: F401,F403


def data_preparation_4():
    # جلب وتجهيز البيانات باستخدام كود الطالب 1
    df = get_data()
    X, y = preprocess_data(df)

    return X
