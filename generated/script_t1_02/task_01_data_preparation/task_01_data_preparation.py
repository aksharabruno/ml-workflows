from dependency import *  # noqa: F401,F403


def data_preparation_1():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)
    return X_test, X_train, y_train
