from dependency import *  # noqa: F401,F403


def data_preparation_3(X, y):
    X.head()

    # Split into validation and training data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    return train_X, train_y, val_X, val_y
