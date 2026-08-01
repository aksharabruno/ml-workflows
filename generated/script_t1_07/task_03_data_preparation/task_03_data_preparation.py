from dependency import *  # noqa: F401,F403


def data_preparation_3(X, df):
    Y = df['Pass']

    # Train / Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.19, random_state=42
    )

    return X_test, X_train, Y_test, Y_train
