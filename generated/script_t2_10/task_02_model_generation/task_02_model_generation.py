from dependency import *  # noqa: F401,F403


def model_generation_2(x_train, y_train):
    lr = LinearRegression()

    lr.fit(x_train, y_train)

    return lr
