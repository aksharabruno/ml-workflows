from dependency import *  # noqa: F401,F403


def model_generation_4(x_train, y_train):
    ls  = Lasso(alpha=0.01)

    ls.fit(x_train, y_train)

    return ls
