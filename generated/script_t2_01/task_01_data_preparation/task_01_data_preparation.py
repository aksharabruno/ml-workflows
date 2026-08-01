from dependency import *  # noqa: F401,F403


def data_preparation_1():
    # get the data
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

    # get shapes
    N, D = Xtrain.shape
    return D, Xtest, Xtrain, Ytest, Ytrain
