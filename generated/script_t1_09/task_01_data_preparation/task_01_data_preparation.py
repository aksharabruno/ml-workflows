from dependency import *  # noqa: F401,F403


def data_preparation_1():
    # 1. Load Datasets
    red_wine = pd.read_csv("C:\\Users\\donsh\\Downloads\\wine+quality\\winequality-red.csv", sep=";")
    white_wine = pd.read_csv("C:\\Users\\donsh\\Downloads\\wine+quality\\winequality-white.csv", sep=";")

    return red_wine, white_wine
