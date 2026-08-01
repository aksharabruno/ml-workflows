from dependency import *  # noqa: F401,F403


def data_preparation_1():
    # 1. Load Datasets
    red_wine = pd.read_csv("C:\\Users\\donsh\\Downloads\\wine+quality\\winequality-red.csv", sep=";")
    white_wine = pd.read_csv("C:\\Users\\donsh\\Downloads\\wine+quality\\winequality-white.csv", sep=";")

    # Add wine type feature
    red_wine["wine_type"] = 0   # Red
    white_wine["wine_type"] = 1 # White

    # Combine datasets
    df = pd.concat([red_wine, white_wine], axis=0)

    return df
