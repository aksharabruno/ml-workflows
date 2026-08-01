from dependency import *  # noqa: F401,F403


def feature_engineering_2(df):
    print("Dataset Shape:", df.shape)
    print(df.head())

    # 2. Define Features & Target
    X = df.drop("quality", axis=1)
    return X
