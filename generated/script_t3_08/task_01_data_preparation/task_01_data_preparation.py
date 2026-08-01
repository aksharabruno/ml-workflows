from dependency import *  # noqa: F401,F403


def data_preparation_1():
    df = df.dropna(subset=["type"]).copy()

    return df
