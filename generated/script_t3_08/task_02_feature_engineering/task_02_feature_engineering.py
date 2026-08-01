from dependency import *  # noqa: F401,F403


def feature_engineering_2(df):
    # Features we want to use
    feature_columns = [
        "title",
        "description",
        "director",
        "country",
        "release_year"
    ]

    X = df[feature_columns].copy()

    return X
