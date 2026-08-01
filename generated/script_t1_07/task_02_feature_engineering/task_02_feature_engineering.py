from dependency import *  # noqa: F401,F403


def feature_engineering_2(df):
    # Features and Target
    X = df[['Hours_Studied', 'Attendance']]
    return X
