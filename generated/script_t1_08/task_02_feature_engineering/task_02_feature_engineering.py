from dependency import *  # noqa: F401,F403


def feature_engineering_2(df):
    # Check updated data types
    print(df.info())

    # Features and target
    X = df[[
        "income",
        "credit_score",
        "loan_amount",
        "employment_status"
    ]]

    return X
