from dependency import *  # noqa: F401,F403


def feature_engineering_2(home_data):
    # Create X (After completing the exercise, you can return to modify this line!)
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

    # Select columns corresponding to features, and preview the data
    X = home_data[features]
    X.head()

    return X
