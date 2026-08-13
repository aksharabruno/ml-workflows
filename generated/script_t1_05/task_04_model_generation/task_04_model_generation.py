from dependency import *  # noqa: F401,F403


def model_generation_4(train_X, train_y):
    # Define a random forest model
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(train_X, train_y)
    return rf_model
