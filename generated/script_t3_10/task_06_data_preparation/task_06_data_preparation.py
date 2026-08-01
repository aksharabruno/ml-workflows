from dependency import *  # noqa: F401,F403


def data_preparation_6(best_params, df):
    print("Best parameters found: ", best_params)

    # Split the full dataset into training and testing sets
    X = df['tweet_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_test, X_train, y_test, y_train
