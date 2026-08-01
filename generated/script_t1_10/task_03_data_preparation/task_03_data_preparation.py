from dependency import *  # noqa: F401,F403


def data_preparation_3(X, df):
    y = df['diagnosis']

    # --------------------------------
    # Train Test Split
    # --------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # --------------------------------
    # Feature Scaling
    # --------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_test, X_train, y_test, y_train
