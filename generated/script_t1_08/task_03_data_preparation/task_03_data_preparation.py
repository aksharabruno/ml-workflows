from dependency import *  # noqa: F401,F403


def data_preparation_3(X, df):
    y = df["loan_sanctioned"]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_test, X_train, scaler, y_test, y_train
