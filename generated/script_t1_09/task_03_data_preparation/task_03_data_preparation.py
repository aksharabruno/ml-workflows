from dependency import *  # noqa: F401,F403


def data_preparation_3(X, df):
    y = df["quality"]

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_test_scaled, X_train_scaled, y, y_test, y_train
