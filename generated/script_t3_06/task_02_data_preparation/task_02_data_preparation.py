from dependency import *  # noqa: F401,F403


def data_preparation_2():
    """Train multiple models and save the best candidate to disk."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    return X_test, X_train, X_val, y_test, y_train, y_val
