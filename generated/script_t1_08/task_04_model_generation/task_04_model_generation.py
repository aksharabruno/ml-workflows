from dependency import *  # noqa: F401,F403


def model_generation_4(X_test, X_train, y_test, y_train):
    # Build neural network
    model = Sequential()

    model.add(
        Dense(
            units=4,
            activation="relu",
            input_dim=X_train.shape[1]
        )
    )

    model.add(
        Dense(
            units=2,
            activation="relu"
        )
    )

    model.add(
        Dense(
            units=1,
            activation="sigmoid"
        )
    )

    # Compile model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=50,
        validation_data=(X_test, y_test)
    )

    return model
