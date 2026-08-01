from dependency import *  # noqa: F401,F403


def model_generation_4(X_test, X_train, batch_size, checkpointer, nb_epoch):
    history = autoencoder.fit(X_train, X_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(X_test, X_test),
                        verbose=1,
                        callbacks=[checkpointer, tensorboard]).history

    autoencoder = load_model('model.h5')

    return autoencoder, history
