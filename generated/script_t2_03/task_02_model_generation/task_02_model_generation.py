from dependency import *  # noqa: F401,F403


def model_generation_2(X_test, X_train):
    # build model
    input_dim = X_train.shape[1]
    encoding_dim = 14

    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="tanh", 
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)

    decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)

    nb_epoch = 100
    batch_size = 32

    autoencoder.compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath="model.h5",
                                verbose=0,
                                save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs',
                            histogram_freq=0,
                            write_graph=True,
                            write_images=True)

    history = autoencoder.fit(X_train, X_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(X_test, X_test),
                        verbose=1,
                        callbacks=[checkpointer, tensorboard]).history

    autoencoder = load_model('model.h5')

    return autoencoder, history
