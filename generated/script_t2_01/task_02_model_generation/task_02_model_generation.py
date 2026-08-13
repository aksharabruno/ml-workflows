from dependency import *  # noqa: F401,F403


def model_generation_2(Xtest, Xtrain, Ytest, Ytrain):
    # get shapes
    N, D = Xtrain.shape
    K = len(set(Ytrain))

    # ANN with layers [784] -> [500] -> [300] -> [10]
    i = Input(shape=(D,))
    x = Dense(500, activation='relu')(i)
    x = Dense(300, activation='relu')(x)
    x = Dense(K, activation='softmax')(x)

    # instantiate the model object
    model = Model(inputs=i, outputs=x)

    # list of losses: https://keras.io/losses/
    # list of optimizers: https://keras.io/optimizers/
    # list of metrics: https://keras.io/metrics/
    model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy']
    )

    # note: multiple ways to choose a backend
    # either theano, tensorflow, or cntk
    # https://keras.io/backend/


    # gives us back a <keras.callbacks.History object at 0x112e61a90>
    r = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=15, batch_size=32)
    print("Returned:", r)

    return model, r
