from dependency import *  # noqa: F401,F403


def model_evaluation_3(Xtest, Ytest, model, r):
    # print the available keys
    # should see: dict_keys(['val_loss', 'acc', 'loss', 'val_acc'])
    print(r.history.keys())

    # plot some data
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    # accuracies
    plt.plot(r.history['acc'], label='acc')
    plt.plot(r.history['val_acc'], label='val_acc')
    plt.legend()
    plt.show()


    # make predictions and evaluate
    probs = model.predict(Xtest) # N x K matrix of probabilities
    Ptest = np.argmax(probs, axis=1)
    print("Validation acc:", np.mean(Ptest == Ytest))
