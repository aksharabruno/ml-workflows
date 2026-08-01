from dependency import *  # noqa: F401,F403


def model_evaluation_5(X_test, autoencoder, history, y_test):
    # evaluation
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    predictions = autoencoder.predict(X_test)

    mse = np.mean(np.power(X_test - predictions, 2), axis=1)
    error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})

    error_df.describe()

