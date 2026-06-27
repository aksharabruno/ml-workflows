import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.preprocessing import StandardScaler

def run_autoencoder_example():
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)

    rcParams['figure.figsize'] = 14, 8

    RANDOM_SEED = 42
    LABELS = ["Normal", "Fraud"]

    df = pd.read_csv("data/creditcard.csv")
    count_classes = pd.value_counts(df['Class'], sort = True)
    count_classes.plot(kind = 'bar', rot=0)
    plt.title("Transaction class distribution")
    plt.xticks(range(2), LABELS)
    plt.xlabel("Class")
    plt.ylabel("Frequency")

    frauds = df[df.Class == 1]
    normal = df[df.Class == 0]

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.suptitle('Amount per transaction by class')

    bins = 50

    ax1.hist(frauds.Amount, bins = bins)
    ax1.set_title('Fraud')

    ax2.hist(normal.Amount, bins = bins)
    ax2.set_title('Normal')

    plt.xlabel('Amount ($)')
    plt.ylabel('Number of Transactions')
    plt.xlim((0, 20000))
    plt.yscale('log')
    plt.show();

    # prepare data
    data = df.drop(['Time'], axis=1)
    data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
    X_train = X_train[X_train.Class == 0]
    X_train = X_train.drop(['Class'], axis=1)

    y_test = X_test['Class']
    X_test = X_test.drop(['Class'], axis=1)

    X_train = X_train.values
    X_test = X_test.values

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

if __name__ == "__main__":
    run_autoencoder_example()