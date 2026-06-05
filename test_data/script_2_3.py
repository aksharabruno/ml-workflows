import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

import pandas_ml as pdml

def run_fraud_detection_example():
    df = pd.read_csv('creditcard.csv', low_memory=False)
    X = df.iloc[:,:-1]
    y = df['Class']

    df.head()

    frauds = df.loc[df['Class'] == 1]
    non_frauds = df.loc[df['Class'] == 0]
    print("We have", len(frauds), "fraud data points and", len(non_frauds), "regular data points.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    print("Size of training set: ", X_train.shape)

    model = Sequential()
    model.add(Dense(30, input_dim=30, activation='relu'))     # kernel_initializer='normal'
    model.add(Dense(1, activation='sigmoid'))                 # kernel_initializer='normal'
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    model.fit(X_train.values(), y_train, epochs=1)

    print("Loss: ", model.evaluate(X_test.values(), y_test, verbose=0))

    y_predicted = model.predict(X_test.values()).T[0].astype(int)

    y_right = np.array(y_test)
    confusion_matrix = pdml.ConfusionMatrix(y_right, y_predicted)
    print("Confusion matrix:\n%s" % confusion_matrix)
    confusion_matrix.plot(normalized=True)
    plt.show()

if __name__ == "__main__":
    run_fraud_detection_example()