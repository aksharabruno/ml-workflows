import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_and_train_toxicity_model():
    dataset_path = 'jigsaw-toxic-comment-train.csv'
    df = pd.read_csv(dataset_path).sample(5000) # Subset for speed
    
    X = df['comment_text'].values
    y = df['toxic'].values
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    # Vectorization layer (Standard TF Preprocessing)
    vectorizer = layers.TextVectorization(max_tokens=20000, output_sequence_length=200)
    vectorizer.adapt(X_train)
    
    # training
    model = tf.keras.Sequential([
        layers.Input(shape=(1,), dtype=tf.string),
        vectorizer,
        layers.Embedding(20000, 128),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=2, validation_data=(X_val, y_val))
    
    # evaluation
    results = model.evaluate(X_val, y_val)
    print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

if __name__ == "__main__":
    prepare_and_train_toxicity_model()