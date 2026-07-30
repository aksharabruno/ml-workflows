import numpy as np
from keras.losses import binary_crossentropy

# Example true labels and predicted probabilities
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

# Compute Binary Cross-Entropy using NumPy
def binary_cross_entropy(y_true, y_pred):
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce

bce_loss = binary_cross_entropy(y_true, y_pred)
print(f"Binary Cross-Entropy Loss (manual calculation): {bce_loss}")

# Compute Binary Cross-Entropy using Keras
bce_loss_keras = binary_crossentropy(y_true, y_pred).numpy()
print(f"Binary Cross-Entropy Loss (Keras): {bce_loss_keras}")