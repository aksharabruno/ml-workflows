# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv("loan_data.csv")

# View first 5 rows
print(df.head())

# Dataset information
print(df.info())

# Encode categorical columns
le = LabelEncoder()

df["employment_status"] = le.fit_transform(df["employment_status"])
df["loan_sanctioned"] = le.fit_transform(df["loan_sanctioned"])

# Check updated data types
print(df.info())

# Features and target
X = df[[
    "income",
    "credit_score",
    "loan_amount",
    "employment_status"
]]

y = df["loan_sanctioned"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

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

# New data prediction
new_data = np.array([
    [45000, 450, 12000, 1]  # Employment status encoded value
])

# Scale input
new_scaled = scaler.transform(new_data)

# Predict probability
prediction_prob = model.predict(new_scaled)

# Convert to binary prediction
prediction = (prediction_prob > 0.5).astype(int)

# Display result
if prediction[0][0] == 1:
    print("Loan Sanctioned: Yes")
else:
    print("Loan Sanctioned: No")