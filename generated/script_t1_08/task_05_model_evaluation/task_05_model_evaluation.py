from dependency import *  # noqa: F401,F403


def model_evaluation_5(model, scaler):
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
