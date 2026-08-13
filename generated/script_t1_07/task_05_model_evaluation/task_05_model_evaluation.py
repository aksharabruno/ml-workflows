from dependency import *  # noqa: F401,F403


def model_evaluation_5(X_test, Y_test, model):
    # Predictions & Evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)

    print("--- Test Set Predictions ---")
    print(f"Predictions: {predictions}")
    print(f"Accuracy Score: {accuracy * 100:.2f}%\n")

    print("--- Classification Report ---")
    print(classification_report(Y_test, predictions))
