from dependency import *  # noqa: F401,F403


def model_evaluation_3(model, x_test_scaled, y_test):
    y_pred = model.predict(x_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")

