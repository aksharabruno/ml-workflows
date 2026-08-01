from dependency import *  # noqa: F401,F403


def model_evaluation_10(X_test_embeddings, final_svm_model, y_test):
    # Predict on the test set
    print("Predicting on the test set...")
    y_pred = final_svm_model.predict(X_test_embeddings)
    y_pred_proba = final_svm_model.predict_proba(X_test_embeddings)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate classification report
    classification_rep = classification_report(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7], digits=4)

    return accuracy, classification_rep, y_pred, y_pred_proba
