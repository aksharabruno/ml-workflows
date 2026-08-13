from dependency import *  # noqa: F401,F403


def model_evaluation_5(X_test_scaled, log_model, y, y_test):
    y_pred = log_model.predict(X_test_scaled)

    # 6. Model Evaluation
    print("\n📌 Logistic Regression Performance")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 7. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Wine Quality")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # 8. Multiclass ROC-AUC (One-vs-Rest)
    y_test_bin = label_binarize(y_test, classes=sorted(y.unique()))
    y_prob = log_model.predict_proba(X_test_scaled)

    roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr")

    print("ROC-AUC Score (OvR):", roc_auc)
