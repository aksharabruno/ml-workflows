from dependency import *  # noqa: F401,F403


def model_evaluation_10(X_test_embeddings, final_svm_model, label_encoder, y_test):
    # Predict on the test set
    print("Predicting on the test set...")
    y_pred = final_svm_model.predict(X_test_embeddings)
    y_pred_proba = final_svm_model.predict_proba(X_test_embeddings)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate classification report
    classification_rep = classification_report(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7], digits=4)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_rep)

    # Save the model and label encoder
    print("Saving the model and label encoder...")
    joblib.dump(final_svm_model, 'best_svm_cyberbullying_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

    # Calculate the AUC-ROC curve
    print("Calculating the AUC-ROC curve...")
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(len(label_encoder.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot the AUC-ROC curve
    plt.figure(figsize=(10, 8))
    for i in range(len(label_encoder.classes_)):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC Curve for SVM')
    plt.legend(loc="lower right")
    plt.show()

    # Generate and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[i for i in range(len(label_encoder.classes_))])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for SVM')
    plt.show()

    # Output additional metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")

    print("Done!")
