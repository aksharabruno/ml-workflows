from dependency import *  # noqa: F401,F403


def model_evaluation_3(X_test, model, y_test):
    # --------------------------------
    # Prediction
    # --------------------------------

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:,1]

    # --------------------------------
    # Accuracy
    # --------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy")

    print(round(accuracy*100,2), "%")

    # --------------------------------
    # Precision
    # --------------------------------

    precision = precision_score(y_test, y_pred)

    print("\nPrecision")

    print(round(precision,4))

    # --------------------------------
    # Recall
    # --------------------------------

    recall = recall_score(y_test, y_pred)

    print("\nRecall")

    print(round(recall,4))

    # --------------------------------
    # ROC AUC
    # --------------------------------

    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nROC AUC Score")

    print(round(roc_auc,4))

    # --------------------------------
    # Classification Report
    # --------------------------------

    print("\nClassification Report\n")

    print(classification_report(y_test, y_pred))

    # --------------------------------
    # Confusion Matrix
    # --------------------------------

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign","Malignant"]
    )

    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")

    plt.show()

    # --------------------------------
    # ROC Curve
    # --------------------------------

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    plt.figure(figsize=(7,5))

    plt.plot(fpr,tpr,label="ROC Curve")

    plt.plot([0,1],[0,1],'r--')

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.show()

    # --------------------------------
    # Threshold Example
    # --------------------------------

    threshold = 0.60

    custom_prediction = (y_prob >= threshold).astype(int)

    print("\nPrediction using Threshold =", threshold)

    print(custom_prediction[:20])

    # --------------------------------
    # Sigmoid Function
    # --------------------------------

    x = np.linspace(-10,10,200)

    sigmoid = 1/(1+np.exp(-x))

    plt.figure(figsize=(7,5))

    plt.plot(x,sigmoid)

    plt.title("Sigmoid Function")

    plt.xlabel("x")

    plt.ylabel("Sigmoid(x)")

    plt.grid(True)

    plt.show()
