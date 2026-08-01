from dependency import *  # noqa: F401,F403


def model_evaluation_3(class_names, history, model, n_params, test_loader):
    # -----------------------------------------------------------------------------
    # 6. TRAINING CURVES
    # -----------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 5: Training Curves")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", color="#3498db")
    axes[0].plot(history["val_loss"], label="Val Loss", color="#e74c3c")
    axes[0].set_title("Loss Curve", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Accuracy", color="#3498db")
    axes[1].plot(history["val_acc"], label="Val Accuracy", color="#e74c3c")
    axes[1].set_title("Accuracy Curve", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.suptitle("Figure 2: CNN Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("fig2_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig2_training_curves.png")

    # -----------------------------------------------------------------------------
    # 7. TEST SET EVALUATION
    # -----------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 6: Test Set Evaluation")
    print("=" * 60)

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    test_acc = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_targets, all_preds)

    print(f"\n  Test Accuracy: {test_acc:.4f}\n")
    print(classification_report(all_targets, all_preds, target_names=class_names))

    # Figure 3: Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="gray")
    plt.title(f"Figure 3: Confusion Matrix (Test Acc: {test_acc:.3f})", fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("fig3_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig3_confusion_matrix.png")

    # -----------------------------------------------------------------------------
    # 8. SUMMARY EXPORT
    # -----------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 7: Summary Export")
    print("=" * 60)

    summary_df = pd.DataFrame([{
        "Model": "Custom CNN (3 conv blocks)",
        "Test Accuracy": f"{test_acc:.4f}",
        "Precision (weighted)": f"{report['weighted avg']['precision']:.4f}",
        "Recall (weighted)": f"{report['weighted avg']['recall']:.4f}",
        "F1-Score (weighted)": f"{report['weighted avg']['f1-score']:.4f}",
        "Trainable Params": n_params,
        "Epochs": EPOCHS,
    }])

    summary_df.to_csv("model_summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print("\n  Saved model_summary.csv")

    print("\n" + "=" * 60)
    print("ALL DONE! Figures and summary saved successfully.")
    print("=" * 60)
