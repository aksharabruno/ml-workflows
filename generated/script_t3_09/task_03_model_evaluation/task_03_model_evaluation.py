from dependency import *  # noqa: F401,F403


def model_evaluation_3(best_acc, criterion, model, test_accs, test_loader, test_losses, train_accs, train_losses):
    print(f"\nTraining completed. Best test accuracy: {best_acc:.4f}")

    # Plot curves
    epochs = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accs, 'r-', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "training_curves.png", dpi=150)
    plt.show()
    print(f"Training curves saved to {PLOT_DIR / 'training_curves.png'}")

    # Final evaluation with best model
    model.load_state_dict(torch.load(MODEL_DIR / "best_svhn_cnn.pth", map_location=device))
    final_loss, final_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal test accuracy (best model): {final_acc:.4f} ({final_acc*100:.2f}%)")

    # Sample predictions
    model.eval()
    sample_images, sample_labels = next(iter(test_loader))
    sample_images = sample_images[:10].to(device)
    sample_labels = sample_labels[:10].cpu().numpy()
    with torch.no_grad():
        outputs = model(sample_images)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    for i in range(10):
        img = sample_images[i].cpu().numpy().transpose((1, 2, 0))
        img = img * 0.5 + 0.5  # denormalize
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"True: {sample_labels[i]}, Pred: {preds[i]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "sample_predictions.png", dpi=150)
    plt.show()
    print(f"Sample predictions saved to {PLOT_DIR / 'sample_predictions.png'}")

    # Print architecture diagram
    print("\n" + "="*60)
    print("CNN Architecture Diagram (text description):")
    print("="*60)
    print("""
    Input: 32x32x3 (RGB)
    |
    Conv2d(3->32, 3x3, pad=1) → BatchNorm → ReLU → MaxPool2d → 16x16x32
    |
    Conv2d(32->64, 3x3, pad=1) → BatchNorm → ReLU → MaxPool2d → 8x8x64
    |
    Conv2d(64->128, 3x3, pad=1) → BatchNorm → ReLU → MaxPool2d → 4x4x128
    |
    Flatten → 2048
    |
    Linear(2048->256) → BatchNorm → ReLU → Dropout(0.5)
    |
    Linear(256->128) → BatchNorm → ReLU → Dropout(0.5)
    |
    Linear(128->10) → logits (0-9)
    """)

    print("\nProject completed successfully.")
    print(f"All outputs saved in: {OUTPUT_DIR.resolve()}")
