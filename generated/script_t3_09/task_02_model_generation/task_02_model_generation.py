from dependency import *  # noqa: F401,F403


def model_generation_2(test_loader, train_loader):
    # Model, loss, optimizer, scheduler
    model = SVHN_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    print("\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Training loop
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc = 0.0

    print("\nStarting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        scheduler.step(test_acc)

        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_svhn_cnn.pth")
            print(f"  -> New best model saved (acc={test_acc:.4f})")

    # Save final model
    torch.save(model.state_dict(), MODEL_DIR / "final_svhn_cnn.pth")
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

    return criterion, model
