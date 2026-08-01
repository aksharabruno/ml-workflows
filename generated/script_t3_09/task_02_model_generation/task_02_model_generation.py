from dependency import *  # noqa: F401,F403


def model_generation_2(test_dataset, test_loader, train_dataset, train_loader):
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

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
    return best_acc, criterion, model, test_accs, test_losses, train_accs, train_losses
