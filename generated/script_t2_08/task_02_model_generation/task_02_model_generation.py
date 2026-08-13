from dependency import *  # noqa: F401,F403


def model_generation_2(class_names, train_loader, val_loader):
    # -----------------------------------------------------------------------------
    # 4. MODEL DEFINITION
    # -----------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 3: Building CNN")
    print("=" * 60)

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            )
            reduced = IMG_SIZE // 8  # three MaxPool2d(2) halvings
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * reduced * reduced, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    model = SimpleCNN(num_classes=len(class_names)).to(device)
    print(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -----------------------------------------------------------------------------
    # 5. TRAINING
    # -----------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 4: Training")
    print("=" * 60)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def run_epoch(loader, train_mode):
        model.train() if train_mode else model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train_mode):
            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                if train_mode:
                    optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                if train_mode:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == targets).sum().item()
                total += images.size(0)
        return total_loss / total, correct / total

    best_val_acc = 0.0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(train_loader, train_mode=True)
        val_loss, val_acc = run_epoch(val_loader, train_mode=False)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

        print(f"  Epoch {epoch:2d}/{EPOCHS} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    print(f"\n  Training completed in {time.time() - start:.1f}s")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")

    # Load best weights before final evaluation
    model.load_state_dict(torch.load("best_model.pt"))

    return history, model, n_params
