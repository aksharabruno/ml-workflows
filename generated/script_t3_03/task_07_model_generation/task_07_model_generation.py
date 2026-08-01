from dependency import *  # noqa: F401,F403


def model_generation_7(loss_fn, modelnew, optimizer, scheduler, train_loader, val_loader):
    def train_epoch(modelnew, optimizer, loss_fn, data_loader, device="cpu"):
        training_loss = 0.0
        modelnew.train()

        for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = modelnew(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        return training_loss / len(data_loader.dataset)

    def train(
        modelnew,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        epochs=20,
        device="cpu",
        scheduler=None,
        checkpoint_path=None,
        early_stopping=None,
    ):
        # Track metrics
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        learning_rates = []

        best_val_loss = float("inf")
        early_stopping_counter = 0


        for epoch in range(1, epochs + 1):
            print("\n")
            print(f"Starting epoch {epoch}/{epochs}")

            # Train one epoch
            train_epoch(modelnew, optimizer, loss_fn, train_loader, device)

            # Evaluate training
            train_loss, train_accuracy = score(modelnew, train_loader, loss_fn, device)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Evaluate validation
            validation_loss, validation_accuracy = score(modelnew, val_loader, loss_fn, device)
            val_losses.append(validation_loss)
            val_accuracies.append(validation_accuracy)

            print(f"Epoch: {epoch}")
            print(f"Training loss: {train_loss:.4f}")
            print(f"Training accuracy: {train_accuracy*100:.4f}%")
            print(f"Validation loss: {validation_loss:.4f}")
            print(f"Validation accuracy: {validation_accuracy*100:.4f}%")

            # Log LR
            lr = optimizer.param_groups[0]["lr"]
            learning_rates.append(lr)

            if scheduler:
                scheduler.step()

            # Checkpointing
            if checkpoint_path:
                checkpointing(
                    validation_loss, best_val_loss, modelnew, optimizer, checkpoint_path
                )

            # Early stopping
            if early_stopping:
                early_stopping_counter, stop = early_stopping(
                    validation_loss, best_val_loss, early_stopping_counter
                )
                if stop:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break

            # Update best loss
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss

        return (
            learning_rates,
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            epoch,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelnew.to(device)

    epochs_to_train = 50

    checkpoint_path = Path(
        r"C:\Users\Jelil\Desktop\New folder (7)\best_modelnew.pth"
    )

    train_results = train(
        modelnew,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        epochs=epochs_to_train,
        device=device,
        scheduler=scheduler,
        checkpoint_path= checkpoint_path,
        early_stopping=early_stopping,
        )

    checkpoint = torch.load(
        r"C:\Users\Jelil\Desktop\New folder (7)\best_modelnew.pth",
        map_location="cpu"
    )

    modelnew.load_state_dict(checkpoint["modelnew_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    modelnew.eval()

