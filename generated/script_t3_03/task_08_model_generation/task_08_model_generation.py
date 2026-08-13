from dependency import *  # noqa: F401,F403


def model_generation_8(loss_fn, modelnew, optimizer, scheduler, train_loader, val_loader):
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

