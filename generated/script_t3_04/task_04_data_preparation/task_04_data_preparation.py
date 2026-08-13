from dependency import *  # noqa: F401,F403


def data_preparation_4(X_train, X_val, y_train, y_val):
    # --- 5. Datasets PyTorch ---
    train_dataset = TextClassificationDataset(X_train, y_train, tokenizer, args.max_length)
    val_dataset   = TextClassificationDataset(X_val,   y_val,   tokenizer, args.max_length)

    # DataLoaders : shuffle=True pour le train, False pour la val
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader
