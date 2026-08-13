from dependency import *  # noqa: F401,F403


def data_preparation_4():
    train_loader, val_loader, _ = prepare_data()

    return train_loader, val_loader
