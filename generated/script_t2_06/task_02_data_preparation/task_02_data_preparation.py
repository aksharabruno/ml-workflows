from dependency import *  # noqa: F401,F403


def data_preparation_2(transform):
    # Create TRAIN dataset
    dataset_train = AnimalDataset(
        root=DATASET_PATH,
        transform=transform,
        train=True
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    # Create VAL dataset
    dataset_val = AnimalDataset(
        root=DATASET_PATH,
        transform=transform,
        train=False
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    return dataloader_train, dataloader_val, dataset_train
