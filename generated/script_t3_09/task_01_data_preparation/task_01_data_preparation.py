from dependency import *  # noqa: F401,F403


def data_preparation_1():
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load data (automatically downloads if not present)
    print("Loading SVHN dataset...")
    train_dataset = torchvision.datasets.SVHN(
        root=DATA_ROOT, split='train', download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.SVHN(
        root=DATA_ROOT, split='test', download=True, transform=test_transform
    )

    # Use num_workers=0 to avoid multiprocessing issues on Windows
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return test_loader, train_loader
