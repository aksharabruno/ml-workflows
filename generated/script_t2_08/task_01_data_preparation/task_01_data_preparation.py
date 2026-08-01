from dependency import *  # noqa: F401,F403


def data_preparation_1():
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------------
    # 2. AUTO-DETECT DATASET LOCATION
    # -----------------------------------------------------------------------------

    print("=" * 60)
    print("STEP 1: Locating Dataset")
    print("=" * 60)

    def find_train_test_dirs(start="."):
        """Walk the directory tree looking for sibling Training/ and Testing/
        folders (case-insensitive), so the user never has to hardcode a path."""
        for root, dirs, _ in os.walk(start):
            # skip hidden/system dirs and common noise to keep the walk fast
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in
                       ("__pycache__", "node_modules", ".git", ".ipynb_checkpoints")]
            dirs_lower = {d.lower(): d for d in dirs}
            if "training" in dirs_lower and "testing" in dirs_lower:
                return (os.path.join(root, dirs_lower["training"]),
                         os.path.join(root, dirs_lower["testing"]))
        return None, None

    train_dir, test_dir = find_train_test_dirs(".")

    if train_dir is None:
        raise FileNotFoundError(
            "Could not find 'Training' and 'Testing' folders anywhere under the "
            "current directory. Download + extract the Kaggle brain-tumor-mri-dataset "
            "somewhere inside this notebook's working folder, then re-run this cell."
        )

    print(f"  Train dir: {train_dir}")
    print(f"  Test  dir: {test_dir}")

    # -----------------------------------------------------------------------------
    # 3. DATA LOADING
    # -----------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 2: Loading Dataset")
    print("=" * 60)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    full_train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    test_ds       = datasets.ImageFolder(test_dir, transform=eval_transform)

    class_names = full_train_ds.classes
    print(f"Classes: {class_names}")

    val_size   = int(len(full_train_ds) * VAL_SPLIT)
    train_size = len(full_train_ds) - val_size
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(SEED))

    # validation set should use eval transform (no augmentation) - override
    val_ds.dataset.transform = eval_transform

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Training samples  : {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    print(f"  Testing samples   : {len(test_ds)}")

    # Class distribution figure
    labels = [full_train_ds.samples[i][1] for i in range(len(full_train_ds))]
    counts = pd.Series(labels).map(dict(enumerate(class_names))).value_counts()

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar", color=["#2ecc71", "#f39c12", "#3498db", "#e74c3c"], edgecolor="black")
    plt.title("Figure 1: Training Set Class Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("fig1_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    return class_names, test_loader, train_loader, val_loader
