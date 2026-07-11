# =============================================================================
# Brain Tumor MRI Classification using a Convolutional Neural Network (CNN)
# Team Members: Surianandhan Sridhar, Pattan Sameera Hussainy
# Model: Custom CNN (trained from scratch, PyTorch)
# Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
#
# Just download + extract the Kaggle dataset anywhere inside your Jupyter
# working directory. This script auto-detects the Training/ and Testing/
# folders wherever they are - no manual path editing needed.
# =============================================================================

# -----------------------------------------------------------------------------
# 0. AUTO-INSTALL DEPENDENCIES
# -----------------------------------------------------------------------------
import subprocess
import sys

def install_if_missing(pip_name, import_name=None):
    import_name = import_name or pip_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

for pip_name, import_name in [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("scikit-learn", "sklearn"),
]:
    install_if_missing(pip_name, import_name)

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------------------------------------------------------
# 1. CONFIG
# -----------------------------------------------------------------------------

IMG_SIZE    = 128
BATCH_SIZE  = 32
EPOCHS      = 15
LR          = 1e-3
VAL_SPLIT   = 0.15
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
print("  Saved fig1_class_distribution.png")

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

# -----------------------------------------------------------------------------
# 6. TRAINING CURVES
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5: Training Curves")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history["train_loss"], label="Train Loss", color="#3498db")
axes[0].plot(history["val_loss"], label="Val Loss", color="#e74c3c")
axes[0].set_title("Loss Curve", fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history["train_acc"], label="Train Accuracy", color="#3498db")
axes[1].plot(history["val_acc"], label="Val Accuracy", color="#e74c3c")
axes[1].set_title("Accuracy Curve", fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.suptitle("Figure 2: CNN Training Curves", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig2_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig2_training_curves.png")

# -----------------------------------------------------------------------------
# 7. TEST SET EVALUATION
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 6: Test Set Evaluation")
print("=" * 60)

model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.numpy())

test_acc = accuracy_score(all_targets, all_preds)
report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
cm = confusion_matrix(all_targets, all_preds)

print(f"\n  Test Accuracy: {test_acc:.4f}\n")
print(classification_report(all_targets, all_preds, target_names=class_names))

# Figure 3: Confusion Matrix
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor="gray")
plt.title(f"Figure 3: Confusion Matrix (Test Acc: {test_acc:.3f})", fontweight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("fig3_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig3_confusion_matrix.png")

# -----------------------------------------------------------------------------
# 8. SUMMARY EXPORT
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 7: Summary Export")
print("=" * 60)

summary_df = pd.DataFrame([{
    "Model": "Custom CNN (3 conv blocks)",
    "Test Accuracy": f"{test_acc:.4f}",
    "Precision (weighted)": f"{report['weighted avg']['precision']:.4f}",
    "Recall (weighted)": f"{report['weighted avg']['recall']:.4f}",
    "F1-Score (weighted)": f"{report['weighted avg']['f1-score']:.4f}",
    "Trainable Params": n_params,
    "Epochs": EPOCHS,
}])

summary_df.to_csv("model_summary.csv", index=False)
print(summary_df.to_string(index=False))
print("\n  Saved model_summary.csv")

print("\n" + "=" * 60)
print("ALL DONE! Figures and summary saved successfully.")
print("=" * 60)
