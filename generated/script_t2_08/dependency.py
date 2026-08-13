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

