import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import streamlit as st

# ================= CONFIG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data/flowers"
MODEL_PATH = "flower_resnet18.pth"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4
PATIENCE = 5
SEED = 42

CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

torch.manual_seed(SEED)


def prepare_data():
    dataset = datasets.ImageFolder(DATA_DIR)

    train_len = int(0.8 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_s, val_s, test_s = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(WrappedDataset(train_s, train_tf), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WrappedDataset(val_s, test_tf), BATCH_SIZE)
    test_loader = DataLoader(WrappedDataset(test_s, test_tf), BATCH_SIZE)

    return train_loader, val_loader, test_loader

class FlowerModel(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)

    def forward(self, x):
        return self.model(x)
