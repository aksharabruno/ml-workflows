from dataset import AnimalDataset
from model import CNN
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import os
from config import Config

IMAGE_SIZE = Config.IMAGE_SIZE
BATCH_SIZE = Config.BATCH_SIZE
EPOCHS = Config.EPOCHS
LEARNING_RATE = Config.LEARNING_RATE

DATASET_PATH = Config.DATASET_PATH
LAST_SAVE_PATH = Config.LAST_SAVE_PATH
BEST_SAVE_PATH = Config.BEST_SAVE_PATH


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
