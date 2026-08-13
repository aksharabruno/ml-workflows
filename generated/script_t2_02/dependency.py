#!/usr/bin/env python
# coding: utf-8

# 20 Newsgroups text classification with pre-trained word embeddings
#
# In this script, we'll use pre-trained [GloVe word
# embeddings](http://nlp.stanford.edu/projects/glove/) for text
# classification using PyTorch.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from packaging.version import Version as LV

from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from datetime import datetime

import os
import sys

import numpy as np

torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)
assert(LV(torch.__version__) >= LV("1.0.0"))



try:
    import tensorboardX
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logdir = os.path.join(os.getcwd(), "logs", "20ng-cnn-" + time_str)
    print('TensorBoard log directory:', logdir)
    os.makedirs(logdir)
    log = tensorboardX.SummaryWriter(logdir)
except (ImportError, FileExistsError):
    log = None


class Net(nn.Module):
    def __init__(self, embedding_matrix):
        super(Net, self).__init__()
        self.emb = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.layers = nn.Sequential(
            nn.Conv1d(100, 128, 5),  # output: batch_size x 128 x seq_len-4
            nn.ReLU(),
            nn.MaxPool1d(5),         # output: bs x 128 x 199
            nn.Conv1d(128, 128, 5),  # output: bs x 128 x 199
            nn.ReLU(),
            nn.MaxPool1d(5),         # output: bs x 128 x 39
            nn.Conv1d(128, 128, 5),  # output: bs x 128 x 35
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # output: bs x 128 x 1
            )
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )

    def forward(self, x):
        x = self.emb(x)      # output from embedding: batch_size x seq_len x embedding dim.
        x = x.transpose(1,2) # change to: batch_size x embedding_dim x seq_len
        x = self.layers(x)
        x = self.linear_layers(x)
        return x

def correct(output, target):
    predicted = output.argmax(1) # pick class with largest network output
    correct_ones = (predicted == target).type(torch.float)
    return correct_ones.sum().item()

def train(data_loader, model, criterion, optimizer):
    model.train()

    num_batches = 0
    num_items = 0

    total_loss = 0
    total_correct = 0
    for data, target in data_loader:
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device)

        # Do a forward pass
        output = model(data)

        # Calculate the loss
        loss = criterion(output, target)
        total_loss += loss
        num_batches += 1

        # Count number of correct
        total_correct += correct(output, target)
        num_items += len(target)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return {
        'loss': total_loss/num_batches,
        'accuracy': total_correct/num_items
        }

def test(test_loader, model, criterion):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Copy data and targets to GPU
            data = data.to(device)
            target = target.to(device)

            # Do a forward pass
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()

            # Count number of correct digits
            total_correct += correct(output, target)

    return {
        'loss': test_loss/num_batches,
        'accuracy': total_correct/num_items
    }

def log_measures(ret, log, prefix, epoch):
    if log is not None:
        for key, value in ret.items():
            log.add_scalar(prefix + "_" + key, value, epoch)
