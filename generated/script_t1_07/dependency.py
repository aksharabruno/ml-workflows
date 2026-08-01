# logistic_regression.py
"""
Logistic Regression Module
Demonstrates train/test split, binary classification model training, accuracy evaluation, and classification reporting.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "logistic_data.csv")

