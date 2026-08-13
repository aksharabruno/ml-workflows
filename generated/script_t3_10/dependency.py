import pandas as pd
import numpy as np
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import torch


# Load BERT tokenizer and model
print("Loading BERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')




def get_bert_embeddings(text_list, tokenizer, model):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in text_list:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())  # Use the CLS token representation
    return np.array(embeddings)
