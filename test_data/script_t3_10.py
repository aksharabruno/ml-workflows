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

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

# Load the dataset
file_path = "MultiClass_CyberBully_Dataset.xlsx"
df = pd.read_excel(file_path)

# Text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # remove special characters
    text = text.lower()  # convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # remove stopwords
    return text

print("Cleaning text data...")
df['tweet_text'] = df['tweet_text'].apply(clean_text)

# Encode the labels
print("Encoding labels...")
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['cyberbullying_type'])

# Use a smaller subset for hyperparameter tuning
df_subset = df.sample(n=2000, random_state=42)

# Split the dataset for tuning
X_subset = df_subset['tweet_text']
y_subset = df_subset['label']
X_train_subset, X_test_subset, y_train_subset, y_test_subset = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

# Load BERT tokenizer and model
print("Loading BERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to extract BERT embeddings
def get_bert_embeddings(text_list, tokenizer, model):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in text_list:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())  # Use the CLS token representation
    return np.array(embeddings)

print("Extracting BERT embeddings for subset...")
X_train_subset_embeddings = get_bert_embeddings(X_train_subset.tolist(), tokenizer, model)
X_test_subset_embeddings = get_bert_embeddings(X_test_subset.tolist(), tokenizer, model)

# Apply SMOTE to balance the subset dataset (if needed)
print("Applying SMOTE to subset...")
smote = SMOTE(random_state=42)
X_train_subset_res, y_train_subset_res = smote.fit_resample(X_train_subset_embeddings, y_train_subset)

# Hyperparameter tuning for SVM with RBF and Polynomial kernels
print("Tuning SVM hyperparameters...")
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['rbf', 'poly'],
    'svc__gamma': ['scale', 'auto'],
    'svc__degree': [3, 4, 5]  # Only relevant for polynomial kernel
}
svm_model = make_pipeline(StandardScaler(), SVC(decision_function_shape='ovr', probability=True, random_state=42))
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_subset_res, y_train_subset_res)

best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Split the full dataset into training and testing sets
X = df['tweet_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract BERT embeddings for the full dataset
print("Extracting BERT embeddings for full dataset...")
X_train_embeddings = get_bert_embeddings(X_train.tolist(), tokenizer, model)
X_test_embeddings = get_bert_embeddings(X_test.tolist(), tokenizer, model)

# Apply SMOTE to balance the full dataset (if needed)
print("Applying SMOTE to full dataset...")
X_train_res, y_train_res = smote.fit_resample(X_train_embeddings, y_train)

# Extract the best parameters without the 'svc__' prefix
best_params = {k.replace('svc__', ''): v for k, v in grid_search.best_params_.items()}

print("Best parameters found: ", best_params)

# Train the final SVM classifier with the best parameters
print("Training the final SVM classifier...")
final_svm_model = make_pipeline(StandardScaler(), SVC(decision_function_shape='ovr', probability=True, random_state=42, **best_params))
final_svm_model.fit(X_train_res, y_train_res)

# Predict on the test set
print("Predicting on the test set...")
y_pred = final_svm_model.predict(X_test_embeddings)
y_pred_proba = final_svm_model.predict_proba(X_test_embeddings)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate classification report
classification_rep = classification_report(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7], digits=4)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_rep)

# Save the model and label encoder
print("Saving the model and label encoder...")
joblib.dump(final_svm_model, 'best_svm_cyberbullying_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Calculate the AUC-ROC curve
print("Calculating the AUC-ROC curve...")
fpr = {}
tpr = {}
roc_auc = {}
for i in range(len(label_encoder.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the AUC-ROC curve
plt.figure(figsize=(10, 8))
for i in range(len(label_encoder.classes_)):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC-ROC Curve for SVM')
plt.legend(loc="lower right")
plt.show()

# Generate and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[i for i in range(len(label_encoder.classes_))])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for SVM')
plt.show()

# Output additional metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

print("Done!")

 