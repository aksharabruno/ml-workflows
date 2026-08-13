from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
label_names, labels, texts = data_preparation_1()
from task_02_model_generation.task_02_model_generation import model_generation_2
num_labels = model_generation_2(label_names)
from task_03_data_preparation.task_03_data_preparation import data_preparation_3
X_train, X_val, y_train, y_val = data_preparation_3(labels, texts)
from task_04_data_preparation.task_04_data_preparation import data_preparation_4
train_loader, val_loader = data_preparation_4(X_train, X_val, y_train, y_val)
from task_05_model_generation.task_05_model_generation import model_generation_5
best_model_path, final_labels, final_preds, history = model_generation_5(label_names, num_labels, train_loader, val_loader)
from task_06_model_evaluation.task_06_model_evaluation import model_evaluation_6
model_evaluation_6(best_model_path, final_labels, final_preds, history, label_names)
