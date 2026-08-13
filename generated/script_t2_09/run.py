from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
labels, texts = data_preparation_1()
from task_02_feature_engineering.task_02_feature_engineering import feature_engineering_2
X_ids, X_mask, X_type = feature_engineering_2(texts)
from task_03_data_preparation.task_03_data_preparation import data_preparation_3
X_test_ids, X_test_mask, X_test_type, X_train_ids, X_train_mask, X_train_type, y_test, y_train = data_preparation_3(X_ids, X_mask, X_type, labels, texts)
from task_04_model_generation.task_04_model_generation import model_generation_4
history, model = model_generation_4(X_train_ids, X_train_mask, X_train_type, y_train)
from task_05_model_evaluation.task_05_model_evaluation import model_evaluation_5
model_evaluation_5(X_test_ids, X_test_mask, X_test_type, history, model, y_test)
