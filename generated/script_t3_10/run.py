from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
X_test_subset, X_train_subset, df, label_encoder, y_train_subset = data_preparation_1()
from task_02_model_generation.task_02_model_generation import model_generation_2
model = model_generation_2()
from task_03_feature_engineering.task_03_feature_engineering import feature_engineering_3
X_train_subset_embeddings = feature_engineering_3(X_test_subset, X_train_subset, model)
from task_04_data_preparation.task_04_data_preparation import data_preparation_4
X_train_subset_res, smote, y_train_subset_res = data_preparation_4(X_train_subset_embeddings, y_train_subset)
from task_05_model_generation.task_05_model_generation import model_generation_5
best_params, grid_search = model_generation_5(X_train_subset_res, y_train_subset_res)
from task_06_data_preparation.task_06_data_preparation import data_preparation_6
X_test, X_train, y_test, y_train = data_preparation_6(best_params, df)
from task_07_feature_engineering.task_07_feature_engineering import feature_engineering_7
X_test_embeddings, X_train_embeddings = feature_engineering_7(X_test, X_train, model)
from task_08_data_preparation.task_08_data_preparation import data_preparation_8
X_train_res, y_train_res = data_preparation_8(X_train_embeddings, smote, y_train)
from task_09_model_generation.task_09_model_generation import model_generation_9
final_svm_model = model_generation_9(X_train_res, grid_search, y_train_res)
from task_10_model_evaluation.task_10_model_evaluation import model_evaluation_10
accuracy, classification_rep, y_pred, y_pred_proba = model_evaluation_10(X_test_embeddings, final_svm_model, y_test)
from task_11_model_generation.task_11_model_generation import model_generation_11
model_generation_11(accuracy, classification_rep, final_svm_model, label_encoder)
from task_12_model_evaluation.task_12_model_evaluation import model_evaluation_12
model_evaluation_12(label_encoder, y_pred, y_pred_proba, y_test)
