from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
red_wine, white_wine = data_preparation_1()
from task_02_feature_engineering.task_02_feature_engineering import feature_engineering_2
feature_engineering_2(red_wine, white_wine)
from task_03_data_preparation.task_03_data_preparation import data_preparation_3
X_test_scaled, X_train_scaled, y, y_test, y_train = data_preparation_3(red_wine, white_wine)
from task_04_model_generation.task_04_model_generation import model_generation_4
log_model = model_generation_4(X_train_scaled, y_train)
from task_05_model_evaluation.task_05_model_evaluation import model_evaluation_5
model_evaluation_5(X_test_scaled, log_model, y, y_test)
