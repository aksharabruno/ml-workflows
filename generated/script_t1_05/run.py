from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
home_data, y = data_preparation_1()
from task_02_feature_engineering.task_02_feature_engineering import feature_engineering_2
X = feature_engineering_2(home_data)
from task_03_data_preparation.task_03_data_preparation import data_preparation_3
train_X, train_y, val_X, val_y = data_preparation_3(X, y)
from task_04_model_generation.task_04_model_generation import model_generation_4
rf_model = model_generation_4(train_X, train_y)
from task_05_model_evaluation.task_05_model_evaluation import model_evaluation_5
model_evaluation_5(rf_model, val_X, val_y)
