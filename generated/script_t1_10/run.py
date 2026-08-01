from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
df = data_preparation_1()
from task_02_feature_engineering.task_02_feature_engineering import feature_engineering_2
X = feature_engineering_2(df)
from task_03_data_preparation.task_03_data_preparation import data_preparation_3
X_test, X_train, y_test, y_train = data_preparation_3(X, df)
from task_04_model_generation.task_04_model_generation import model_generation_4
model = model_generation_4(X_train, y_train)
from task_05_model_evaluation.task_05_model_evaluation import model_evaluation_5
y_prob = model_evaluation_5(X_test, model, y_test)
from task_06_model_generation.task_06_model_generation import model_generation_6
custom_prediction, threshold = model_generation_6(y_prob)
from task_07_model_evaluation.task_07_model_evaluation import model_evaluation_7
model_evaluation_7(custom_prediction, threshold)
