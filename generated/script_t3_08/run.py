from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
df = data_preparation_1()
from task_02_feature_engineering.task_02_feature_engineering import feature_engineering_2
X = feature_engineering_2(df)
from task_03_data_preparation.task_03_data_preparation import data_preparation_3
X_test, X_train, label_encoder, y_test, y_train = data_preparation_3(X, df)
from task_04_data_preparation.task_04_data_preparation import data_preparation_4
X_test, X_train, label_encoder, y_test, y_train = data_preparation_4()
from task_05_model_generation.task_05_model_generation import model_generation_5
model = model_generation_5(X_test, X_train, y_train)
from task_06_model_evaluation.task_06_model_evaluation import model_evaluation_6
model_evaluation_6(X_test, model, y_test)
from task_07_model_generation.task_07_model_generation import model_generation_7
model_generation_7(label_encoder, model)
