from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
X_test, X_train, y_test, y_train = data_preparation_1()
from task_02_model_generation.task_02_model_generation import model_generation_2
model = model_generation_2(X_train, y_train)
from task_03_model_evaluation.task_03_model_evaluation import model_evaluation_3
mae, r2 = model_evaluation_3(X_test, model, y_test)
from task_04_data_preparation.task_04_data_preparation import data_preparation_4
X, mae, model, r2 = data_preparation_4()
from task_05_model_evaluation.task_05_model_evaluation import model_evaluation_5
model_evaluation_5(X, mae, model, r2)
