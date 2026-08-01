from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
x_test_scaled, x_train_scaled, y_test, y_train = data_preparation_1()
from task_02_model_generation.task_02_model_generation import model_generation_2
model = model_generation_2(x_train_scaled, y_train)
from task_03_model_evaluation.task_03_model_evaluation import model_evaluation_3
model_evaluation_3(model, x_test_scaled, y_test)
