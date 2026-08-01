from dependency import *  # noqa: F401,F403

from task_01_model_generation.task_01_model_generation import model_generation_1
num_classes = model_generation_1()
from task_02_data_preparation.task_02_data_preparation import data_preparation_2
input_shape, x_test, x_train, y_test, y_train = data_preparation_2(num_classes)
from task_03_model_generation.task_03_model_generation import model_generation_3
model = model_generation_3(input_shape, num_classes, x_train, y_train)
from task_04_model_evaluation.task_04_model_evaluation import model_evaluation_4
model_evaluation_4(model, x_test, y_test)
