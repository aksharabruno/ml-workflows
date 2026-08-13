from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
transformer, x_test, x_train, y_test, y_train = data_preparation_1()
from task_02_model_generation.task_02_model_generation import model_generation_2
lr = model_generation_2(x_train, y_train)
from task_03_model_evaluation.task_03_model_evaluation import model_evaluation_3
test_data_prediction = model_evaluation_3(lr, x_test, x_train, y_test, y_train)
from task_04_model_generation.task_04_model_generation import model_generation_4
ls = model_generation_4(x_train, y_train)
from task_05_model_evaluation.task_05_model_evaluation import model_evaluation_5
model_evaluation_5(ls, x_train, y_train)
from task_06_model_generation.task_06_model_generation import model_generation_6
model_generation_6(ls, x_test, y_test)
from task_07_model_evaluation.task_07_model_evaluation import model_evaluation_7
test_data_prediction = model_evaluation_7(ls, x_test, y_test)
from task_08_model_evaluation.task_08_model_evaluation import model_evaluation_8
model_evaluation_8(lr, test_data_prediction, x_test, y_test)
from task_09_model_evaluation.task_09_model_evaluation import model_evaluation_9
model_evaluation_9(lr, transformer)
