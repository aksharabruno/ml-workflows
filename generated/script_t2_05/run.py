from dependency import *  # noqa: F401,F403

from task_01_model_generation.task_01_model_generation import model_generation_1
model_generation_1()
from task_02_data_preparation.task_02_data_preparation import data_preparation_2
test_tf = data_preparation_2()
from task_03_model_generation.task_03_model_generation import model_generation_3
model = model_generation_3()
from task_04_data_preparation.task_04_data_preparation import data_preparation_4
train_loader, val_loader = data_preparation_4()
from task_05_model_generation.task_05_model_generation import model_generation_5
model = model_generation_5(train_loader, val_loader)
from task_06_data_preparation.task_06_data_preparation import data_preparation_6
img = data_preparation_6(test_tf)
from task_07_model_evaluation.task_07_model_evaluation import model_evaluation_7
model_evaluation_7(img, model)
from task_08_data_preparation.task_08_data_preparation import data_preparation_8
img = data_preparation_8()
from task_09_model_evaluation.task_09_model_evaluation import model_evaluation_9
model_evaluation_9(img)
