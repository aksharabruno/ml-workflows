from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
class_names, test_loader, train_loader, val_loader = data_preparation_1()
from task_02_model_generation.task_02_model_generation import model_generation_2
history, model, n_params = model_generation_2(class_names, train_loader, val_loader)
from task_03_model_evaluation.task_03_model_evaluation import model_evaluation_3
model_evaluation_3(class_names, history, model, n_params, test_loader)
