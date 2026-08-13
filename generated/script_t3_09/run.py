from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
test_loader, train_loader = data_preparation_1()
from task_02_model_generation.task_02_model_generation import model_generation_2
criterion, model = model_generation_2(test_loader, train_loader)
from task_03_model_evaluation.task_03_model_evaluation import model_evaluation_3
model_evaluation_3(criterion, model, test_loader)
