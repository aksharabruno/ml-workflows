from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
transform = data_preparation_1()
from task_02_data_preparation.task_02_data_preparation import data_preparation_2
dataloader_train, dataloader_val, dataset_train = data_preparation_2(transform)
from task_03_model_generation.task_03_model_generation import model_generation_3
model_generation_3(dataloader_train, dataloader_val, dataset_train)
