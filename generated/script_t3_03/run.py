from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
base_dir, downloadedfile = data_preparation_1()
from task_02_data_preparation.task_02_data_preparation import data_preparation_2
classes, dataset_path, datasetz = data_preparation_2(base_dir, downloadedfile)
from task_03_model_generation.task_03_model_generation import model_generation_3
n_class = model_generation_3(datasetz)
from task_04_data_preparation.task_04_data_preparation import data_preparation_4
TestLoader, train_loader, transformNorm, val_loader = data_preparation_4(classes, dataset_path, datasetz)
from task_05_model_generation.task_05_model_generation import model_generation_5
loss_fn, modelnew, optimizer = model_generation_5(n_class)
from task_06_model_generation.task_06_model_generation import model_generation_6
scheduler = model_generation_6(optimizer)
from task_07_model_generation.task_07_model_generation import model_generation_7
model_generation_7()
from task_08_model_generation.task_08_model_generation import model_generation_8
model_generation_8(loss_fn, modelnew, optimizer, scheduler, train_loader, val_loader)
from task_09_model_evaluation.task_09_model_evaluation import model_evaluation_9
predictions = model_evaluation_9(TestLoader, modelnew)
from task_10_model_evaluation.task_10_model_evaluation import model_evaluation_10
targets_val = model_evaluation_10(TestLoader, classes, predictions)
from task_11_model_evaluation.task_11_model_evaluation import model_evaluation_11
model_evaluation_11(classes, modelnew, predictions, targets_val, transformNorm)
