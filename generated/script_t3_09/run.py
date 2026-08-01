from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
test_dataset, test_loader, train_dataset, train_loader = data_preparation_1()
from task_02_model_generation.task_02_model_generation import model_generation_2
best_acc, criterion, model, test_accs, test_losses, train_accs, train_losses = model_generation_2(test_dataset, test_loader, train_dataset, train_loader)
from task_03_model_evaluation.task_03_model_evaluation import model_evaluation_3
model_evaluation_3(best_acc, criterion, model, test_accs, test_loader, test_losses, train_accs, train_losses)
