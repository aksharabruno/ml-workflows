from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
data_preparation_1()
from task_02_data_preparation.task_02_data_preparation import data_preparation_2
X_test, X_train, y_test = data_preparation_2()
from task_03_model_generation.task_03_model_generation import model_generation_3
autoencoder, batch_size, checkpointer, nb_epoch = model_generation_3(X_train)
from task_04_model_generation.task_04_model_generation import model_generation_4
autoencoder, history = model_generation_4(X_test, X_train, batch_size, checkpointer, nb_epoch)
from task_05_model_evaluation.task_05_model_evaluation import model_evaluation_5
model_evaluation_5(X_test, autoencoder, history, y_test)
