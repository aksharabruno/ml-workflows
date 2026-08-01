from dependency import *  # noqa: F401,F403

from task_01_model_generation.task_01_model_generation import model_generation_1
iteration_number, loss_list, model, mse, optimizer = model_generation_1()
from task_02_data_preparation.task_02_data_preparation import data_preparation_2
car_price_tensor, number_of_car_sell_tensor = data_preparation_2()
from task_03_model_generation.task_03_model_generation import model_generation_3
model_generation_3(car_price_tensor, iteration_number, loss_list, model, mse, number_of_car_sell_tensor, optimizer)
from task_04_model_evaluation.task_04_model_evaluation import model_evaluation_4
model_evaluation_4(iteration_number, loss_list)
