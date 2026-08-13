from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
X_test, lgb_eval, lgb_train, y_test = data_preparation_1()
from task_02_model_generation.task_02_model_generation import model_generation_2
gbm = model_generation_2(lgb_eval, lgb_train)
from task_03_model_evaluation.task_03_model_evaluation import model_evaluation_3
model_evaluation_3(X_test, gbm, y_test)
