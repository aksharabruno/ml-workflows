from dependency import *  # noqa: F401,F403

from task_01_model_generation.task_01_model_generation import model_generation_1
model_generation_1()
from task_02_data_preparation.task_02_data_preparation import data_preparation_2
X_test, X_train, X_val, y_test, y_train, y_val = data_preparation_2()
from task_03_model_generation.task_03_model_generation import model_generation_3
best_model_name, best_pipeline, best_score, model_name, pipeline, results = model_generation_3(X_train, y_train)
from task_04_model_evaluation.task_04_model_evaluation import model_evaluation_4
test_metrics, val_metrics = model_evaluation_4(X_test, X_val, pipeline, y_test, y_val)
from task_05_model_generation.task_05_model_generation import model_generation_5
model_generation_5(model_name, pipeline)
from task_06_model_evaluation.task_06_model_evaluation import model_evaluation_6
model_evaluation_6(X_test, model_name, pipeline, test_metrics, val_metrics, y_test)
from task_07_model_generation.task_07_model_generation import model_generation_7
model_generation_7(model_name, pipeline)
from task_08_model_evaluation.task_08_model_evaluation import model_evaluation_8
best_model_name, best_pipeline, best_score = model_evaluation_8(model_name, pipeline, results, test_metrics, val_metrics)
from task_09_model_generation.task_09_model_generation import model_generation_9
model_generation_9(best_pipeline)
from task_10_model_evaluation.task_10_model_evaluation import model_evaluation_10
model_evaluation_10(best_model_name, best_score, results)
