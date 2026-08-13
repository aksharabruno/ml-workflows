from dependency import *  # noqa: F401,F403

from task_01_model_generation.task_01_model_generation import model_generation_1
model_name, num_labels = model_generation_1()
from task_02_data_preparation.task_02_data_preparation import data_preparation_2
ds, label_col, text_col, train_columns = data_preparation_2()
from task_03_feature_engineering.task_03_feature_engineering import feature_engineering_3
tokenized = feature_engineering_3(ds, label_col, text_col, train_columns)
from task_04_data_preparation.task_04_data_preparation import data_preparation_4
data_collator, eval_dataset, train_dataset = data_preparation_4(tokenized)
from task_05_model_generation.task_05_model_generation import model_generation_5
model, optimizer, training_args = model_generation_5(model_name, num_labels)
from task_06_model_evaluation.task_06_model_evaluation import model_evaluation_6
model_evaluation_6()
from task_07_model_generation.task_07_model_generation import model_generation_7
trainer = model_generation_7(data_collator, eval_dataset, model, optimizer, train_dataset, training_args)
from task_08_model_generation.task_08_model_generation import model_generation_8
model_generation_8(trainer)
from task_09_model_evaluation.task_09_model_evaluation import model_evaluation_9
model_evaluation_9(trainer)
from task_10_model_generation.task_10_model_generation import model_generation_10
model_generation_10(model)
