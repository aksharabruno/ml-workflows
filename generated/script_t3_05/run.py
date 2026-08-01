from dependency import *  # noqa: F401,F403

from task_01_model_generation.task_01_model_generation import model_generation_1
num_labels = model_generation_1()
from task_02_data_preparation.task_02_data_preparation import data_preparation_2
candidate_label_cols, candidate_text_cols, ds, train_columns = data_preparation_2()
from task_03_feature_engineering.task_03_feature_engineering import feature_engineering_3
label_col, text_col = feature_engineering_3(candidate_label_cols, candidate_text_cols, train_columns)
from task_04_data_preparation.task_04_data_preparation import data_preparation_4
data_preparation_4(label_col, text_col, train_columns)
from task_05_feature_engineering.task_05_feature_engineering import feature_engineering_5
feature_engineering_5(ds, label_col, text_col, train_columns)
from task_06_data_preparation.task_06_data_preparation import data_preparation_6
data_collator, eval_dataset, train_dataset = data_preparation_6(label_col)
from task_07_model_generation.task_07_model_generation import model_generation_7
model, optimizer, training_args = model_generation_7(num_labels)
from task_08_model_evaluation.task_08_model_evaluation import model_evaluation_8
model_evaluation_8()
from task_09_model_generation.task_09_model_generation import model_generation_9
trainer = model_generation_9(data_collator, eval_dataset, model, optimizer, train_dataset, training_args)
from task_10_model_evaluation.task_10_model_evaluation import model_evaluation_10
eval_res = model_evaluation_10(trainer)
from task_11_model_generation.task_11_model_generation import model_generation_11
model_generation_11(eval_res, model)
