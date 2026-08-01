from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
embeddings_index, labels, texts = data_preparation_1()
from task_02_feature_engineering.task_02_feature_engineering import feature_engineering_2
MAX_NUM_WORDS, tokens = feature_engineering_2(texts)
from task_03_data_preparation.task_03_data_preparation import data_preparation_3
MAX_SEQUENCE_LENGTH = data_preparation_3()
from task_04_feature_engineering.task_04_feature_engineering import feature_engineering_4
word_index = feature_engineering_4(MAX_NUM_WORDS, tokens)
from task_05_data_preparation.task_05_data_preparation import data_preparation_5
test_dataset, test_loader, train_loader, validation_loader = data_preparation_5(MAX_SEQUENCE_LENGTH, labels)
from task_06_feature_engineering.task_06_feature_engineering import feature_engineering_6
embedding_matrix, n_not_found = feature_engineering_6(MAX_NUM_WORDS, embeddings_index, test_dataset, word_index)
from task_07_model_generation.task_07_model_generation import model_generation_7
criterion, end_time, model, start_time = model_generation_7(embedding_matrix, n_not_found, train_loader, validation_loader)
from task_08_model_evaluation.task_08_model_evaluation import model_evaluation_8
model_evaluation_8(criterion, end_time, model, start_time, test_loader)
