from dependency import *  # noqa: F401,F403

from task_01_data_preparation.task_01_data_preparation import data_preparation_1
embeddings_index, labels, texts = data_preparation_1()
from task_02_feature_engineering.task_02_feature_engineering import feature_engineering_2
MAX_NUM_WORDS, data, word_index = feature_engineering_2(labels, texts)
from task_03_data_preparation.task_03_data_preparation import data_preparation_3
test_loader, train_loader, validation_loader = data_preparation_3(data, labels)
from task_04_feature_engineering.task_04_feature_engineering import feature_engineering_4
embedding_matrix = feature_engineering_4(MAX_NUM_WORDS, embeddings_index, word_index)
from task_05_model_generation.task_05_model_generation import model_generation_5
criterion, model = model_generation_5(embedding_matrix, train_loader, validation_loader)
from task_06_model_evaluation.task_06_model_evaluation import model_evaluation_6
model_evaluation_6(criterion, model, test_loader)
