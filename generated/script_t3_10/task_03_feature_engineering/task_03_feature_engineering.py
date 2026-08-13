from dependency import *  # noqa: F401,F403


def feature_engineering_3(X_test_subset, X_train_subset, model):
    print("Extracting BERT embeddings for subset...")
    X_train_subset_embeddings = get_bert_embeddings(X_train_subset.tolist(), tokenizer, model)
    X_test_subset_embeddings = get_bert_embeddings(X_test_subset.tolist(), tokenizer, model)
    return X_train_subset_embeddings
