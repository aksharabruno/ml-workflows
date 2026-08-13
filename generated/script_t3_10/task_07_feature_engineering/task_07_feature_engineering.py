from dependency import *  # noqa: F401,F403


def feature_engineering_7(X_test, X_train, model):
    # Extract BERT embeddings for the full dataset
    print("Extracting BERT embeddings for full dataset...")
    X_train_embeddings = get_bert_embeddings(X_train.tolist(), tokenizer, model)
    X_test_embeddings = get_bert_embeddings(X_test.tolist(), tokenizer, model)
    return X_test_embeddings, X_train_embeddings
