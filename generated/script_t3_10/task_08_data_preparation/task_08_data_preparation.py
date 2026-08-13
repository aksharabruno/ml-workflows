from dependency import *  # noqa: F401,F403


def data_preparation_8(X_train_embeddings, smote, y_train):

    # Apply SMOTE to balance the full dataset (if needed)
    print("Applying SMOTE to full dataset...")
    X_train_res, y_train_res = smote.fit_resample(X_train_embeddings, y_train)

    return X_train_res, y_train_res
