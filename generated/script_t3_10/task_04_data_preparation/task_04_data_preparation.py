from dependency import *  # noqa: F401,F403


def data_preparation_4(X_train_subset_embeddings, y_train_subset):
    # Apply SMOTE to balance the subset dataset (if needed)
    print("Applying SMOTE to subset...")
    smote = SMOTE(random_state=42)
    X_train_subset_res, y_train_subset_res = smote.fit_resample(X_train_subset_embeddings, y_train_subset)

    return X_train_subset_res, smote, y_train_subset_res
