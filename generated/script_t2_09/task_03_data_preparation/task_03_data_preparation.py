from dependency import *  # noqa: F401,F403


def data_preparation_3(X_ids, X_mask, X_type, labels, texts):
    # 4. Stratified Split (80/20 train/test)
    print("Step 4: Creating train/test split...")
    rng = np.random.RandomState(42)
    indices = np.arange(len(texts))

    train_idx, test_idx = [], []
    for c in range(len(CLASSES)):
        class_indices = rng.permutation(np.where(labels == c)[0])
        split_pt = int(len(class_indices) * 0.8)
        train_idx.extend(class_indices[:split_pt])
        test_idx.extend(class_indices[split_pt:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    X_train_ids, X_test_ids = X_ids[train_idx], X_ids[test_idx]
    X_train_mask, X_test_mask = X_mask[train_idx], X_mask[test_idx]
    X_train_type, X_test_type = X_type[train_idx], X_type[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    return X_test_ids, X_test_mask, X_test_type, X_train_ids, X_train_mask, X_train_type, test_idx, train_idx, y_test, y_train
