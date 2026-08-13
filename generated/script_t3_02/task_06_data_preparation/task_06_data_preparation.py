from dependency import *  # noqa: F401,F403


def data_preparation_6(X, y):
    y_cont = y.to_numpy(dtype=float)
    n = len(X)
    indices = np.arange(n)

    strat = None
    if task == "classification" and label_scheme == "exact":
        y_cls_full = np.clip(np.round(y_cont), 0, 10).astype(np.int64)
        if n >= 30 and int(np.bincount(y_cls_full, minlength=11).min()) >= 2:
            strat = y_cls_full

    tr_idx, te_idx = train_test_split(indices, test_size=test_size, random_state=42, stratify=strat)
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train_c, y_test_c = y_cont[tr_idx], y_cont[te_idx]

    if task == "classification" and label_scheme == "tier3":
        y_train = _quantile_labels(y_train_c, y_train_c, q=3)
        y_test = _quantile_labels(y_train_c, y_test_c, q=3)
    elif task == "classification":
        y_train = np.clip(np.round(y_train_c), 0, 10).astype(np.int64)
        y_test = np.clip(np.round(y_test_c), 0, 10).astype(np.int64)
    else:
        y_train, y_test = y_train_c, y_test_c

    return X_test, X_train, y_test, y_train, y_train_c
