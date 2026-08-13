from dependency import *  # noqa: F401,F403


def data_preparation_3(labels, texts):
    # Split stratifié : conserve la distribution des classes dans chaque split
    # stratify=labels garantit que chaque classe est représentée proportionnellement
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    print(f"[main] Split → Train : {len(X_train)} | Val : {len(X_val)}")

    return X_train, X_val, y_train, y_val
