from dependency import *  # noqa: F401,F403


def feature_engineering_4(df):
    # Preprocessing + split


    X_train, X_test, y_train, y_test, label_encoder = preprocess(
        df,
        args.seed
    )

    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples :", X_test.shape[0])



    return X_test, X_train, label_encoder, y_test, y_train
