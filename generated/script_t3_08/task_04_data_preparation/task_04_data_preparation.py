from dependency import *  # noqa: F401,F403


def data_preparation_4():
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)

    print("Data path :", args.data_path)
    print("Model     :", args.model)
    print("Seed      :", args.seed)



    # Load data


    df = load_data(
        args.data_path
    )



    # EDA


    perform_eda(df)



    # Preprocessing + split


    X_train, X_test, y_train, y_test, label_encoder = preprocess(
        df,
        args.seed
    )

    return X_test, X_train, label_encoder, y_test, y_train
