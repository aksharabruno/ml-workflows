from dependency import *  # noqa: F401,F403


def model_generation_5(X_test, X_train, y_train):
    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples :", X_test.shape[0])



    # Build models


    models = build_models()



    # Select requested model


    model = models[args.model]



    # Train


    print(f"\nTraining: {args.model}")

    model.fit(
        X_train,
        y_train
    )



    return model
