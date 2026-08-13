from dependency import *  # noqa: F401,F403


def model_evaluation_6(X_test, label_encoder, model, y_test):
    # Evaluate


    score = evaluate(
        args.model,
        model,
        X_test,
        y_test
    )



    # Save model


    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = f"models/{args.model}.pkl"

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        label_encoder,
        "models/label_encoder.pkl"
    )

    print("\nTraining completed.")
    print("Model saved to:", model_path)
    print("Accuracy:", score)
