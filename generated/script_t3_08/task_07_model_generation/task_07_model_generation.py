from dependency import *  # noqa: F401,F403


def model_generation_7(label_encoder, model):
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

