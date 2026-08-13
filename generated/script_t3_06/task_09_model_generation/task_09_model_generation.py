from dependency import *  # noqa: F401,F403


def model_generation_9(best_pipeline):
    if best_pipeline is None:
        raise RuntimeError("Training failed to produce a model")

    joblib.dump(best_pipeline, MODELS_DIR / "best_model.joblib")
    feature_names = list(X.columns)
    with (MODELS_DIR / "feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2)

