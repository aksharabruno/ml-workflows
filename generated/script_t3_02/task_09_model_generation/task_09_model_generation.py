from dependency import *  # noqa: F401,F403


def model_generation_9(pipe, y_train, y_train_c):
    model_path = out_dir / "tmdb_rating_pipeline.joblib"
    joblib.dump(pipe, model_path)

    meta: dict[str, Any] = {
        "label_scheme": label_scheme if task == "classification" else "regression",
        "task": task,
        "text_features": text_features,
    }
    if task == "classification":
        meta["class_to_mean_vote"] = _class_to_mean_vote(y_train, y_train_c)
        meta["classes"] = [int(c) for c in pipe.classes_]
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return model_path
