from dependency import *  # noqa: F401,F403


def data_preparation_2():
    print("Loading dataset:", args.dataset_name)
    ds = load_dataset(args.dataset_name)

    print(ds)

    # ---------------------------------------------------------
    # ✅ Robust column detection
    # ---------------------------------------------------------
    candidate_text_cols = ["text", "review", "sentence", "content"]
    candidate_label_cols = ["label", "labels", "sentiment", "target"]

    train_columns = ds["train"].column_names

    text_col = next((c for c in candidate_text_cols if c in train_columns), None)
    label_col = next((c for c in candidate_label_cols if c in train_columns), None)

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not infer text/label columns. Found columns: {train_columns}"
        )

    print(f"Using text column: '{text_col}', label column: '{label_col}'")

    return ds, label_col, text_col, train_columns
