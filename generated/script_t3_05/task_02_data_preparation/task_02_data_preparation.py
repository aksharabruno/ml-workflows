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

    return candidate_label_cols, candidate_text_cols, ds, train_columns
